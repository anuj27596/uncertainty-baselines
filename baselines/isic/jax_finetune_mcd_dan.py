# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finetuning."""
import functools
import itertools
import multiprocessing
import numbers
import os
import time

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
from clu import preprocess_spec
import flax
import jax
import jax.numpy as jnp
import ml_collections.config_flags
import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.experimental.set_visible_devices([], 'TPU_SYSTEM')
tf.config.experimental.set_visible_devices([], 'TPU')

logging.info(tf.config.experimental.get_visible_devices())

# pylint: disable=g-import-not-at-top,line-too-long
import uncertainty_baselines as ub
# import checkpoint_utils  # local file import from baselines.isic  # EDIT(anuj)
# import input_utils  # local file import from baselines.isic  # EDIT(anuj)
# import preprocess_utils  # local file import from baselines.isic  # EDIT(anuj)
# import train_utils  # local file import from baselines.isic  # EDIT(anuj)
# from utils import results_storage_utils  # EDIT(anuj)
# from utils import vit_utils  # EDIT(anuj)
import baselines.isic.checkpoint_utils as checkpoint_utils  # EDIT(anuj)
import baselines.isic.input_utils as input_utils  # EDIT(anuj)
import baselines.isic.preprocess_utils as preprocess_utils  # EDIT(anuj)
import baselines.isic.train_utils as train_utils  # EDIT(anuj)
from baselines.isic.utils import results_storage_utils  # EDIT(anuj)
from baselines.isic.utils import vit_utils  # EDIT(anuj)
import wandb

# pylint: enable=g-import-not-at-top,line-too-long

# TODO(nband): lock config after separating total and warmup steps arguments.
ml_collections.config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=False)
# Set up extraneous flags for use in Googler job launching.
flags.DEFINE_string('output_dir', default=None, help='Work unit directory.')
flags.DEFINE_integer(
    'num_cores', default=None, help='Unused. How many devices being used.')
flags.DEFINE_boolean(
    'use_gpu', default=None, help='Unused. Whether or not running on GPU.')
flags.DEFINE_string('tpu', None,
                    'Unused. Name of the TPU. Only used if use_gpu is False.')
flags.DEFINE_integer('num_mc_samples', default=5, help='number of Monte Carlo samples')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg

  config = FLAGS.config

  # Unpack total and warmup steps
  # TODO(nband): revert this to separate arguments.
  total_steps = config.total_and_warmup_steps[0]
  warmup_steps = config.total_and_warmup_steps[1]
  del config.total_and_warmup_steps
  config.total_steps = total_steps
  config.lr.warmup_steps = warmup_steps

  # Wandb and Checkpointing Setup
  output_dir = FLAGS.output_dir
  config.output_dir = FLAGS.output_dir  # EDIT(anuj)
  wandb_run, output_dir = vit_utils.maybe_setup_wandb(config)
  tf.io.gfile.makedirs(output_dir)
  logging.info('Saving checkpoints at %s', output_dir)

  # Dataset Split Flags
  dist_shift = config.distribution_shift
  print(f'Distribution Shift: {dist_shift}.')
  dataset_names, split_names = vit_utils.get_dataset_and_split_names(dist_shift)

  # LR / Optimization Flags
  batch_size = config.batch_size
  grad_clip_norm = config.grad_clip_norm
  weight_decay = config.weight_decay
  print('wandb hyperparameters:')
  print({
      'batch_size': batch_size,
      'grad_clip_norm': grad_clip_norm,
      'weight_decay': weight_decay,
      'total_steps': config.total_steps,
      'lr': config.lr
  })

  # Reweighting loss for class imbalance
  # class_reweight_mode = config.class_reweight_mode
  # if class_reweight_mode == 'constant':
  #   class_weights = utils.get_diabetic_retinopathy_class_balance_weights()
  # else:
  #   class_weights = None

  if config.class_reweight_mode == 'constant':  # EDIT(anuj): class weighting
    class_weights = 0.5 * 15160 / jnp.array([int(0.95*15160), int(0.05*15160)])  # TODO(anuj): remove hardcode
    if config.loss == 'softmax_xent':
      base_loss_fn = train_utils.reweighted_softmax_xent(class_weights)
    else:
      raise NotImplementedError(f'loss `{config.loss}` not implemented for `constant` reweighting mode')
  else:
    base_loss_fn = getattr(train_utils, config.loss)
    
  # Shows the number of available devices.
  # In a CPU/GPU runtime this will be a single device.
  # In a TPU runtime this will be 8 cores.
  print('Number of Jax local devices:', jax.local_devices())

  # TODO(nband): fix sigmoid loss issues.
  # assert config.get('loss', None) == 'softmax_xent'  # EDIT(anuj)

  seed = config.get('seed', 0)
  rng = jax.random.PRNGKey(seed)
  tf.random.set_seed(seed)

  if config.get('data_dir'):
    logging.info('data_dir=%s', config.data_dir)
  logging.info('Output dir: %s', output_dir)
  tf.io.gfile.makedirs(output_dir)
  tf.io.gfile.makedirs(os.path.join(output_dir, 'checkpoints'))  # EDIT(anuj)

  save_checkpoint_path = None
  if config.get('checkpoint_steps'):
    save_checkpoint_path = os.path.join(output_dir, 'checkpoints', 'checkpoint.npz')  # EDIT(anuj)

  # Create an asynchronous multi-metric writer.
  writer = metric_writers.create_default_writer(
      output_dir, just_logging=jax.process_index() > 0)

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  def write_note(note):
    if jax.process_index() == 0:
      logging.info('NOTE: %s', note)

  write_note('Initializing...')

  # Verify settings to make sure no checkpoints are accidentally missed.
  if config.get('keep_checkpoint_steps'):
    assert config.get('checkpoint_steps'), 'Specify `checkpoint_steps`.'
    assert config.keep_checkpoint_steps % config.checkpoint_steps == 0, (
        f'`keep_checkpoint_steps` ({config.checkpoint_steps}) should be'
        f'divisible by `checkpoint_steps ({config.checkpoint_steps}).`')

  batch_size_eval = config.get('batch_size_eval', batch_size)
  if (batch_size % jax.device_count() != 0 or
      batch_size_eval % jax.device_count() != 0):
    raise ValueError(f'Batch sizes ({batch_size} and {batch_size_eval}) must '
                     f'be divisible by device number ({jax.device_count()})')

  local_batch_size = batch_size // jax.process_count()
  local_batch_size_eval = batch_size_eval // jax.process_count()
  logging.info(
      'Global batch size %d on %d hosts results in %d local batch size. '
      'With %d devices per host (%d devices total), that\'s a %d per-device '
      'batch size.', batch_size, jax.process_count(), local_batch_size,
      jax.local_device_count(), jax.device_count(),
      local_batch_size // jax.local_device_count())

  write_note('Initializing preprocessing function...')
  # Same preprocessing function for training and evaluation
  preproc_fn = preprocess_spec.parse(
      spec=config.pp_train, available_ops=preprocess_utils.all_ops())

  write_note('Initializing train dataset...')
  rng, train_ds_rng = jax.random.split(rng)
  train_ds_rng = jax.random.fold_in(train_ds_rng, jax.process_index())
  train_base_dataset = ub.datasets.get(
      dataset_names['in_domain_dataset'],
      split=split_names['train_split'],
      data_dir=config.get('data_dir'),
      builder_config=f'isic_id/{config.builder_config}')
  train_dataset_builder = train_base_dataset._dataset_builder  # pylint: disable=protected-access
  train_ds = input_utils.get_data(
      dataset=train_dataset_builder,
      split=split_names['train_split'],
      rng=train_ds_rng,
      process_batch_size=local_batch_size,
      preprocess_fn=preproc_fn,
      shuffle_buffer_size=config.shuffle_buffer_size,
      prefetch_size=config.get('prefetch_to_host', 2),
      data_dir=config.get('data_dir'))

  # Start prefetching already.
  train_iter = input_utils.start_input_pipeline(
      train_ds, config.get('prefetch_to_device', 1))

  rng, train_ood_ds_rng = jax.random.split(rng)
  train_ood_ds_rng = jax.random.fold_in(train_ood_ds_rng, jax.process_index())
  train_ood_base_dataset = ub.datasets.get(
      dataset_names['ood_dataset'],
      split=split_names['ood_validation_split'],
      data_dir=config.get('data_dir'),
      builder_config=f'isic_ood/{config.builder_config}')
  train_ood_dataset_builder = train_ood_base_dataset._dataset_builder  # pylint: disable=protected-access
  train_ood_ds = input_utils.get_data(
      dataset=train_ood_dataset_builder,
      split=split_names['ood_validation_split'],
      rng=train_ood_ds_rng,
      process_batch_size=local_batch_size,
      preprocess_fn=preproc_fn,
      shuffle_buffer_size=config.shuffle_buffer_size,
      prefetch_size=config.get('prefetch_to_host', 2),
      # percent=config.get('ood_val_percent'),
      data_dir=config.get('data_dir'))

  # Start prefetching already.
  train_ood_iter = input_utils.start_input_pipeline(
      train_ood_ds,
      config.get('prefetch_to_device', 1))

  write_note('Initializing val dataset(s)...')

  # Load in-domain and OOD validation and/or test datasets.
  # Please specify the desired shift (Country Shift or Severity Shift)
  # in the config.
  eval_iter_splits = vit_utils.init_evaluation_datasets(
      use_train=config.eval_on_train,  # EDIT(anuj)
      use_validation=config.use_validation,
      use_test=config.use_test,
      dataset_names=dataset_names,
      split_names=split_names,
      config=config,
      preproc_fn=preproc_fn,
      batch_size_eval=batch_size_eval,
      local_batch_size_eval=local_batch_size_eval)

  ntrain_img = input_utils.get_num_examples(
      train_dataset_builder,
      split=split_names['train_split'],
      process_batch_size=local_batch_size,
      data_dir=config.get('data_dir'))
  steps_per_epoch = ntrain_img // batch_size

  if config.get('num_epochs'):
    total_steps = int(config.num_epochs * steps_per_epoch)
    assert not config.get('total_steps'), 'Set either num_epochs or total_steps'
  else:
    total_steps = config.total_steps

  logging.info('Total train data points: %d', ntrain_img)
  logging.info(
      'Running for %d steps, that means %f epochs and %d steps per epoch',
      total_steps, total_steps * batch_size / ntrain_img, steps_per_epoch)

  write_note('Initializing model...')
  model_dict = vit_utils.initialize_model('dan', config)
  model = model_dict['model']

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def init(rng):
    image_size = tuple(train_ds.element_spec['image'].shape[2:])
    logging.info('image_size = %s', image_size)
    dummy_input = jnp.zeros((local_batch_size,) + image_size, jnp.float32)
    params = flax.core.unfreeze(model.init(rng, dummy_input,
                                           train=False))['params']

    # Set bias in the head to a low value, such that loss is small initially.
    params['head']['bias'] = jnp.full_like(
        params['head']['bias'], config.get('init_head_bias', 0))

    # init head kernel to all zeros for fine-tuning
    if config.get('model_init'):
      params['head']['kernel'] = jnp.full_like(params['head']['kernel'], 0)

    return params

  rng, rng_init = jax.random.split(rng)
  params_cpu = init(rng_init)

  if jax.process_index() == 0:
    num_params = sum(p.size for p in jax.tree_flatten(params_cpu)[0])
    parameter_overview.log_parameter_overview(params_cpu)
    writer.write_scalars(step=0, scalars={'num_params': num_params})

  @functools.partial(jax.pmap, axis_name='batch')
  def evaluation_fn(params, images, labels, rng):
    logits_list = []
    pre_list = []
    for _ in range(FLAGS.num_mc_samples):
      rng, rng_model = jax.random.split(rng, 2)
      rng_model_local = jax.random.fold_in(rng_model, jax.lax.axis_index('batch'))
      logits, out = model.apply(
          {'params': flax.core.freeze(params)}, images,
          train=True, rngs={'dropout': rng_model_local})
      logits_list.append(logits)
      pre_list.append(out['pre_logits'])

    logits = jnp.stack(logits_list, axis=-1)
    pre_list = jnp.stack(pre_list, axis=-2)
    probs = jnp.mean(jax.nn.softmax(logits, axis=-2), axis=-1)
    # import pdb; pdb.set_trace()
    losses = base_loss_fn(logits=logits, labels=labels, reduction=False) #-jnp.sum(labels * jnp.log(probs), axis=-1)

    loss = jax.lax.psum(losses, axis_name='batch')
    top1_idx = jnp.argmax(probs, axis=1)

    # Extracts the label at the highest logit index for each image.
    top1_correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]

    ncorrect = jax.lax.psum(top1_correct, axis_name='batch')
    n = batch_size_eval
    metric_args = jax.lax.all_gather([
        logits, labels, pre_list, out['domain_pred']], axis_name='batch')
    return ncorrect, loss, n, metric_args, rng

  # Load the optimizer from flax.
  opt_name = config.get('optim_name')
  write_note(f'Initializing {opt_name} optimizer...')
  opt_def = getattr(flax.optim, opt_name)(**config.get('optim', {}))

  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  opt_cpu = jax.jit(opt_def.create)(params_cpu)

  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
  def update_fn(opt, lr, images, labels, rng):
    """Update step."""
    measurements = {}

    # Get device-specific loss rng.
    rng, rng_model = jax.random.split(rng, 2)
    rng_model_local = jax.random.fold_in(rng_model, jax.lax.axis_index('batch'))

    def loss_fn(params, images, labels):
      images_id, images_ood = images
      logits, out_id = model.apply(
          {'params': flax.core.freeze(params)}, images_id,
          train=True, rngs={'dropout': rng_model_local})
      _, out_ood = model.apply(
          {'params': flax.core.freeze(params)}, images_ood,
          train=True, rngs={'dropout': rng_model_local})

      domain_pred = jnp.concatenate([
          out_id['domain_pred'],
          out_ood['domain_pred']])
      domain_labels = jnp.concatenate([
          jnp.zeros((*out_id['domain_pred'].shape[:-1], 1)),
          jnp.ones((*out_ood['domain_pred'].shape[:-1], 1))])

      domain_loss = train_utils.sigmoid_xent(logits=domain_pred, labels=domain_labels)
      loss = (
          base_loss_fn(logits=logits, labels=labels)
          + config.dp_loss_coeff * domain_loss)  # EDIT(anuj)
      return loss

    # Implementation considerations compared and summarized at
    # https://docs.google.com/document/d/1g3kMEvqu1DOawaflKNyUsIoQ4yIVEoyE5ZlIPkIl4Lc/edit?hl=en#
    l, g = train_utils.accumulate_gradient(
        jax.value_and_grad(loss_fn), opt.target, images, labels,
        config.get('grad_accum_steps'))
    l, g = jax.lax.pmean((l, g), axis_name='batch')

    # Log the gradient norm only if we need to compute it anyways (clipping)
    # or if we don't use grad_accum_steps, as they interact badly.
    if config.get('grad_accum_steps', 1) == 1 or grad_clip_norm is not None:
      grads, _ = jax.tree_flatten(g)
      l2_g = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads]))
      measurements['l2_grads'] = l2_g

    # Optionally resize the global gradient to a maximum norm. We found this
    # useful in some cases across optimizers, hence it's in the main loop.
    if grad_clip_norm is not None:
      g_factor = jnp.minimum(1.0, grad_clip_norm / l2_g)
      g = jax.tree_util.tree_map(lambda p: g_factor * p, g)
    opt = opt.apply_gradient(g, learning_rate=lr)

    decay_rules = weight_decay or []
    if isinstance(decay_rules, numbers.Number):
      decay_rules = [('.*kernel.*', decay_rules)]
    sched_m = lr / config.lr.base if config.get(
        'weight_decay_decouple') else lr

    def decay_fn(v, wd):
      return (1.0 - sched_m * wd) * v

    opt = opt.replace(
        target=train_utils.tree_map_with_regex(decay_fn, opt.target,
                                               decay_rules))

    params, _ = jax.tree_flatten(opt.target)
    measurements['l2_params'] = jnp.sqrt(
        sum([jnp.vdot(p, p) for p in params]))

    return opt, l, rng, measurements

  rng, train_loop_rngs = jax.random.split(rng)
  reint_params = ('head/kernel', 'head/bias')
  if config.get('only_eval', False) or not config.get('reint_head', True):
    reint_params = []
  checkpoint_data = checkpoint_utils.maybe_load_checkpoint(
      train_loop_rngs=train_loop_rngs,
      save_checkpoint_path=save_checkpoint_path,
      init_optimizer=opt_cpu,
      init_params=params_cpu,
      init_fixed_model_states=None,
      default_reinit_params=reint_params,
      config=config,
  )
  train_loop_rngs = checkpoint_data.train_loop_rngs
  opt_cpu = checkpoint_data.optimizer
  accumulated_train_time = checkpoint_data.accumulated_train_time

  write_note('Kicking off misc stuff...')
  first_step = int(opt_cpu.state.step)  # Might be a DeviceArray type.
  if first_step == 0 and jax.process_index() == 0:
    writer.write_hparams(dict(config))
  chrono = train_utils.Chrono(
      first_step, total_steps, batch_size, accumulated_train_time)
  # Note: switch to ProfileAllHosts() if you need to profile all hosts.
  # (Xprof data become much larger and take longer to load for analysis)
  profiler = periodic_actions.Profile(
      # Create profile after every restart to analyze pre-emption related
      # problems and assure we get similar performance in every run.
      logdir=output_dir, first_profile=first_step + 10)

  # Prepare the learning-rate and pre-fetch it to device to avoid delays.
  lr_fn = train_utils.create_learning_rate_schedule(total_steps,
                                                    **config.get('lr', {}))

  # TODO(dusenberrymw): According to flax docs, prefetching shouldn't be
  # necessary for TPUs.
  lr_iter = train_utils.prefetch_scalar(
      map(lr_fn, range(total_steps)), config.get('prefetch_to_device', 1))

  write_note(f'Replicating...\n{chrono.note}')
  opt_repl = flax.jax_utils.replicate(opt_cpu)

  checkpoint_writer = None

  # Note: we return the train loss, val loss, and fewshot best l2s for use in
  # reproducibility unit tests.
  # train_loss = -jnp.inf
  # val_loss = -jnp.inf
  # results = {'dummy': {(0, 1): -jnp.inf}}

  write_note(f'First step compilations...\n{chrono.note}')
  logging.info('first_step = %s', first_step)
  # Advance the iterators if we are restarting from an earlier checkpoint.
  # TODO(dusenberrymw): Look into checkpointing dataset state instead.
  if first_step > 0:
    write_note('Advancing iterators after resuming from a checkpoint...')
    lr_iter = itertools.islice(lr_iter, first_step, None)
    train_iter = itertools.islice(train_iter, first_step, None)
    train_ood_iter = itertools.islice(train_ood_iter, first_step, None)

  # Using a python integer for step here, because opt.state.step is allocated
  # on TPU during replication.
  for step, train_batch, train_ood_batch, lr_repl in zip(
      range(first_step + 1, total_steps + 1), train_iter, train_ood_iter, lr_iter):

    with jax.profiler.TraceAnnotation('train_step', step_num=step, _r=1):
      if not config.get('only_eval', False):
        if train_loop_rngs.shape[0] == 1 and train_loop_rngs.shape[0] < jax.device_count():  # EDIT(anuj): temp fix
          train_loop_rngs = jax.random.split(train_loop_rngs[0])
        opt_repl, loss_value, train_loop_rngs, extra_measurements = update_fn(
            opt_repl,
            lr_repl,
            (
              train_batch['image'],
              train_ood_batch['image'],
            ),
            train_batch['labels'],
            rng=train_loop_rngs)

    # if jax.process_index() == 0:  # EDIT(anuj)
    #   profiler(step)

    # Checkpoint saving
    if not config.get('only_eval', False) and train_utils.itstime(
        step, config.get('checkpoint_steps'), total_steps, process=0):
      write_note('Checkpointing...')
      chrono.pause()
      train_utils.checkpointing_timeout(checkpoint_writer,
                                        config.get('checkpoint_timeout', 1))
      accumulated_train_time = chrono.accum_train_time
      # We need to transfer the weights over now or else we risk keeping them
      # alive while they'll be updated in a future step, creating hard to debug
      # memory errors (see b/160593526). Also, takes device 0's params only.
      opt_cpu = jax.tree_util.tree_map(lambda x: np.array(x[0]), opt_repl)

      # Check whether we want to keep a copy of the current checkpoint.
      copy_step = None
      if train_utils.itstime(step, config.get('keep_checkpoint_steps'),
                             total_steps):
        write_note('Keeping a checkpoint copy...')
        copy_step = step

      # Checkpoint should be a nested dictionary or FLAX datataclasses from
      # `flax.struct`. Both can be present in a checkpoint.
      checkpoint_data = checkpoint_utils.CheckpointData(
          train_loop_rngs=train_loop_rngs,
          optimizer=opt_cpu,
          accumulated_train_time=accumulated_train_time)

      checkpoint_writer = pool.apply_async(
          checkpoint_utils.checkpoint_trained_model,
          (checkpoint_data, f'{save_checkpoint_path[:-4]}_{step}.npz', copy_step))  # EDIT(anuj)
      chrono.resume()

    # Report training progress
    if not config.get('only_eval', False) and train_utils.itstime(
        step, config.log_training_steps, total_steps, process=0):
      write_note('Reporting training progress...')
      train_loss = loss_value[0]  # Keep to return for reproducibility tests.
      timing_measurements, note = chrono.tick(step)
      write_note(note)
      train_measurements = {}
      train_measurements.update({
          'learning_rate': lr_repl[0],
          'training_loss': train_loss,
      })
      train_measurements.update(flax.jax_utils.unreplicate(extra_measurements))
      train_measurements.update(timing_measurements)
      writer.write_scalars(step, train_measurements)

    # Report validation performance
    if train_utils.itstime(step, config.log_eval_steps, total_steps):
      write_note('Evaluating on the validation set...')
      chrono.pause()

      all_eval_results = {}

      for eval_name, (eval_iter, eval_steps) in eval_iter_splits.items():
        start_time = time.time()

        # Runs evaluation loop.
        results_arrs = {
            'y_true': [],
            'y_pred': [],
            'logits': [],
            'domain_pred': [],
            'y_pred_entropy': [],
        }
        if config.only_eval:  # EDIT(anuj)
          results_arrs['pre_logits'] = []

        write_note(f'Evaluating on split: {eval_name}')
        for _, batch in zip(range(eval_steps), eval_iter):
          batch_ncorrect, batch_losses, batch_n, batch_metric_args, train_loop_rngs = (  # pylint: disable=unused-variable
              evaluation_fn(
                  opt_repl.target, batch['image'], batch['labels'], train_loop_rngs))

          # All results are a replicated array shaped as follows:
          # (local_devices, per_device_batch_size, elem_shape...)
          # with each local device's entry being identical as they got psum'd.
          # So let's just take the first one to the host as numpy.

          # from jft/deterministic.py

          # Here we parse batch_metric_args to compute uncertainty metrics.
          logits, labels, pre_logits, domain_pred = batch_metric_args  # EDIT(anuj)
          logits = np.array(logits[0])
          probs = jax.nn.softmax(logits, axis=-2)
          probs = jnp.mean(probs, axis=-1)

          # From one-hot to integer labels.
          labels = np.argmax(np.array(labels[0]), axis=-1)  # EDIT(anuj)
          domain_pred = np.array(domain_pred[0])  # EDIT(anuj)

          probs = np.reshape(probs, (probs.shape[0] * probs.shape[1], -1))
          logits = np.reshape(logits, (logits.shape[0] * logits.shape[1], -1))
          # labels = np.reshape(labels, (labels.shape[0] * labels.shape[1], -1))
          labels = labels.flatten()
          
          domain_pred = domain_pred.flatten()
          # domain_pred = np.reshape(domain_pred, (domain_pred.shape[0] * domain_pred.shape[1], -1))

          # domain_pred = domain_pred.flatten()
          # int_labels = int_labels.flatten()  # EDIT(anuj)

          y_pred = probs[:, 1] #np.max(probs, axis=-1) # karm

          # import pdb; pdb.set_trace()
          batch_trunc = int(batch['mask'].sum())  # EDIT(anuj)

          results_arrs['y_true'].append(labels[:batch_trunc])
          results_arrs['y_pred'].append(y_pred[:batch_trunc])
          results_arrs['logits'].append(logits[:batch_trunc])
          results_arrs['domain_pred'].append(domain_pred[:batch_trunc])

          if config.only_eval:  # EDIT(anuj)
            pre_logits = np.array(pre_logits[0])
            pre_logits = np.reshape(pre_logits, (pre_logits.shape[0] * pre_logits.shape[1], -1))
            results_arrs['pre_logits'].append(pre_logits[:batch_trunc])

          # Entropy is computed at the per-epoch level (see below).
          results_arrs['y_pred_entropy'].append(probs[:batch_trunc])

        results_arrs['y_true'] = np.concatenate(results_arrs['y_true'], axis=0)
        results_arrs['logits'] = np.concatenate(results_arrs['logits'], axis=0)
        results_arrs['y_pred'] = np.concatenate(
            results_arrs['y_pred'], axis=0).astype('float64')
        results_arrs['domain_pred'] = np.concatenate(results_arrs['domain_pred'], axis=0)
        results_arrs['y_pred_entropy'] = vit_utils.entropy(
            np.concatenate(results_arrs['y_pred_entropy'], axis=0), axis=-1)
        if config.only_eval:  # EDIT(anuj)
          results_arrs['pre_logits'] = np.concatenate(results_arrs['pre_logits'], axis=0)

        time_elapsed = time.time() - start_time
        results_arrs['total_ms_elapsed'] = time_elapsed * 1e3
        results_arrs['dataset_size'] = results_arrs['y_true'].shape[0]

        domain_pred_mean = np.mean(results_arrs['domain_pred'] > 0)
        results_arrs['domain_pred_recall'] = domain_pred_mean if 'ood' in eval_name else (1 - domain_pred_mean)

        all_eval_results[eval_name] = results_arrs

      per_pred_results, metrics_results = vit_utils.evaluate_vit_predictions(  # pylint: disable=unused-variable
          dataset_split_to_containers=all_eval_results,
          is_deterministic=True,
          num_bins=15,
          return_per_pred_results=True
      )

      write_note(f"=========================\n {metrics_results} \n =========================")
      # import pdb; pdb.set_trace()
      
      for eval_name in eval_iter_splits.keys():
        metrics_results[eval_name][f'{eval_name}/domain_pred_recall'] = all_eval_results[eval_name]['domain_pred_recall']

      # `metrics_results` is a dict of {str: jnp.ndarray} dicts, one for each
      # dataset. Flatten this dict so we can pass to the writer and remove empty
      # entries.
      flattened_metric_results = {}
      for dic in metrics_results.values():
        for key, value in dic.items():
          if value is not None:
            flattened_metric_results[key] = value
      writer.write_scalars(step, flattened_metric_results)

      # Optionally log to wandb
      if config.use_wandb:
        wandb.log(metrics_results, step=step)

      # Save per-prediction metrics
      results_storage_utils.save_per_prediction_results(
          output_dir, step, per_pred_results, verbose=False)
      chrono.resume()

    # End of step.
    if config.get('testing_failure_step'):
      # Break early to simulate infra failures in test cases.
      if config.testing_failure_step == step:
        break

  write_note(f'Done!\n{chrono.note}')
  pool.close()
  pool.join()
  writer.close()

  if wandb_run is not None:
    wandb_run.finish()

  # Return final training loss, validation loss, and fewshot results for
  # reproducibility test cases.
  # return train_loss, val_loss, results
  # TODO(nband): fix result reporting for DR ViT-16 reproducibility unit tests


if __name__ == '__main__':
  # Adds jax flags to the program.
  jax.config.config_with_absl()
  app.run(main)
