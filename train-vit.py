# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""XManager launcher for CIFAR10.

Usage:

xmanager launch examples/cifar10_tensorflow/launcher.py -- \
	--xm_wrap_late_bindings [--image_path = gcr.io/path/to/image/tag]
"""
import asyncio
import itertools
import os

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local
from xmanager.cloud import vertex

FLAGS = flags.FLAGS
flags.DEFINE_string('tensorboard', None, 'Tensorboard instance.')
flags.DEFINE_string('bucket', 'ue-usrl-anuj', 'GCS Bucket name')


docker_instructions = list(filter(None, '''

ENV LANG=C.UTF-8
SHELL ["/bin/bash", "-c"]

RUN rm -rf /src/uncertainty-baselines

COPY uncertainty-baselines/ /src/uncertainty-baselines
RUN chown -R 1000:root /src/uncertainty-baselines && chmod -R 775 /src/uncertainty-baselines

RUN git config --global --add safe.directory /src/uncertainty-baselines

RUN source activate base

RUN pip install -U flax==0.5.*
RUN pip install -U jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /src/uncertainty-baselines

'''.split('\n')))



def main(_):
	with xm_local.create_experiment(experiment_title = 'ub-vit-dan') as experiment:
		spec = xm.PythonContainer(
				# Package the current directory that this script is in.
				path = '.',
				base_image = 'gcr.io/external-collab-experiment/make-conda',
				entrypoint = xm.ModuleName('baselines.diabetic_retinopathy_detection.jax_finetune_dan'),
				docker_instructions = docker_instructions,
		)

		[executable] = experiment.package([
				xm.Packageable(
						executable_spec = spec,
						executor_spec = xm_local.Vertex.Spec(),
						args = {},
				),
		])


		tensorboard = FLAGS.tensorboard
		if not tensorboard:
			tensorboard = vertex.get_default_client().get_or_create_tensorboard(
					FLAGS.bucket)
			tensorboard = asyncio.get_event_loop().run_until_complete(tensorboard)

		root_output_dir = f"gs://{FLAGS.bucket}/vit/vit-dan-outputs/{experiment.experiment_id}"

		config_file = 'configs/drd/vit_finetune.py'

		seeds = [1, 2, 3, 4, 5]

		for seed in seeds:

			output_dir = os.path.join(root_output_dir, str(seed))

			tensorboard_capability = xm_local.TensorboardCapability(
					name = tensorboard, base_output_directory = output_dir)

			experiment.add(
					xm.Job(
							executable = executable,
							executor = xm_local.Vertex(
								xm.JobRequirements(
									A100 = 1,
								),
								tensorboard = tensorboard_capability,
							),
							args = {
								'output_dir': output_dir,
								'config': config_file,
								'config.seed': seed,
							},
					))


if __name__ == '__main__':
	app.run(main)

