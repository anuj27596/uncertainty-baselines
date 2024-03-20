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
import os
import time
import asyncio
import itertools

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local
from xmanager.cloud import vertex

from google.cloud import aiplatform

from entrypoint_histo import get_run_ids, get_output_dir


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


LAUNCHED = [1,2,3]
SUCCEEDED = [4]
FAILED = [5,6,7,8,9]

# 8: JobState.JOB_STATE_PAUSED
# 0: JobState.JOB_STATE_UNSPECIFIED
# 10: JobState.JOB_STATE_UPDATING
# 11: JobState.JOB_STATE_PARTIALLY_SUCCEEDED


def logger(exp_id):
	logfile = f'.logs/xm-{exp_id}.log'
	
	def log(msg):
		with open(logfile, 'a') as f:
			f.write(ts() + '\t' + msg + '\n')
			f.flush()
			os.fsync(f.fileno())

	return log


def ts():
	return f'[{time.ctime()[4:-5]}]'


def main(_):

	jobs = {}
	done = set([]) # dan done # 14, 15, 17, 18, 20, 22, 23, 26, 28, 29, 33, 37, 43, 46, 62, 68

	with xm_local.create_experiment(
		experiment_title = 'karm-ub-vit'
	) as experiment:
		spec = xm.PythonContainer(
			# Package the current directory that this script is in.
			path = '.',
			base_image = 'gcr.io/external-collab-experiment/make-conda',
			entrypoint = xm.ModuleName('entrypoint_histo'),
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
			tensorboard = asyncio.get_event_loop().run_until_complete(
				vertex.get_default_client()
				.get_or_create_tensorboard(FLAGS.bucket))

		logg = logger(experiment.experiment_id)

		rid_set = get_run_ids()

		for run_id in itertools.cycle(rid_set):

			if len(done) == len(rid_set):
				break

			if run_id in done:
				continue

			if run_id in jobs:
				state = jobs[run_id].state
				if state in SUCCEEDED:
					done.add(run_id)
					logg(f'run {run_id} succeeded')
					continue
				elif state in FAILED:
					logg(f'run {run_id} failed')
				# elif state in LAUNCHED:
				else:
					continue

			exp_add = experiment.add(
				xm.Job(
					executable = executable,
					executor = xm_local.Vertex(
						xm.JobRequirements(
							A100 = 1,
						),
						tensorboard = xm_local.TensorboardCapability(
							name = tensorboard,
							base_output_directory = get_output_dir(
								run_id, experiment.experiment_id)),
					),
					args = dict(
						exp_id = experiment.experiment_id,
						run_id = run_id,
						run = True,
					),
			))

			got_wu = False
			while not got_wu:
				try:
					work_unit = exp_add.result()
					got_wu = True
				except asyncio.exceptions.InvalidStateError:
					time.sleep(1)

			jobs[run_id] = aiplatform.CustomJob.get(
				work_unit._non_local_execution_handles[0].job_name)
			logg(f'run {run_id} launched')


	logg('-' * 54)
	logg('all done')


if __name__ == '__main__':
	app.run(main)

