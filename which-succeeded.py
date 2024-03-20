import sys
from google.cloud import aiplatform
from tqdm import tqdm

def cancel(s):
	try:
		j = aiplatform.CustomJob.get(s)
		j.cancel()
	finally:
		return

def print_if_success(s):
	try:
		j = aiplatform.CustomJob.get(s)
		a = j.job_spec.worker_pool_specs[0].container_spec.args
		i = [s.split('=')[-1] for s in a if s.startswith('--run_id=')][0]
		if j.state in [4]:
			print(i)
	finally:
		return


def parse_log_file(fname):
	with open(fname, 'r') as f:
		data = f.read().split('\n')
	return [
		line.split('\'')[1]
		for line in data
		if line.startswith('custom_job = aiplatform.CustomJob.get')
	]


fname = sys.argv[1]
l = parse_log_file(fname)

for s in tqdm(l):
	# cancel(s)
	print_if_success(s)
