import jax.numpy as jnp
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

x = 17
print(f"allocating empty")

# arr = 
jnp.empty((x*(2**27),1))
# jnp.empty(((2**27),1))

# print(arr.shape)
print(f"allocated")
time.sleep(180)
print(f"allocating zeros")
arr2 = jnp.zeros((x*(2**28),1))
print(arr2.shape)
time.sleep(5)