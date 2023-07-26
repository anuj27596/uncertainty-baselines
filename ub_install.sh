### Do outside
# conda create -n ub python=3.8
# conda activate ub2

### Tensorflow installation
conda install -c conda-forge cudatoolkit=11.8.0 -y
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))/lib
echo $LD_LIBRARY_PATH
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

### UB-DRD requirements
pip install -e "git+https://github.com/google/uncertainty-baselines.git#egg=uncertainty_baselines"[models,jax,tensorflow,torch,retinopathy]
pip install 'git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics'

# pip install 'git+https://github.com/google/edward2.git'
# or
# if edward2 gives = "No backend avalable error" then run following commands
# pip install edward2

### Custom
pip install -U flax==0.5.*
pip install jax[cuda11_cudnn82]==0.4.7 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
# # incase above commands fails, 
# # conda update -n base -c defaults conda
# # conda clean --all
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# pip install tensorflow
# python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# conda install -c anaconda git 
# pip install -e "git+https://github.com/google/uncertainty-baselines.git#egg=uncertainty_baselines"[models,jax,tensorflow]
# pip install 'git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics'
# pip install 'git+https://github.com/google/edward2.git'

# python -c "import jax; jax.random.PRNGKey(0)"
# python -c "import edward2.jax"