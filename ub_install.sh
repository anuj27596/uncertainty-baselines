### Do outside
# conda create -n ub python=3.8 -y
# conda activate ub

### Tensorflow installation
conda install -c "conda-forge cudatoolkit=11.8.0" -y
python3 -m pip install "nvidia-cudnn-cu11==8.6.0.163" "tensorflow==2.12.1"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))/lib
echo $LD_LIBRARY_PATH
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# ### UB-DRD requirements
# conda install -c anaconda git -y
# pip install -e "git+https://github.com/google/uncertainty-baselines.git#egg=uncertainty_baselines"[models,jax,tensorflow,torch,retinopathy]
# or
# pip install -e .[models,jax,tensorflow,torch,retinopathy]
# pip install -e .[models,tensorflow,retinopathy]


# pip install 'git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics'

# pip install 'git+https://github.com/google/edward2.git'
# or
# if edward2 gives = "No backend avalable error" then run following commands
# pip install edward2

### Custom
conda install -c anaconda git -y
pip install -U "flax==0.5.3"
pip install "jax[cuda11_cudnn82]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install "clu==0.0.8"
pip install "opencv-python==4.7.0.72"
pip install "tensorflow-addons==0.19.0"
pip install "tensorflow-probability==0.19.0"
pip install 'tensorflow_datasets==4.9.2'
pip install 'tensorflow_hub==0.16.0'
pip install "seaborn==0.12.2"
pip install "wandb==0.14.0"
pip install edward2
pip install "dm-haiku==0.0.9"
pip install 'git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics'
## dataset setup

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
# python -c "import edward2.jax"'