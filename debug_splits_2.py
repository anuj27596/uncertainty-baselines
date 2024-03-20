import os
import tensorflow as tf
import matplotlib.pyplot as plt
import uncertainty_baselines as ub
import baselines.mammographic.input_utils as input_utils

from tqdm import tqdm
from PIL import Image                
from numpy import asarray

import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure
import cv2
import numpy as np
import pandas as pd
import tqdm
import uncertainty_baselines.datasets as datasets

os.environ['CUDA_VISIBLE_DEVICES'] = ''
dataset='mammographic'
data_dir = f'/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/{dataset}'
for ds_name in ['inbreast']:
    for split in ['test','validation']:
        if 'ood' in ds_name and split == "train":
            continue
        # ds_name = 'rxrx1_id' # zhang_pneumonia
        # split = 'validation'
        # op_csv = "eye_labels/train_eyepacs_eyelabels.csv"
        batch_size = 16
        config = "processed"
        # config = "processed_512_onehot"
        

        builder = datasets.get(
            ds_name,
            split = split,
            data_dir = data_dir,
            cache = False,
            shuffle_buffer_size = 256,
            builder_config = f'{ds_name}/{config}',
            download_data = True,
        )

        dataset = builder.load(batch_size = batch_size)
        batch = next(iter(dataset))
        images, labels, names = batch['features'], batch['labels'], batch['name']

        print("finished")
        
        
# labels = []
# for each in iter(dataset):
#     labels.extend(list(each['labels'].numpy()))