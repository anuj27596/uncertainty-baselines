import os
import tensorflow as tf
import matplotlib.pyplot as plt
import uncertainty_baselines as ub
import baselines.isic.input_utils as input_utils

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

data_dir = '/troy/anuj/gub-mod/uncertainty-baselines/data/downloads/manual/rxrx1'
for ds_name in ['rxrx1_id', 'rxrx1_ood']:
    for split in ['train','test','validation']:
        if 'ood' in ds_name and split == "train":
            continue
        # ds_name = 'rxrx1_id' # zhang_pneumonia
        # split = 'validation'
        # op_csv = "eye_labels/train_eyepacs_eyelabels.csv"
        batch_size = 16
        config = "processed"

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