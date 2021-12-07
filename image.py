import scipy.spatial
from PIL import Image
import scipy.io as io
import scipy
import numpy as np
import h5py
import cv2
import random
import os
from config import args

def load_data(img_path, args, train=True):
    # img_path =
    # gt_path = img_path.replace('img', 'gt').replace('jpg', 'h5')  # vanilla
    # gt_path = img_path.replace('images_crop', 'gt_density_map_crop') .replace('jpg', 'h5')  # shanghaitech
    gt_path = img_path.split("__", 1)[0].replace('img', 'gt') .replace('jpg', 'h5')  # dataaug
    if train:
        gt_path = './data/train/gt/' + os.path.basename(img_path.split("__",1)[0]).split(".jpg")[0] + ".h5"
    else:
        gt_path = img_path.replace('img', 'gt').replace('jpg', 'h5')

    img = Image.open(img_path)

    img = img.resize((224, 224), Image.BICUBIC).convert('RGB')

    while True:
        try:
            gt_file = h5py.File(gt_path)
            gt_count = np.asarray(gt_file['gt_count'])
            break  # Success!
        except OSError:
            print("load error:", img_path, gt_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    gt_count = gt_count.copy()

    return img, gt_count
