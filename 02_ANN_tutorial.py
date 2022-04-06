
# https://www.kaggle.com/code/kanncaa1/deep-learning-tutorial-for-beginners/notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings('ignore')
import sys, os
import subprocess

PROJECT_DIR = os.path.abspath(__file__)
data_folder = f"{os.path.dirname(PROJECT_DIR)}/data/Sign language digits dataset"
x_l = np.load(f"{data_folder}/X.npy")
y_l = np.load(f"{data_folder}/X.npy")
# data_path   = f"{data_folder}/Iris.csv"
# save_dir    = f"{data_folder}/{os.path.basename(os.path.splitext(data_path)[0])}_output.txt"

# out = subprocess.run(args=[sys.executable, f'{data_folder}/mytest.py'],
#                      capture_output=True)

# print(subprocess.check_output(['ls', '../input']).decode('utf8'))

img_size = 64
plt.subplot(1, 2, 1)
# print(x_l[260].reshape(img_size, img_size).shape)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')
plt.show()

# Join a sequence of arrays along an row axis.
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)
