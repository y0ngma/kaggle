import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
import os, sys

PROJECT_DIR = os.path.abspath(__file__)
data_folder = f"{os.path.dirname(PROJECT_DIR)}/data"
data_path   = f"{data_folder}/Iris.csv"
save_dir    = f"{data_folder}/{os.path.basename(os.path.splitext(data_path))}_output.txt"

# df = pd.read_csv(data_path)
# print(data_folder)

myencoding = 'utf-8'
out = subprocess.run(args=[sys.executable, f'{data_folder}/mytest.py'],
                     capture_output=True)

with open(save_dir, 'w', encoding=myencoding) as f:
    f.write(out.stdout.decode())
print(save_dir)

