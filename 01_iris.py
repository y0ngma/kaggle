import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
import os, sys

PROJECT_DIR = os.path.abspath(__file__)
data_folder = f"{os.path.dirname(PROJECT_DIR)}/data"
data_path = f"{data_folder}/Iris.csv"

# df = pd.read_csv(data_path)
# print(data_folder)

# print(check_output(["dir", data_folder], shell=True).decode("utf8"))
# print(check_output(["ls", "c:/home/jason/kaggle/data"]).decode("utf8"))

# subprocess.run([f'{data_folder}/mytest.py'], shell=True)

# myencoding = 'euc-kr'
myencoding = 'utf-8'
save_dir = f"{data_folder}/output.txt"
# out = subprocess.run(f'{data_folder}/mytest.py', shell=True, capture_output=True, encoding='utf8', text=True).stdout
out = subprocess.run(args=[sys.executable, f'{data_folder}/mytest.py'], capture_output=True)
print(out.stdout.decode())

with open(save_dir, 'w', encoding=myencoding) as f:
    f.write(out.stdout.decode())
    print(save_dir)

