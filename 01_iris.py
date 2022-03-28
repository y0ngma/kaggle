import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
import os

PROJECT_DIR = os.path.abspath(__file__)
data_folder = f"{os.path.dirname(PROJECT_DIR)}/data"
data_path = f"{data_folder}/Iris.csv"

df = pd.read_csv(data_path)
# print(df.head())

print(data_folder)
# print(os.listdir(data_folder))

# print(check_output(["ls", data_folder]).decode("utf8"))
print(check_output(["ls", "c:/home/jason/kaggle/data"]).decode("utf8"))