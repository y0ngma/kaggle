import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
import os, sys

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = os.path.abspath(__file__)
data_folder = f"{os.path.dirname(PROJECT_DIR)}/data"
data_path   = f"{data_folder}/Iris.csv"
save_dir    = f"{data_folder}/{os.path.basename(os.path.splitext(data_path)[0])}_output.txt"

df = pd.read_csv(data_path)
df.drop('Id', axis=1, inplace=True)
# print(df.head())
# print(df.info())

# fig = df[df.Species=='Iris-setosa'].plot(
#     kind='scatter', 
#     x='SepalLengthCm', 
#     y='SepalWidthCm',
#     color='orange',
#     label='Setosa'
# )
# df[df.Species=='Iris-versicolor'].plot(
#     kind='scatter',
#     x='SepalLengthCm',
#     y='SepalWidthCm',
#     color='blue',
#     label='versicolor',
#     ax=fig
# )
# df[df.Species=='Iris-virginica'].plot(
#     kind='scatter',
#     x='SepalLengthCm',
#     y='SepalWidthCm',
#     color='green',
#     label='virginica',
#     ax=fig
# )
# fig.set_xlabel('Sepal Length')
# fig.set_ylabel("Sepal Width")
# fig.set_title('Sepal Lenth vs Width')
# fig=plt.gcf()
# fig.set_size_inches(10, 6)
# plt.show()

# fig = df[df.Species=='Iris-setosa'].plot.scatter(
#     x='PetalLengthCm',
#     y='PetalWidthCm',
#     color='orange',
#     label='Setosa'
# )
# df[df.Species=='Iris-versicolor'].plot.scatter(
#     x='PetalLengthCm',
#     y='PetalWidthCm',
#     color='blue',
#     label='versicolor',
#     ax=fig
# )
# df[df.Species=='Iris-virginica'].plot.scatter(
#     x='PetalLengthCm',
#     y='PetalWidthCm',
#     color='green',
#     label='virginica',
#     ax=fig
# )
# fig.set_xlabel("Petal Length")
# fig.set_ylabel("Petal Width")
# fig.set_title("Petal Length vs Width")
# fig=plt.gcf()
# fig.set_size_inches(10, 6)
# plt.show()


# df.hist(edgecolor='black', linewidth=1.2)
# fig=plt.gcf()
# fig.set_size_inches(12,6)
# plt.show()


# plt.figure(figsize=(15,10))
# plt.subplot(2,2,1)
# sns.violinplot(x='Species', y='PetalLengthCm', data=df)
# plt.subplot(2,2,2)
# sns.violinplot(x='Species', y='PetalWidthCm', data=df)
# plt.subplot(2,2,3)
# sns.violinplot(x='Species', y='SepalLengthCm', data=df)
# plt.subplot(2,2,4)
# sns.violinplot(x='Species', y='SepalWidthCm', data=df)
# plt.show()

# plt.figure(figsize=(7,4))
# # draws heatmap with input as the correlation matrix calculated by df.corr()
# sns.heatmap(df.corr(), annot=True, cmap='cubehelix_r')
# plt.show()

train, test = train_test_split(df, test_size=0.3)
print('학습용:{}, 검증용:{}'.format(train.shape, test.shape))

train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
train_y = train.Species
test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
test_y = test.Species

# print(train_X.head(2))
# print(test_y.head())

model = svm.SVC() # select the algorithm
model.fit(train_X, train_y) # train the algorithm with data and output
prediction = model.predict(test_X) # pass the testing data to the trained algorithm
whichacc = 'Acc. of the SVM'
print("{0: <35} : {1: <+4,.5f}".format(whichacc, metrics.accuracy_score(prediction, test_y)))

model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
whichacc = 'Acc. of the Logistic Regression'
print("{0: <35} : {1: <+4,.5f}".format(whichacc, metrics.accuracy_score(prediction, test_y)))

model = DecisionTreeClassifier()
model.fit(train_X, train_y)
prediction=model.predict(test_X)
whichacc = 'Acc. of the Decisition Tree'
print("{0: <35} : {1: <+4,.5f}".format(whichacc, metrics.accuracy_score(prediction, test_y)))

model = KNeighborsClassifier(n_neighbors=3) # this examines 3 neighbors for putting the new data into a class
model.fit(train_X, train_y)
prediction=model.predict(test_X)
whichacc = 'Acc. of the KNN'
print("{0: <35} : {1: <+4,.5f}".format(whichacc, metrics.accuracy_score(prediction, test_y)))


a_index = list(range(1,11))
a = pd.Series(dtype='float64')
# print(a.dtype)
for i in a_index:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    tmp = metrics.accuracy_score(prediction, test_y)
    print("{0: <35} : {1:?<+4,.5f}".format(f"Acc. of KNN with n_neightbors={i}", tmp))
    a = a.append(pd.Series(tmp))
plt.plot(a_index, a)
plt.xticks(a_index)
plt.show()



# myencoding = 'utf-8'
# out = subprocess.run(args=[sys.executable, f'{data_folder}/mytest.py'],
#                      capture_output=True)

# with open(save_dir, 'w', encoding=myencoding) as f:
#     f.write(out.stdout.decode())
# print(save_dir)

