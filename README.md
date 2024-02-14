# Hand written digit prediction
#Objective of the projrct
The digit dataset consists of 8*8 pixel images of digits. The images attribute of the dataset stroes 8*8 arrays of the grayscale value for each images. We will use these arrays to visualize the frist 4 images. The target attribute of the dataset stroes the digit of each image represents
#import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
from sklearn.datasets import load_digits

df = load_digits()

_, axes = plt.subplots(nrows =1, ncols =4, figsize = (10,3))
for ax, image, label in zip (axes, df.images, df.target):
  ax.set_axis_off()
  ax.imshow(image,cmap = plt.cm.gray_r, interpolation = "nearest")
  ax.set_title("training:%i" % label)

#data processing
# Flatten Image

df.images.shape

df.images[0]

df.images[0].shape

len(df.images)

n_samples = len(df.images)
data = df.images.reshape((n_samples, -1))

data[0]

data[0].shape

data.shape

#scalling image data
data.min()

#scalling image
data.max()


data = data/16
data.min()
#scalling Image Data
data.max()

data[0]

#train_test_split data
from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test = train_test_split(data,df.target,test_size = 0.3)

x_train.shape,x_test.shape,y_train.shape,y_test.shape

# random forest model
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier()
rf.fit(x_train,y_train)

#Predict test data
y_pred = rf.predict(x_train)


y_pred

# Model Accuracy 
from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test,y_pred)

print(classification_report(y_test, y_pred))























