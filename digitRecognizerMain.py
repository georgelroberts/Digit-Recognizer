# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 11:56:09 2018

@author: George

Python foray into computer vision.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
# %% First load the data

os.chdir('C:\\Users\\George\OneDrive - University Of Cambridge\\Others\\Machine learning\\Kaggle\\Digit Recognizer')


def loadData():
    train = np.genfromtxt('Data\\train.csv', delimiter=',', skip_header=1,
                          dtype='int32')
    trainX = train[:, 1:]
    trainY = train[:, 0]
    test = np.genfromtxt('Data\\test.csv', delimiter=',', skip_header=1,
                         dtype='int32')
    return trainX, trainY, test


trainX, trainY, test = loadData()

# %% Convert to black and white and take a look at some training data

figSize = (28, 28)

randLabels = np.random.randint(0, len(trainX)-1, 25)
plt.figure(figsize=(15, 15))

for i in range(len(randLabels)):
    image = np.reshape(trainX[randLabels[i], :], figSize)
    ax = plt.subplot(5, 5, i+1)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Training class: ' + str(trainY[randLabels[i]]))

# %% Convert data to B&W, then separate into test and train for classification.
# Set the train size to low for the time being in order to minimize train time.


def imToBW(imArray):
    """Convert features to black and white """
    BWBoundary = 128
    imArray[imArray <= BWBoundary] = 0
    imArray[imArray > BWBoundary] = 1
    return imArray


train_X, test_X = imToBW(trainX), imToBW(test)

train_X, cvX, train_Y, cvY = train_test_split(train_X, trainY,
                                              train_size=0.05, random_state=0)

# %% Support vector classifier

train_sizes, train_scores, valid_scores = learning_curve(
        SVC(kernel='linear'), train_X, train_Y, train_sizes=[50, 80, 110], cv=5)
fig, ax = plt.subplots()
ax.plot(train_sizes, train_scores)
ax.plot(train_sizes, valid_scores)

parameters = {'C': np.logspace(-3, 1, 5)}
clf = GridSearchCV(SVC(kernel='linear'), parameters)

clf.fit(train_X, train_Y)

cvPredictSVM = clf.predict(cvX)
print('CV Score = ' + str(accuracy_score(cvY, cvPredictSVM)))
testPredictSVM = clf.predict(test_X)

# %% Plot predicted data

randLabels = np.random.randint(0, len(cvX)-1, 25)
plt.figure(figsize=(15, 15))

for i in range(len(randLabels)):
    image = np.reshape(cvX[randLabels[i], :], figSize)
    ax = plt.subplot(5, 5, i+1)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Predicted class: ' + str(cvPredictSVM[randLabels[i]]))
