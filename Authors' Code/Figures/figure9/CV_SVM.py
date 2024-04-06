# -*- coding: utf-8 -*-
"""
Created on Fri May  3 03:37:35 2019

@author: zking
"""

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from NeuralNet import NeuralNet
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import pickle


def convertToSet(X_test, y_test):
    newX_test = []
    for i,p in enumerate(X_test):
        if len(newX_test) == 0:
            newX_test = p
            newy_test = y_test[i]
        else:
            newX_test = np.concatenate((newX_test, p),axis = 0)
            newy_test = np.concatenate((newy_test, y_test[i]), axis=0)
    return newX_test, newy_test

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def trainNN(X_train,y_train,X_test):
    train_len = len(y_train)
    with tf.Graph().as_default():
        session = tf.Session()
        batch_size = 1
        initialiser = tf.random_uniform_initializer(-0.1, 0.1)

        with tf.variable_scope('model', reuse=None, initializer=initialiser):
            # feature_size: the size of feature, I assume 36.
            # output_size: the size of expected output, should be as same as labels.
            m = NeuralNet   (batch_size, 10, 1, is_training=True)
        with tf.variable_scope("model", reuse=True):
            m_test = NeuralNet(batch_size, 10, 1, is_training=False)

        session.run(tf.global_variables_initializer())
        variables = tf.trainable_variables()
        saver = tf.train.Saver()
        values = session.run(variables)

    # delta_loss=[0]
    loss_opt = 99999
    for ite in range(10):
        loss_train = 0
        iterateon = train_len / 10
        for i in range(train_len):
            batch_data = np.array(X_train[i * batch_size:(i + 1) * batch_size])
            batch_label = np.array(y_train[i * batch_size:(i + 1) * batch_size])
            batch_label = batch_label.reshape((1, 1))
            output, loss, _ = session.run([m.outputs, m.cost, m.train_op],
                                          {m.input_data: batch_data,
                                           m.targets: batch_label})
            loss_train += loss
        loss_val = 0
    NNpred = []
    for i in range(len(X_test)):
        batch_data = np.array(X_test[i * batch_size:(i + 1) * batch_size])
        output, loss = session.run([m_test.outputs, m_test.cost],
                                   {m_test.input_data: batch_data,
                                    m_test.targets: np.array([1]).reshape(1,1)})
        output = np.squeeze(output)
        NNpred.append(output)

    NNpred = np.squeeze(NNpred)
    y_pred = []
    for x in NNpred:
        if x >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred


selected_features = ['min','max','median','mode','80_percentile','20_percentile',
                     'COV_M','RMSSD','zcross','Lf']
directory = "../../../Data/In-lab/"
y = []
X = []
parts = []
# labels = pd.read_csv('C:/Users/zking/WorkSpace/Stress/Micro_EMA_Selection/Micro-EMA-Selection/labels_activity.csv')
labels = pd.read_csv('./labels_activity.csv')
participants = []
for filename in os.listdir(directory):
    parttestlabel = []
    parttestdata = []
    partData = pd.read_csv(directory + filename + '/processed.csv')
    partData = partData[(partData['mean'] >= 300) & (partData['mean'] <= 1500) & (partData['Count'] > 0.7)]
    partData.iloc[:,range(2,31)] = normalize(partData.iloc[:,range(2,31)])

    partLabel = pd.read_csv(directory + filename + '/annotations.csv')



    for index, row in partLabel.iterrows():
        activity = row['EventType']
        if 'Rest- TSST' not in activity and 'cry-rest' not in activity:
            if 'Cry 7' in activity:
                activity = 'Cry 7-2'
            start = row['Start Timestamp (ms)']
            stop = row['Stop Timestamp (ms)']
            activity = labels[activity][0]

            participant = filename

            actecg = partData.loc[(partData['Start'] >= start - 30000)
                              & (partData['End'] <= stop + 30000)].iloc[:,range(2,31)]
            if len(parttestdata) == 0:
                parttestdata = actecg[selected_features].as_matrix()
                parttestlabel = [activity] * len(actecg)
            else:
                parttestdata = np.concatenate((parttestdata, actecg[selected_features].as_matrix()),axis = 0)

                parttestlabel.extend([activity] * len(actecg))

    parts.append(participant)
    participants.append(filename)
    X.append(parttestdata)
    y.append(parttestlabel)

X_train, y_train = convertToSet(X, y)        # Loading the Digits dataset
X_trainDF = pd.DataFrame(data = X_train, columns = selected_features)
X_trainDF['Induced'] = y_train
X_trainDF.to_csv('InLabDF.csv')

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:

scores = []
scoretype = []
clftype = []

clf = SVC(C=107, gamma=0.001, kernel='rbf')

clfcstress = SVC(C=724.077, gamma=0.0220971, kernel='rbf')
clfDT = tree.DecisionTreeClassifier()
bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
gnb = GaussianNB()
i = 0
f1score = []
part = []
for p in X:

    tempX_train = X
    tempy_train = y
    print(len(tempX_train))
    X_test = X[i-1]
    tempX_train = X[::-(i+1)]
    y_test = y[i-1]
    tempy_train = y[::-(i+1)]
    
    X_train, y_train = convertToSet(tempX_train, tempy_train)
    clf.fit(X_train,y_train)
    clfcstress.fit(X_train, y_train)
    clfDT.fit(X_train,y_train)
    bdt.fit(X_train, y_train)
    gnb.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    scores.append(f1_score(y_test,y_pred))
    f1score.append(f1_score(y_test,y_pred))
    part.append(participants[i])
    scoretype.append('f1_score')
    clftype.append('SVM \n(Grid Search)')
    scores.append(recall_score(y_test,y_pred))
    scoretype.append('recall')
    clftype.append('SVM \n(Grid Search)')
    scores.append(precision_score(y_test,y_pred))
    scoretype.append('precision')
    clftype.append('SVM \n(Grid Search)')

    y_pred = clfcstress.predict(X_test)
    scores.append(f1_score(y_test, y_pred))
    scoretype.append('f1_score')
    clftype.append('SVM \n(CStress)')
    scores.append(recall_score(y_test, y_pred))
    scoretype.append('recall')
    clftype.append('SVM \n(CStress)')
    scores.append(precision_score(y_test, y_pred))
    scoretype.append('precision')
    clftype.append('SVM \n(CStress)')
    
    y_pred = clfDT.predict(X_test)
    scores.append(f1_score(y_test,y_pred))
    scoretype.append('f1_score')
    clftype.append('Decision \nTree')
    scores.append(recall_score(y_test,y_pred))
    scoretype.append('recall')
    clftype.append('Decision \nTree')
    scores.append(precision_score(y_test,y_pred))
    scoretype.append('precision')
    clftype.append('Decision \nTree')

    y_pred = bdt.predict(X_test)
    scores.append(f1_score(y_test,y_pred))
    scoretype.append('f1_score')
    clftype.append('AdaBoost')
    scores.append(recall_score(y_test,y_pred))
    scoretype.append('recall')
    clftype.append('AdaBoost')
    scores.append(precision_score(y_test,y_pred))
    scoretype.append('precision')
    clftype.append('AdaBoost')

    y_pred = gnb.predict(X_test)
    scores.append(f1_score(y_test, y_pred))
    scoretype.append('f1_score')
    clftype.append('Naive \nBayes')
    scores.append(recall_score(y_test, y_pred))
    scoretype.append('recall')
    clftype.append('Naive \nBayes')
    scores.append(precision_score(y_test, y_pred))
    scoretype.append('precision')
    clftype.append('Naive \nBayes')

    y_pred = trainNN(X_train, y_train, X_test)
    scores.append(f1_score(y_test, y_pred))
    scoretype.append('f1_score')
    clftype.append('Neural \nNet')
    scores.append(recall_score(y_test, y_pred))
    scoretype.append('recall')
    clftype.append('Neural \nNet')
    scores.append(precision_score(y_test, y_pred))
    scoretype.append('precision')
    clftype.append('Neural \nNet')
    i += 1

# ax = sns.barplot(x=part, y=f1score)
plt.ylabel('F1_score')
plt.xlabel('Participant')

sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
ax = sns.boxplot(x=clftype, y=scores, hue=scoretype, palette="Set3")
plt.show()
