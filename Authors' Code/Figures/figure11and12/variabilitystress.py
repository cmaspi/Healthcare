# -*- coding: utf-8 -*-
"""
Created on Fri May  3 00:04:17 2019

@author: zking
"""

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import os
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


selected_features = ['min', 'max', 'median', 'mode', '80_percentile', '20_percentile',
                     'COV_M', 'RMSSD', 'zcross', 'Lf']
directory = '../../../Data/In-lab/'
y = []
X = []
parttestdata = pd.DataFrame([])
labels = pd.read_csv('./labels_activity.csv')

for filename in os.listdir(directory):
    partData = pd.read_csv(directory + filename + '/processed.csv')
    partData = partData[(partData['mean'] >= 300) & (partData['mean'] <= 1500) & (partData['Count'] > 0.7)]
    partData.iloc[:, range(2, 31)] = normalize(partData.iloc[:, range(2, 31)])

    partData['Participant'] = filename
    partLabel = pd.read_csv(directory + filename + '/annotations.csv')

    for index, row in partLabel.iterrows():
        activity = row['EventType']
        if 'Rest- TSST' not in activity and 'cry-rest' not in activity:
            if 'Cry 7' in activity:
                activity = 'Cry 7-2'
            start = row['Start Timestamp (ms)']
            stop = row['Stop Timestamp (ms)']
            activity = labels[activity][0]
            participant = filename[1:]
            actecg = partData.loc[(partData['Start'] >= start - 30000) & (partData['End'] <= stop + 30000)].iloc[:, range(2, 31)]
            actecg['participant'] = int(participant)
            actecg['Induced'] = activity

            if len(parttestdata) == 0:
                parttestdata = actecg
            else:
                parttestdata = parttestdata.append(actecg)


unique_parts = np.unique(parttestdata['participant'])
f1score = []
partti = []
type = []
i = 1
for part in unique_parts:
    trainDF = parttestdata[parttestdata['participant'] != part]
    clf = SVC(C=54, gamma=0.0002, kernel='rbf')
    clf.fit(trainDF[selected_features], trainDF['Induced'])
    testDF = parttestdata[parttestdata['participant'] == part]
    y_pred = clf.predict(testDF[selected_features])
    partti.append(i)
    partti.append(i)
    f1score.append(f1_score(testDF['Induced'],y_pred, average='binary'))
    f1score.append(f1_score(testDF['Induced'], y_pred, average='binary',pos_label=0))
    type.append('Positive')
    type.append('Negative')
    i += 1

ax = sns.barplot(x=partti, y=f1score,hue = type)
plt.ylabel('F1_score')
plt.xlabel('Participant')
plt.show()

selected_features = ['min','max','median','20_percentile','Lf']

y = []
x = []
hue = []
yno = []
i = 0
feats = []
goodbad = [0,2,4,6,10]
totest = unique_parts[goodbad]
for part in totest:
    for feat in selected_features:
        stress = parttestdata[(parttestdata['participant'] == part) & parttestdata['Induced'] == 1][feat]
        nostress = parttestdata[(parttestdata['participant'] == part) & parttestdata['Induced'] == 0][feat]
        y.extend(stress)
        y.extend(nostress)
        x.extend([goodbad[i]+1] * (len(nostress) + len(stress)))
        feats.extend([feat] * (len(nostress) + len(stress)))
        hue.extend(['stress']*len(stress))
        hue.extend(['no stress'] * len(nostress))
    i += 1

data = pd.DataFrame([])
data['col'] = x
data['row'] = feats
data['y'] = y
data['hue'] = hue

box = np.concatenate((y,yno))
g = sns.catplot(data, x="hue",y="y", col="col",row="row",kind="box",height=1.5, aspect=.7,margin_titles=True)
plt.show()
