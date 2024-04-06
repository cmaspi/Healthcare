# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:29:34 2019

@author: zking
"""
import pandas as pd
import statistics
import os
from datetime import datetime, timedelta
from sklearn.svm import SVC
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pickle
import sklearn


def dt_to_ms(dt):
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return int(delta.total_seconds() * 1000)

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


Participant_lookup = {
    101:3,
    102:13,
    104:15,
    103:16,
    105:17,
    106:20,
    107:24,
    108:27,
    110:38,
    109:41,
    113:47,
    111:49,
    112:52,
    114:76,
    115:80,
    116:79,
    117:81

}

print(sklearn.__version__)
questions = ['BinaryStress', 'LikertStress','WorriedStress']
selected_features = ['min','max','median','mode','80_percentile','20_percentile',
                     'COV_M','RMSSD','zcross','Lf']

srdata = pd.read_csv('./EMA_TP101_29_2019.csv')
directory = '../../../Data/In-wild/'
trainDF = pd.read_csv('./InLabDF.csv')


dots = sns.load_dataset("dots").query("align == 'dots'")
clf = SVC(C=54, gamma=0.0002, kernel='rbf')
clf.fit(trainDF[selected_features], trainDF['Induced'])
filename = 'SVM_StressModel.sav'
pickle.dump(clf, open(filename, 'wb'))
hours = range(1,61)
ho = []
data = []
hue = []
sty = []
Before = 0
After = 0
for h in range(1,60):
    print(h)
    for q in ['BinaryStress','LikertStress','WorriedStress']:
        stressval =[]
        noval = []
        avgval = []
        for filename in os.listdir(directory):
            partsr = srdata[srdata['Participant'] == Participant_lookup[int(filename[1:])]]
            stressperc = 0
            count = 0
            for day in os.listdir(directory + filename + '/processed/'):
                parttestlabel = []
                parttestdata = []
                partData = pd.read_csv(directory + filename + '/processed/' + day)
                stressperc += partData['Count'].mean()
                count += 1
                Before += len(partData)
            
                partData = partData[(partData['mean'] >= 300) & (partData['mean'] <= 1500) & (partData['Count'] > 0.7)]
                After += len(partData)
                partData.iloc[:,range(2,31)] = normalize(partData.iloc[:,range(2,31)])
                
                for index, row in partsr.iterrows():
                    TimeStamp = row['UTC_Time']
                    end_datetime_object = dt_to_ms(datetime.strptime(TimeStamp, '%m/%d/%y %H:%M'))
                    start_datetime_object = dt_to_ms(datetime.strptime(TimeStamp, '%m/%d/%y %H:%M') - timedelta(hours=h/60))
                    x = partData.loc[(partData['End'] - 30000 >= start_datetime_object) &
                                     (partData['Start'] + 30000 <= end_datetime_object)]
                    if len(x) > 0:
                        y_pred = clf.predict(x[selected_features])
                        predperc = sum(y_pred) / len(y_pred)
                        
                        response = row[q]
                        partAVG = partsr[q].mean()
                        avgval.append(predperc)
                        if response > partAVG:
                            stressval.append(predperc)
                        else:
                            noval.append(1-predperc)

        ho.append(h)
        ho.append(h)
        ho.append(h)
        data.append(statistics.mean(noval))
        data.append(statistics.mean(stressval))
        data.append(statistics.mean([statistics.mean(stressval),statistics.mean(noval)]))
        hue.append('No Stress')
        hue.append('Stress')
        hue.append('Average')
        sty.append(q)
        sty.append(q)
        sty.append(q)

ax = sns.lineplot(x=ho, y=data, hue=hue,style = sty)
plt.legend(loc=(-1,-1),ncol=2)
plt.xlabel('Window Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

