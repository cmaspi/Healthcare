import csv
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
from activitycl import actcl
from sklearn.metrics import cohen_kappa_score,v_measure_score,adjusted_rand_score,f1_score,accuracy_score

from scipy import stats
from sklearn.cluster import KMeans,AgglomerativeClustering,Birch,MiniBatchKMeans,SpectralClustering

from sklearn.metrics import cohen_kappa_score,v_measure_score,adjusted_rand_score,f1_score,accuracy_score,normalized_mutual_info_score
from collections import Counter

import matplotlib.patches as mpatches
from activitycl import actcl, clusterdata, purity


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{p:.2f}% ({v:d})'.format(p=pct, v=val)

    return my_autopct


def getpiecharts(cluster, label, participants, activities, numofclusters):
    index = 0

    partdf = {}
    labdf = {}
    actdf = {}
    column = range(len(np.unique(cluster)))
    for i in column:
        partdf.update({i: np.array([])})
        labdf.update({i: np.array([])})
        actdf.update({i: np.array([])})
    for val in cluster:
        partdf[val] = np.append(partdf[val], participants[index])
        labdf[val] = np.append(labdf[val], label[index])
        actdf[val] = np.append(actdf[val], activities[index])
        index += 1
    cmap = plt.cm.prism
    for i in column:
        parti = Counter(partdf[i])
        labi = Counter(labdf[i])
        acti = Counter(actdf[i])
        cmap(np.linspace(0., 1., len(parti)))
        cmap(np.linspace(0., 1., len(labi)))
        cmap(np.linspace(0., 1., len(acti)))
    return 1


def findlabel(part, act, data, definitions):
    index = 0
    for val in data:
        if val[0] == part:
            if val[1] in act:
                return definitions[index]
        index += 1


def purity(a, b, numofclusters):
    index = 0
    df = {}
    labeldist = Counter(b)
    column = range(numofclusters)
    sum = 0
    for i in column:
        df.update({i: np.array([])})
    for val in a:
        df[val] = np.append(df[val], b[index])
        index += 1
    for i in column:
        cnt = Counter(list(df[i]))
        bestkey = 0
        bestres = 0
        for key in cnt:
            if float(cnt[key]) / labeldist[key] > bestres:
                bestkey = key
                bestres = float(cnt[bestkey]) / labeldist[bestkey]
        sum += cnt[bestkey]
    purti = float(sum) / len(a)
    return purti


def actpurity(activities, cluster):
    uniact = np.unique(activities)
    unicl = np.unique(cluster)
    p = {}
    c = {}
    final = {}
    for cl in unicl:
        temp = []
        for i, val in enumerate(cluster):
            if val == cl:
                temp.append(activities[i])
        cnt = Counter(temp)
        c.update({cl: cnt})
        dic = {}
        for act in uniact:
            dic.update({act: cnt[act] / float(len(temp))})
        p.update({cl: dic})
    for a in uniact:
        num = 0
        den = 0
        for ccl in unicl:
            num += p[ccl][a] * c[ccl][a]
            den += c[ccl][a]
        final.update({a: num / float(den)})
    res = sum(final.values()) / float(len(final.values()))
    return res


def actclusterprec(data, part, groups, act):
    uniact = np.unique(act)
    dictnum = {}
    dictkey = {}
    dictcl = {}
    for val in uniact:
        dictnum.update({val: 0})
        dictkey.update({val: 'NA'})
        dictcl.update({val: 'NA'})
    p = actpurity(act, data)
    return p


def convert(cluster, ground):
    index = 0
    df = {}
    column = range(len(np.unique(cluster)))
    for i in column:
        df.update({i: np.array([])})
    for val in cluster:
        df[val] = np.append(df[val], ground[index])
        index += 1
    converted = []
    for i in cluster:
        cnt = Counter(list(df[i]))
        bkey = max(cnt, key=cnt.get)
        converted.append(bkey)
    return converted


def cluster_analysis(cluster, part_label, act_label, deflabelmin, definitions, index):
    ccluster = convert(cluster, definitions)
    return cohen_kappa_score(ccluster, definitions), v_measure_score(ccluster, definitions), adjusted_rand_score(ccluster, definitions), purity(cluster, definitions, index), normalized_mutual_info_score(ccluster,definitions), definitions


def clusterdata(cluster, n_cluster):
    kmeans = AgglomerativeClustering(n_clusters=n_cluster).fit(cluster)
    cluster_labels = kmeans.fit_predict(cluster)
    silhouette_avg = silhouette_score(cluster, cluster_labels)
    return silhouette_avg, cluster_labels


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def ttestselect(act, ttest):
    index = 0
    TSST = []
    Rest = []
    features = []
    pvals = []
    tstat = []
    for val in act:
        if 'Arithmetic' in val or 'Speech' in val or 'Sing' in val:
            TSST.append(index)
        if 'rest' in val or 'Eating' in val or 'Game' in val:
            Rest.append(index)
        index += 1

    for col in ttest:
        if 'Participant' in col or 'activity' in col or 'window_num' in col:
            continue
        else:
            x = ttest[col].iloc[TSST]
            y = ttest[col].iloc[Rest]
            stat, p = stats.ttest_ind(list(x), list(y))
            if float(p) < .01:
                pvals.append(p)
                features.append(col)
                tstat.append(stat)
    return features


def combine(x, y):
    temp = []
    for i in range(len(x)):
        string = str(x[i]) + str(y[i])
        x[i] = string
    return x

def labelize(df):
    result = df.copy()
    for feature_name in df.columns:
        avg_value = df[feature_name].mean()
        result[feature_name] = result[feature_name].apply(lambda x: 1 if x >= avg_value else 0)
    return result


directory = "../../../Data/In-lab/"
for file in [2,3,4,5,6,7,9,11,13,14,15,16,17,19,20,21,22]:
    rootdir = directory + 'P' + str(file) + '/labeledfeatures.csv'
    pd.read_csv(rootdir)
    tempdata = pd.read_csv(rootdir)
    tempdata = tempdata[(tempdata['mean'] >= 300) & (tempdata['mean'] <= 1500) & (tempdata['Count'] > 0.25)]
    tempdata = tempdata.dropna()
    normalizeddata = normalize(tempdata[tempdata.columns[2:30]])
    normalizeddata['Participant'] = file
    normalizeddata['Activity'] = tempdata['EventType']
    normalizeddata['Combined'] = normalizeddata['Activity'].apply(lambda x: str(file) + x)
    templabel = tempdata[tempdata.columns[33:]]
    if file == 2:
        totaldata = normalizeddata
        totallabel = labelize(templabel)
    else:
        totaldata = totaldata.append(normalizeddata)
        totallabel = totallabel.append(labelize(templabel))
totaldata = totaldata.reset_index()
totallabel = totallabel.reset_index()
totaldata.to_csv('allInLabData.csv', index = False)
totallabel.to_csv('allInLabLabel.csv', index = False)
selectfeatures = []
for i in range(10):
    selectfeatures.extend(actcl(totaldata))
selectfeatures = np.unique(np.array(selectfeatures))
print(selectfeatures)
df = totallabel
del df['index']
i = 0
EMAq = ['Did you experience anything stressful?', 'How stressed were you feeling?',
        'Did you feel things are going your way?',
        'Did you feel difficulties piling up so you cannot overcome them?',
        'How happy were you feeling?',
        'How worried were you feeling?', 'Induction']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
silhoutteaverages = []
totalavg = []
print(totaldata.columns)
dm = totaldata[['Participant','Activity']]
da = dm.as_matrix()
del totaldata['index']
puritiesbycol = {}
cluster = totaldata[selectfeatures].as_matrix()
print(selectfeatures)
with open('resutlsofclusteringnoneed.csv', 'w') as f:
    writer = csv.writer(f)
    for j in range(2, 16):
        x, y = clusterdata(cluster, j)
        part = totaldata['Participant']
        groups = totaldata['Combined']
        act = totaldata['Activity']
        p = actclusterprec(y, part, groups, act)
        silhoutteaverages.append(p)
        for col in EMAq:
            if 'Induction' in col:
                print('Induction')
            if j == 2:
                puritiesbycol.update({col: np.array([])})
            testwith = list(df[col])
            n, m, o, r, s, deflabel = cluster_analysis(y, list(totaldata['Participant']),
                                                       list(totaldata['Activity']), da,
                                                       testwith, j)
            puritiesbycol[col] = np.append(puritiesbycol[col], r)
            writer.writerow([n, m, o, r, s, j, col])
        print(getpiecharts(y, deflabel, list(totaldata['Participant']), list(totaldata['Activity']), j))
with open('dict.csv', 'w') as csv_file:
    writerr = csv.writer(csv_file)
    for key, value in puritiesbycol.items():
        temp = list(value)
        temp.append(key)
        writerr.writerow([temp])
if i == 0:
    totalavg = silhoutteaverages
else:
    totalavg = [x + y for x, y in zip(totalavg, silhoutteaverages)]
changesil = [0]
for i in range(1,14):
    print(i)
    changesil.append(silhoutteaverages[i]-silhoutteaverages[i-1])
plt.plot(range(2, 16), silhoutteaverages , 'k--')
plt.plot(range(2,16), changesil, 'b--')
plt.legend(handles=[mpatches.Patch(color='black', label='Silhouette Score'),
                    mpatches.Patch(color='green', label='Change in Silhouette Score')])
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()
i = 0
finaltotal = np.array([0] * 14)
for col in EMAq:
    finaltotal = np.add(finaltotal, np.array(puritiesbycol[col]))
    plt.plot(range(2, 16), puritiesbycol[col], colors[i])
    i += 1

i = 0
for v in finaltotal:
    finaltotal[i] = v/float(len(EMAq))
    i += 1

print(finaltotal)
plt.plot(range(2, 16), finaltotal, 'grey')
plt.legend(handles=[mpatches.Patch(color='blue', label='BinaryStress'), mpatches.Patch(color='green', label='LikertStress'),
                    mpatches.Patch(color='red', label='PSS-Overcome'), mpatches.Patch(color='cyan', label='PSS-YourWay'),
                    mpatches.Patch(color='magenta', label='HappyStress'), mpatches.Patch(color='yellow', label='WorriedStress'),
                    mpatches.Patch(color='black', label='Induction'),mpatches.Patch(color='grey', label='Average')],
           loc=4)
plt.xlabel("Number of Clusters")
plt.ylabel("Cluster Precision")
plt.figure(1)
plt.show()


