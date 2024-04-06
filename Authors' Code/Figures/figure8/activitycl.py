import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans,AgglomerativeClustering,Birch,MiniBatchKMeans,SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import operator
import csv

def purity(activities, cluster):
    uniact = np.unique(activities)
    unicl = np.unique(cluster)
    p = {}
    c = {}
    final = {}
    for cl in unicl:
        temp = []
        for i,val in enumerate(cluster):
            if val == cl:
                temp.append(activities[i])
        cnt = Counter(temp)
        c.update({cl:cnt})
        dic ={}
        for act in uniact:
            dic.update({act : cnt[act]/float(len(temp))})
        p.update({cl: dic})
    for a in uniact:
        num = 0
        den = 0
        for ccl in unicl:
            num += p[ccl][a]*c[ccl][a]
            den += c[ccl][a]
        final.update({a: num/float(den)})
    return final

def clusterdata(cluster,n_cluster):
    kmeans = KMeans(n_clusters = n_cluster).fit(cluster.reshape(-1,1))
    cluster_labels = kmeans.fit_predict(cluster.reshape(-1,1))
    #silhouette_avg = silhouette_score(test, cluster_labels)
    return cluster_labels

def actcl(tempdata):
    data = tempdata.copy()
    part = data['Participant']
    groups = data['Combined']
    act = data['Activity']

    uniact = np.unique(data['Activity'])
    del data['Combined']
    del data['Participant']
    del data['Activity']
    del data['index']

    dictnum = {}
    dictkey = {}
    dictcl = {}
    for val in uniact:
        dictnum.update({val:0})
        dictkey.update({val:'NA'})
        dictcl.update({val:'NA'})

    index = 1
    retfinal = np.zeros([16,35])
    for col in data:
        tocluster = data[col].as_matrix()
#        totest = datatwo[col].as_matrix()
        for i in [16]:
            clusterone = clusterdata(tocluster,i)

            p = purity(act,clusterone)

            index+=1
            if 'Unnamed' not in col:
                for key, value in dictnum.items():
                    if p[key] > dictnum[key]:
                        dictnum[key] = p[key]
                        dictkey[key] = col
                        dictcl[key] = i


    print(dictnum)
    print(dictkey)
    print(dictcl)
    return np.unique(list(dictkey.values()))
