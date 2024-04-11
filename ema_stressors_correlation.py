import numpy as np
from preprocessing import get_ema
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


parent_dir = './Data/In-lab/'
paths = sorted(os.listdir('./Data/In-lab/'))

df_emas = [pd.read_csv(parent_dir + path + '/labeledfeatures.csv') for path in paths]
emas_list = []
for df_ema in df_emas:
    emas, _ = get_ema(df_ema)
    emas_list.append(emas.astype(float))


ema_labels = ['BinaryStress',
              'LikertStress',
              'PSS-Control',
              'PSS-Confident',
              'PSS-Your Way',
              'PSS-Overcome',
              'HappyStress',
              'ExcitedStress',
              'ContentStress',
              'WorriedStress',
              'IrritableStress',
              'SadStress',
              'PSS',
              'Intended']


ema_thresh = [np.mean(emas, axis=0) for emas in emas_list]
pred = [emas >= thresh[None, :] for emas, thresh in zip(emas_list, ema_thresh)]
pred = np.concatenate(pred, axis=0).astype(int)

df = pd.DataFrame(data=pred, columns=ema_labels)

fig = plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, cmap='icefire')
plt.savefig('./Plots/Correlation_ema_intended.png')
plt.show()
