import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme('paper')
df = pd.read_csv('./Results/loso_chunks.csv')

fig = plt.figure(figsize=(8, 6))
sns.boxplot(df)
plt.xticks(rotation=30, fontsize=10)
plt.ylabel('F1 score')
plt.tight_layout()
plt.savefig('./Plots/loso.png')
plt.show()
