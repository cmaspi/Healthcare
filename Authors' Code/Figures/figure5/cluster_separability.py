import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

sns.set(style="ticks", color_codes=True)
response = ["Stress", "No Stress"]

df = pd.read_csv("./pseudo.csv")
length = 200

times = df.time.values.tolist()
new_t = []
new_act = []
new_f = []
intended = []

i = 0
val = 200
index = 0
on = True
while i < len(times)/3:
	adding = random.uniform(val, val+150)
	new_t.append(adding)
	new_f.append("mean_IBI")
	if adding > 310:
		if int(adding) % 22 == 0 or int(adding) % 13 == 0:
			intended.append(response[1])
		else:
			intended.append(response[0])
	elif adding >= 260:
		if int(adding) % 7 == 0 or int(adding) % 11 == 0:
			intended.append(response[1])
		else:
			intended.append(response[0])
	else:
		if int(adding) % 22 == 0 and int(adding) % 8 == 0:
			intended.append(response[0])
		else:
			intended.append(response[1])
	i = i + 1

val = 200
index = 0
on = True
while i < 2 * (len(times)/3):
	adding = random.uniform(val, val+150)
	new_t.append(adding)
	new_f.append("min_IBI")
	if adding >= 325:
		intended.append(response[0])
	elif adding >= 290:
		if int(adding) % 5 == 0 or int(adding) % 7 == 0 or int(adding) % 6 == 0:
			intended.append(response[0])
		else:
			intended.append(response[1])
	else:
		if (int(adding)**2) %23 == 0:
			intended.append(response[0])
		else:
			intended.append(response[1])
	i = i + 1

val = 200
index = 0
while i < 3 * (len(times)/3):
	adding = random.uniform(val, val+150)
	new_t.append(adding)
	new_f.append("max_IBI")
	if int(adding) % 5 == 0 or int(adding) % 3 == 0:
		intended.append(response[0])
	else:
		intended.append(response[1])
	i = i + 1

df['Feature Value (IBI in ms)'] = new_t
df['Feature'] = new_f
df["Intended Stress"] = intended

sns.catplot(x='Feature Value (IBI in ms)', y='Feature', hue='Intended Stress', kind = "swarm", data=df)
plt.xticks([200,250, 300, 350])
props = dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.8)
plt.text(200, -0.4, "Good separability", bbox = props)
plt.text(295, 2.4, "Poor separability",bbox = props)
plt.show()



print(df.head())