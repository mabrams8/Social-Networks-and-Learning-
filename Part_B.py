import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
# Part A - Node Graph of new data (11-15 from given dataset) layered with old data (1-5 from Project 1)
# import csv
df = pd.read_csv("network-cleaned -possible - help-4605.csv")
# drop the uncessary columns for easier data processing
df = df.drop(columns=['Sample number', 'Timestamp', 'raw entries'])
data = {}
cols = (df.columns.values)
for index in range(0,len(cols)):
	cols[index] = cols[index]+"2"
df.columns = cols
#init empty dictionary
for category in cols:
	data[category] = [0, 0]

# go through the df and get vals > 11
for col in df.columns:
	for value in df[col]:
		if type(value) == str:
			val = value.split(',')
			for index in range(0,len(val)):
				val[index] = int(val[index])
				if val[index] >= 11:
					data[col][0] += (val[index] - 10)
					data[col][1] += 1
		else:
			# treat it as a float
			if value >= 11:
				data[col][0] += value - 10
				data[col][1] += 1

for key in list(data):
	if(data[key][1] == 0):
		del data[key]
		continue
#print(data)

#print(data)
stats = pd.read_csv("ECE4605_preprocessed_data_self.csv")
stats = stats.drop(columns=['Sample number', 'Time Stamp', 'Raw entries'])
stats2 = stats.describe()
#generate size list for nodes
node_size_proj1 = stats2.count() * stats2.mean()

proj3CalculatedVals = []
proj3Challenges = []
for key in data:
	proj3CalculatedVals.append(data[key][0] * data[key][1])
	proj3Challenges.append(key)

node_size_proj3 = pd.Series(proj3CalculatedVals, index = proj3Challenges)
#print(node_size_proj3)

# combine the node_size series together.
overall_node_size = node_size_proj1.append(node_size_proj3)

proj1Challenges = []
for challenge, value in node_size_proj1.iteritems():
	proj1Challenges.append(challenge)

#add graph and nodes
G = nx.Graph()
G.add_nodes_from(proj1Challenges)
# add edges for proj1 graph
for index, row in stats.iterrows():
	#print(row)
	#exit()
	selections = stats.columns.values[row <= 5.0]
	#print(selections)
	for i in range (0, len(selections)-1):
		G.add_edge(selections[i], selections[i+1])

# add nodes for proj3
G.add_nodes_from(proj3Challenges)
# add edges for proj3 graph
for index, row in df.iterrows():
	# convert str type entries to float
	for challenge, entry in row.iteritems():
		if type(row[challenge]) == str:
			# split and convert to float
			values = row[challenge].split(',')
			for index in range(0,len(values)):
				values[index] = float(values[index])
			for number in values:
				if number>=11.0 and number <= 15.0:
					values = number
					break
			if type(values) == list:
				values = 0 # don't care since its not between 11 and 15
			row[challenge] = values
	selections = df.columns.values[row >= 11.0]
	#print(selections)
	for i in range (0, len(selections)-1):
		G.add_edge(selections[i], selections[i+1])

#nx.draw(G, node_size=overall_node_size, with_labels=True)
#plt.show() # display

# Part B - Clustering Based on new data (11-15 and 1-5 in given dataset)
rows = df.shape[0]
cols = df.shape[1]

grouped = []
for j in range(cols):
	for i in range(rows):
		# need to catch strings
		if type(df.iloc[i,j]) == str:
			if len(df.iloc[i,j]) > 1:
				# get the largest value
				values = df.iloc[i,j].split(',')
				values.sort()
				largest = int(values[-1])
				if largest >= 11:
					grouped.append((j,largest-10))
			else:
				value = int(df.iloc[i,j])
				if value >= 11:
					grouped.append((j,largest-10))
		else:
			if df.iloc[i,j] >= 11:
				grouped.append((j,df.iloc[i,j]))
tupleChart = np.array(grouped)
print(tupleChart)

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=1000, n_init=10, random_state=0)
pred_y = kmeans.fit(tupleChart)
# print(pred_y)
#tuple chart (feature number, student rating)
X = tupleChart
plt.scatter(X[:,0], X[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')
plt.show()