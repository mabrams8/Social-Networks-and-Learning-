import pandas as pd
import math
from mf import MF
import numpy as np

# import csv
df1 = pd.read_csv("post_processed_11to15.csv")
df2 = pd.read_csv("post_processed_1to5.csv")
df1 = df1.drop(columns=['Sample number', 'Timestamp', 'raw entries'])
df2 = df2.drop(columns=['Sample number', 'Timestamp', 'raw entries'])
# remove any students that didn't leave any ratings at all for the specified 1-5 or 11-15 ranges
df1 = df1.loc[~(df1==0).all(axis=1)]
df2 = df2.loc[~(df2==0).all(axis=1)]

# convert dataframe to numpy array
data_array = df1.to_numpy()
# do training for 11 to 15 data
mf = MF(data_array, K=2, alpha=0.1, beta=0.01, iterations=10)
training_process = mf.train()
#print(mf.full_matrix())

# convert dataframe to numpy array
data_array2 = df2.to_numpy()
# do training for 1 to 5 data
mf2 = MF(data_array2, K=2, alpha=0.1, beta=0.01, iterations=10)
training_process2 = mf2.train()
#print(mf2.full_matrix())
#data_11to15 = mf.full_matrix().tolist()
data_11to15 = mf.full_matrix()
df1 = pd.DataFrame(data_11to15, columns = df1.columns)
#print(data_11to15)
#print(len(data_11to15[0]))
#data_1to5 = mf2.full_matrix().tolist()
data_1to5 = mf2.full_matrix()
df2 = pd.DataFrame(data_1to5, columns = df2.columns)

# clean the estimated data for upper and lower limits on each
rows = df1.shape[0]
cols = df1.shape[1]

# first clean the 11 to 15 data
for j in range(cols):
    for i in range(rows):
        if df1.iloc[i,j] < 11:
            df1.iloc[i,j] = 11
        elif df1.iloc[i,j] > 15:
            df1.iloc[i,j] = 15

# clean the estimated data for upper and lower limits on each
rows = df2.shape[0]
cols = df2.shape[1]

# clean the 1 to 5 data
for j in range(cols):
    for i in range(rows):
        if df2.iloc[i,j] < 1:
            df2.iloc[i,j] = 1
        elif df2.iloc[i,j] > 5:
            df2.iloc[i,j] = 5

df1.to_csv("learned_ranges_11to15_after_factorization.csv", index = False)
df2.to_csv("learned_ranges_1to5_after_factorization.csv", index = False)
'''
# the below is for grouping results as tuples for each student (each tuple represents a category and a rating of 1-5 and 11-15)
compiled_ratings_learned = []
for i in range(0,len(data_11to15)):
    compiled_ratings_learned.append([])
    for j in range(0,len(data_11to15[i])):
        compiled_ratings_learned[i].append((data_1to5[i][j], data_11to15[i][j]))
#print(compiled_ratings_learned)
'''