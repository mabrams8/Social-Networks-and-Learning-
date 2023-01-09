import pandas as pd
import seaborn as sn
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
# import preprocessed data
df = pd.read_csv("network-cleaned -possible - help-4605.csv")
# remove unnecessary columns
df = df.drop(columns=['Timestamp', 'raw entries'])
# *Goal: correlate the students concerns (1-5) to student challenges (11-15)

# Student Concern Extraction from network data
cols = (df.columns.values)
df.columns = cols
#print(df)
rows = df.shape[0]
cols = df.shape[1]
for j in range(1, cols):
    for i in range(rows):
        if type(df.iloc[i,j]) == str:
            # check if its less than 6
            if len(df.iloc[i,j]) >= 1:
                keepValue = 100
                # just get the smallest number
                values = df.iloc[i,j].split(',')
                for value in values:
                    if int(value) <= 5:
                        if int(value) <= keepValue:
                            keepValue = int(value)
                if (keepValue <= 5) and (keepValue < 100):
                    df.iloc[i,j] = keepValue
                else:
                    df.iloc[i,j] = math.nan
            else:
                if int(df.iloc[i,j]) <= 5:
                    df.iloc[i,j] = int(values[0])
                else:
                    df.iloc[i,j] = math.nan
        else:
            if df.iloc[i,j] > 5:
                df.iloc[i,j] = math.nan
#print(df)
indices = range(1,21)
df.to_csv("post_processed_concerns.csv", index = False)

# Postprocessed Data from Part C
df2 = pd.read_csv("post_processed.csv")
# Extract student features vectors from df and df2
df2 = df2.drop(columns=['Timestamp', 'raw entries'])
df3 = pd.DataFrame(df2)
def distance(a, b):
    if (a == b):
        return 0
    elif (a < 0) and (b < 0) or (a > 0) and (b > 0):
        if (a < b):
            return (abs(abs(a) - abs(b)))
        else:
            return (abs(abs(a) - abs(b)))
    else:
        return math.copysign((abs(a) + abs(b)),b)
for i in range(rows):
    for j in range(1, cols):
        if math.isnan(df.iloc[i, j]) or math.isnan(df2.iloc[i, j]):
            df3.iloc[i, j] = 0
        elif math.isnan(df.iloc[i, j]) and math.isnan(df2.iloc[i, j]):
            df3.iloc[i, j] = 0
        else:
            if float(df.iloc[i,j])>=1 and float(df2.iloc[i,j])>=1:
                if ((distance(df2.iloc[i,j],df.iloc[i,j]))==10):
                    df3.iloc()[i,j] = 1
                elif  ((distance(df2.iloc[i,j],df.iloc[i,j]))==11):
                    df3.iloc[i, j] = 0.8
                elif ((distance(df2.iloc[i, j], df.iloc[i, j])) == 12):
                    df3.iloc[i, j] = 0.6
                elif ((distance(df2.iloc[i, j], df.iloc[i, j])) == 13):
                    df3.iloc[i, j] = 0.4
                elif ((distance(df2.iloc[i, j], df.iloc[i, j])) == 14):
                    df3.iloc[i, j] = 0.2
                elif ((distance(df2.iloc[i, j], df.iloc[i, j])) == 9):
                    df3.iloc[i, j] = -0.8
                elif ((distance(df2.iloc[i, j], df.iloc[i, j])) == 8):
                    df3.iloc[i, j] = -0.6
                elif ((distance(df2.iloc[i, j], df.iloc[i, j])) == 7):
                    df3.iloc[i, j] = -0.4
                elif ((distance(df2.iloc[i, j], df.iloc[i, j])) == 6):
                    df3.iloc[i, j] = -0.2
                else:
                    df3.iloc[i, j] = 0
df3.to_csv("post_processed_correlation.csv", index = False)
df3 = df3.drop(columns=['Sample number'])
sn.heatmap(df3, annot=False)
plt.show()

