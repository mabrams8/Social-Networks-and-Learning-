import pandas as pd
import math

# import csv
df = pd.read_csv("network-cleaned -possible - help-4605.csv")
df2 = pd.read_csv("network-cleaned -possible - help-4605.csv")
#df = df.drop(columns=['Sample number', 'Timestamp', 'raw entries'])
cols = (df.columns.values)
cols2 = (df2.columns.values)
df.columns = cols
df2.columns = cols2
#print(df)
rows = df.shape[0]
cols = df.shape[1]
for j in range(3, cols):
    for i in range(rows):
        if type(df.iloc[i,j]) == str:
            # check if its greater than 11
            if len(df.iloc[i,j]) > 1:
                keepValue = 100
                # just get the smallest number
                values = df.iloc[i,j].split(',')
                for value in values:
                    if int(value) >= 11:
                        if int(value) <= keepValue:
                            keepValue = int(value)
                if keepValue >= 11 and keepValue < 100:
                    df.iloc[i,j] = keepValue
                else:
                    df.iloc[i,j] = 0
            else:
                if int(df.iloc[i,j]) >= 11:
                    df.iloc[i,j] = int(df.iloc[i,j])
                else:
                    df.iloc[i,j] = 0
        else:
            if df.iloc[i,j] < 11:
                df.iloc[i,j] = 0
            elif math.isnan(df.iloc[i,j]):
                df.iloc[i,j] = 0
print(df)
df.to_csv("post_processed_11to15.csv", index = False)

rows = df2.shape[0]
cols = df2.shape[1]
for j in range(3, cols):
    for i in range(rows):
        if type(df2.iloc[i,j]) == str:
            # check if its greater than 1 and less than 5
            if len(df2.iloc[i,j]) > 1:
                keepValue = 100
                # just get the smallest number
                values = df2.iloc[i,j].split(',')
                for value in values:
                    if int(value) >= 1 and int(value) <= 5:
                        if int(value) <= keepValue:
                            keepValue = int(value)
                if keepValue >= 1 and keepValue <= 5 and keepValue < 100:
                    df2.iloc[i,j] = keepValue
                else:
                    df2.iloc[i,j] = 0
            else:
                if int(df2.iloc[i,j]) >= 1 and int(df2.iloc[i,j]) <= 5:
                    df2.iloc[i,j] = int(df2.iloc[i,j])
                else:
                    df2.iloc[i,j] = 0
        else:
            if df2.iloc[i,j] > 5:
                df2.iloc[i,j] = 0
            elif math.isnan(df2.iloc[i,j]):
                df2.iloc[i,j] = 0
print(df2)
df2.to_csv("post_processed_1to5.csv", index = False)