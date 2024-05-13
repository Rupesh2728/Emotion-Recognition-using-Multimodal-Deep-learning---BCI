import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("features.csv", header=None)
df.head()
data = np.array(df)
print(data.shape)

val = []
for i in range(data.shape[0]):
    lst = []
    for j in range(5):
        if j < 3:
            lst.append(data[i][j])
        else:
            if data[i][j] <= 4:
                lst.append(0)
            else:
                lst.append(1)
    val.append(lst)

arr = np.array(val)
print(arr.shape)

df = pd.DataFrame(arr)
df.columns = ["Activity","Mobility","Complexity","Valence","Arousal"]
print(df.head(40))

df.to_csv('features_new.csv', header=False, index=False)

scaler = MinMaxScaler()
n_df = df
n_df.head()

n_df = pd.DataFrame(scaler.fit_transform(n_df))

n_df.head()
n_df.columns = ["Activity","Mobility","Complexity","Valence","Arousal"]
n_df.to_csv('features_normalized.csv', header=False, index=False)