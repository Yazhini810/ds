import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('data.csv')

X = df   # no target needed

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

clusters = model.labels_

df['Cluster'] = clusters

print(df.head())