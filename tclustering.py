import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

df = pd.read_csv('paddydataset.csv')

X = df['Soil Types']

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

model = KMeans(n_clusters=3, random_state=42)
model.fit(X_tfidf)

clusters = model.labels_

df['Cluster'] = clusters

print(df.head())