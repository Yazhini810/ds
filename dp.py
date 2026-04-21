import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("paddydataset.csv")
print(df.isnull().sum())
df = df.fillna(df.mean(numeric_only=True))

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = X.select_dtypes(include=['int64', 'float64'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Preprocessing Done ")
print(df.isnull().sum())