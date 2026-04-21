import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv('data.csv')

X = df.drop('target', axis=1)   # input features (numeric)
y = df['target']                # categorical output

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

ans = model.predict(X_test)

print(accuracy_score(y_test, ans))
print(f1_score(y_test, ans, average='weighted'))