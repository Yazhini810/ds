import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = []
labels = []

path = "dataset/train"

for folder in os.listdir(path):
    for img in os.listdir(path + "/" + folder):
        img_path = path + "/" + folder + "/" + img
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64,64))
        data.append(image.flatten())
        labels.append(folder)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

model = SVC()
model.fit(X_train, y_train)

ans = model.predict(X_test)

print(accuracy_score(y_test, ans))