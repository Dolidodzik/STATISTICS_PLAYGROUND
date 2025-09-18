import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
import matplotlib.pyplot as plt
import seaborn as sns
from ISLP.models import ModelSpec, summarize, poly
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression



Caravan = load_data('Caravan')
print(Caravan.Purchase.value_counts())
print(Caravan.columns)

feature_df = Caravan.drop(columns = ['Purchase'])
scaler = StandardScaler(with_mean=True, with_std=True, copy=True)
scaler.fit(feature_df)
X_std = scaler.transform(feature_df)

feature_std = pd.DataFrame(X_std, columns=feature_df.columns)
print("feature std std: ", feature_std.std())

X_train, X_test, y_train, y_test = train_test_split(feature_std, Caravan.Purchase, test_size=1000, random_state=0)
print("x train: \n", X_train)
print("y test: \n ", y_test)

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1_pred = knn1.fit(X_train, y_train).predict(X_test)
print("knn1 accuracy: ", np.mean(knn1_pred == y_test))
print("always NO accuracy: ", np.mean(knn1_pred == 'No'))
print("knn1 conf matrix: ")
print(confusion_matrix(knn1_pred, y_test))

print("tuning hyperparameters: ")
best_acc = 0
for K in range(1, 10):
    print("="*50)
    print(f"KNN with K={K}")
    print("=")
    knn = KNeighborsClassifier(n_neighbors=K)
    knn_pred = knn.fit(X_train, y_train).predict(X_test)
    knn_acc = np.mean(knn_pred == y_test)
    print("accuracy: ", knn_acc)
    if knn_acc > best_acc:
        best_acc = knn_acc
    print("conf matrix: ")
    print(confusion_matrix(knn_pred, y_test))
    print("="*50)
print("best acc found: ", best_acc)

print("="*50)
print("logistic regression")
logit = LogisticRegression(C=1e10, solver='liblinear')
logit.fit(X_train, y_train)
logit_pred = logit.predict_proba(X_test)
logit_labels = np.where(logit_pred[:, 1] > 0.5, 'Yes', 'No')
print("Confusion matrix for logistic regression:")
print(confusion_matrix(y_test, logit_labels))
print("logistic regression accuracy: ", np.mean(logit_labels == y_test))
