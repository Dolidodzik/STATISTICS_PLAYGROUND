import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
import matplotlib.pyplot as plt
import seaborn as sns
from ISLP.models import ModelSpec, summarize, poly
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB


Smarket = load_data('Smarket')
Smarket = pd.get_dummies(Smarket, columns=['Direction'])
print(Smarket.columns)
print(Smarket.dtypes)
print(Smarket.corr())

condition = Smarket.Year < 2005
Smarket_train = Smarket[condition]
Smarket_test = Smarket[~condition]
print("training: ")
print(Smarket_train.shape)
print("====")
print("test:")
print(Smarket_test.shape)

terms = ['Lag1', 'Lag2']
design = ModelSpec(terms)
X_train, X_test = design.fit_transform(Smarket_train), design.fit_transform(Smarket_test)
y_train = Smarket_train.Direction_Up
glm = sm.GLM(y_train, X_train, family=sm.families.Binomial())
results = glm.fit()
print(summarize(results))

probs = results.predict(exog=X_test)
print(probs)

labels = np.zeros(len(probs))
labels[probs > 0.5] = 1
print(labels)

print(confusion_matrix(labels, Smarket_test.Direction_Up))
print("accuracy of logistic regression model: ", np.mean(labels == Smarket_test.Direction_Up))

labels_up = np.ones(len(probs))
print(confusion_matrix(labels_up, Smarket_test.Direction_Up))
print("accuracy of always up model: ", np.mean(labels_up == Smarket_test.Direction_Up))

print("predictions for some handpicked data:")
newdata = pd.DataFrame ({
    'Lag1':[1.2 , 1.5], 
    'Lag2':[1.1 , -0.8]
})

newX = design.transform(newdata)
predictions = results.predict(newX)
print(predictions)

# Linear Discriminant Analysis
print("===================\n doing LDA: \n ====================")
X_train = X_train.drop(columns=['intercept'])
X_test = X_test.drop(columns=['intercept'])
L_train = np.where(Smarket_train.Direction_Up == 1, 'Up', 'Down')
L_test = np.where(Smarket_test.Direction_Up == 1, 'Up', 'Down')

lda = LinearDiscriminantAnalysis(store_covariance=True)
lda.fit(X_train, L_train)
'''
print("lda means: ", lda.means_)
print(lda.classes_)
print(lda.priors_)
print("lda scalings: ", lda.scalings_)
'''
lda_pred = lda.predict(X_test)
print("lda confusion_matrix")
print(confusion_matrix(lda_pred, L_test))
lda_prob = lda.predict_proba(X_test)
#print("lda probs: ", lda_prob)

lda_accuracy = accuracy_score(L_test, lda_pred)
print("LDA ACCURACY: ", lda_accuracy)



# QDA
print("===================\n doing QDA: \n ====================")
print(X_train)
print(L_train)
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(X_train, L_train)

print("Covariance matrix for class 0:")
print(qda.covariance_[0])
print("Covariance matrix for class 1:")
print(qda.covariance_[1])

print("qda means:")
print(qda.means_)
print("qda priors:")
print(qda.priors_)

qda_pred = qda.predict(X_test)
print("qda conf matrix: ")
print(confusion_matrix(L_test, qda_pred))
qda_accuracy = np.mean(qda_pred == L_test)
print("qda accuracy: ")
print(qda_accuracy)



# Naive Bayes
print("===================\n doing Naive Bayes: \n ====================")

NB = GaussianNB()
NB.fit(X_train, L_train)
print("NB classes:")
print(NB.classes_)
print("NB class prior:")
print(NB.class_prior_)
print("NB variance: ")
print(NB.var_)

nb_labels = NB.predict(X_test)
print("confustion table NB:")
print(confusion_matrix(nb_labels, L_test))
nb_accuracy = np.mean(nb_labels == L_test)
print("nb acc: ", nb_accuracy)