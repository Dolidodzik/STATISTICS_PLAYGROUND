import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression



students_df =  pd.read_csv('student-mat.csv')
print(students_df)

'''
# this shows us that G1, G2, G3 are very highly correlated, that is because they all are grades from different periods
# hence we drop G1 and G2, and only focus on G3, the final grade
plt.figure(figsize=(10, 8))
heatmap_df = students_df.drop(students_df.select_dtypes(exclude='number').columns.tolist(), axis=1)
sns.heatmap(heatmap_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.show()
'''
students_df.drop(['G1', 'G2'], axis=1)

X = students_df['studytime'].values.reshape(-1, 1)
y = students_df['G3'].values

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Coefficients:", model.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Study Time')
plt.ylabel('Final Grade (G3)')
plt.show()