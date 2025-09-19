# someones elses work on this same dataset - https://www.kaggle.com/code/joyaljosz11/house-price-prediction-using-linear-regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score, KFold



houses_df =  pd.read_csv('House_Price.csv')
print(houses_df)
print(houses_df.describe(include='all'))
print(houses_df.info()) # this suggests that n_hos_beds and waterbody are missing some values (less than 506 are non-null, where 506 is the count of rows)

# corr matrix
sns.heatmap(pd.get_dummies(houses_df).corr(), annot=True, fmt=".2f", cmap='coolwarm')
#plt.show()


# scatterplots
quantitative_cols = houses_df.select_dtypes(include=[np.number]).columns.drop('price')
n_cols = 4
n_rows = 4

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

for i, col in enumerate(quantitative_cols):
    axes[i].scatter(houses_df[col], houses_df['price'], alpha=0.2)
    axes[i].set_ylabel('Price')
    axes[i].set_title(f'Price vs {col}')

plt.tight_layout()
#plt.show()

# bus_ter seems to be 100% yes in this dataset, lets check that
print("value counts for bus_ter:")
print(houses_df['bus_ter'].value_counts()) # YES 506 = 506 which is count of the rows, so we can drop bus_ter

# checking NaNs
print(f"sum of all NaNs: \n {houses_df.isna().sum()}") # this confirms that n_hos_beds and waterbody have some empty rows

# checking for duplicates
print(f"\nNumber of duplicate rows: {houses_df.duplicated().sum()}")

# automatic good possible transformations detections
transformation_selection_df = houses_df.select_dtypes(include=['float64', 'int64'])
num_col_to_test = transformation_selection_df.columns.to_list()
num_col_to_test.remove('price')
y_transformation_selection = transformation_selection_df['price']

def square(x):
    return x ** 2

def linear(x):
    return x

def natural_log(x):
    return np.log1p(np.where(x > -1, x, np.nan))

def one_over_x(x):
    if x == 0:
        x+=0.0001
    return 1/x

transformations = [square, linear, natural_log, one_over_x]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for column in num_col_to_test:
    cv_scores = []
    for transformation in transformations:
        X = pd.DataFrame({column: transformation_selection_df[column].apply(transformation)})
        
        single_feature_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        scores = cross_val_score(single_feature_pipeline, X, y_transformation_selection, cv=kfold, scoring='neg_mean_squared_error')
        cv_scores.append({
            'transformation': transformation.__name__,
            'score': -scores.mean()
        })
    cv_scores.sort(key=lambda x: x['score'], reverse=False)
    
    print(f"\nColumn: {column}")
    print("-" * 40)
    for i, result in enumerate(cv_scores):
        print(f"{i+1}. {result['transformation']:15} MSE: {result['score']:.4f}")

# synergies detection

print("quitting")
quit()

# EDA !TODO! - automated/manual synergies detection, VIFS
'''
# conclusions, EDA or related to data cleaning:
- n_hos_beds missing data we will have to do KNN imputation
- waterbody - we can either drop the column, or mark NaNs as the unknown
- bus_ter - we have to drop it
- dist1 2,3,4 all have very big correlation factors, like over 0.99, which means we can try dropping 3 of them and only keeping 1 for predictions
Scatterplots analysis, predictor plotted against price:
- scatterplot hyperbola shape (like 1/x) - crime_rate, poor_prop
- slower increase, like ln(x) or sqrt(x) - dist1, dist2, dist3, dist4, parks
- parabola shape, like x^2 - maybe age, not really sure tho
=======
Automated transformations results, what might use what transformations:
ln(x) - crime_rate, poor_prop
x^2   - room_num, age
1/x   - one_over_x, dist1
'''

# here we will split datas to validation and test sets, for easier comparisons
houses_df_clean = houses_df.drop(columns=['bus_ter'])
X = houses_df_clean.drop('price', axis=1)
y = houses_df_clean['price']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42
)

train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
train_df.to_csv('train.csv', index=False)
val_df.to_csv('validation.csv', index=False)

# some quick and dirty training, dummy model and simplistic OLS just to set some baseline performance of R2 and MSE on test set.
X_train = X_train.drop(columns=['waterbody'])
X_val = X_val.drop(columns=['waterbody'])

numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

lr = LinearRegression()
lr.fit(X_train_processed, y_train)
y_pred_lr = lr.predict(X_val_processed)
mse_lr = mean_squared_error(y_val, y_pred_lr)
r2_lr = r2_score(y_val, y_pred_lr)

print(f"Simplistic Linear Regression - Validation MSE: {mse_lr:.2f}, R²: {r2_lr:.4f}")

dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train_processed, y_train)
y_pred_dummy = dummy.predict(X_val_processed)
mse_dummy = mean_squared_error(y_val, y_pred_dummy)
r2_dummy = r2_score(y_val, y_pred_dummy)

print(f"Dummy Model - Validation MSE: {mse_dummy:.2f}, R²: {r2_dummy:.4f}")

'''
scores:
Simplistic Linear Regression - Validation MSE: 25.98, R²: 0.6312
Dummy Model - Validation MSE: 72.82, R²: -0.0336
'''