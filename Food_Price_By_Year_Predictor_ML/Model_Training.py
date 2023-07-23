import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import explained_variance_score as evs

from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor  
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor


# Data Import 
df = pd.read_csv('Food_Prices.csv')


# EDA
print(df.info(), ('#' * 100))

df = df.drop('Average Price ', axis= 1)
df = df.drop('Currency ', axis= 1)
print(df.info(), ('#' * 100))


# Data transformation
le = LabelEncoder()

for c in df.columns:
    if df[c].dtype == 'object' :
        df[c] = le.fit_transform(df[c])
    
print(df.dtypes, ('#' * 100))


# Train Test Split
features = df.drop('Price in USD', axis= 1)
target = df['Price in USD']

print(features, ('-' * 100))
print(target, ('#' * 100))

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size= 0.20, random_state= 42)


# Model Training
models = [DecisionTreeRegressor(), ExtraTreeRegressor(), RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor()]

for m in models:
    print(m)
    m.fit(X_train,Y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuracy is : {evs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy is : {evs(Y_test, pred_test)}')