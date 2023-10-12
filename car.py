def remove_str(string):
    if isinstance(string, str):
        element = ''
        for col in string:
            if col.isdigit() or (col == '.' and '.' not in element):
                element = element + col
        try:
            return float(element) if element else None  # Use float to handle decimal values
        except ValueError:
            return None
    else:
        return string
#!-------------------------------------------------------MAIN---------------------------------------------#
if __name__ == '__main__':
    import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import re
import matplotlib.pyplot  as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


#Import dataset
data = pd.read_csv('https://raw.githubusercontent.com/sukhioo7/dataset/main/Car.csv')
dataset = data.drop(['Unnamed: 0','name','seller_type','owner','mileage','torque'],axis=1)

#!========================================Pre-Proecssing========================================

numeric_columns = dataset.select_dtypes(include=[np.number]).columns
catagorical_columns = dataset.select_dtypes(include=[np.number]).columns

#*Imputing ->
numerical_imputer = SimpleImputer(strategy='mean')
dataset[numeric_columns] = numerical_imputer.fit_transform(dataset[numeric_columns])

categorical_imputer = SimpleImputer(strategy='most_frequent')
dataset[catagorical_columns] = categorical_imputer.fit_transform(dataset[catagorical_columns])

#apply
clean_columns = ['engine','max_power']
for col in clean_columns:
    dataset[col] = dataset[col].apply(remove_str)
    
# Define the columns with missing values
columns_with_missing = ["engine", "max_power"]
numerical_imputer = SimpleImputer(strategy='mean')
dataset[columns_with_missing] = numerical_imputer.fit_transform(dataset[columns_with_missing])

#OneHotEncoding ->
One = OneHotEncoder(sparse=False)
enco_columns = ['fuel', 'transmission']

One.fit(dataset[enco_columns])
encoded_col= One.transform(dataset[enco_columns])
temp = pd.DataFrame(encoded_col,columns=One.get_feature_names_out(enco_columns))

dataset = pd.concat([dataset,temp],axis=1)
dataset.drop(columns=enco_columns,inplace=True)

#!--------------------------------------Model----------------------------------------#

X = dataset.drop('selling_price',axis=1)
Y = dataset['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

#scaling->
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#^GridSearchCV-------------->
rf_model = RandomForestRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:")
print(grid_search.best_params_)

best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)


mse_best = mean_squared_error(y_test, y_pred)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred)

print(f"Mean Squared Error (Best Model): {mse_best}")
print(f"Root Mean Squared Error (Best Model): {rmse_best}")
print(f"R-squared (Best Model): {r2_best}")


