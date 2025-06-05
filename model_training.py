import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import pandas as pd


housing = fetch_california_housing()

# data = pd.DataFrame(housing.data,columns=housing.feature_names)
# data['PRICE'] = housing.target
# print(data.head)


# X = data.drop('PRICE', axis=1)
# y = data['PRICE']

# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train_scaled,y_train)


X_train,X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor (n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

joblib.dump(model, 'model.pkl')


print(f"Model trained and saved to model.pkl")

