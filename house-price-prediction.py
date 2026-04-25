import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("housing.csv")

X = df[['RM']]
y = df['MEDV']

model = LinearRegression()
model.fit(X, y)

print(model.predict([[6]]))
