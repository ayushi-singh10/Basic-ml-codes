import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [20, 22, 23, 25, 27]

model = LinearRegression()
model.fit(X, y)

predicted_temp = model.predict([[6]])
print("Predicted Temperature on Day 6:", predicted_temp[0])

y_pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Actual Temperature')
plt.plot(X, y_pred, color='red', label='Regression Line')

plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('Day vs Temperature')

plt.legend()
plt.show()
