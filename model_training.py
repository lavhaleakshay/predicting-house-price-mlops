from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

#Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

#Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

#Save the model
jobline.dump(model, 'model.pkl')


