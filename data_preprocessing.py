from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#load the dataset
data = pd.read_csv('boston_housing.csv')

#split the data into features and target
X = data.drop(columns=['PRICE'])
y = data['PRICE']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Save the scaler for later use
import joblib
joblib.dump(scaler, 'scaler.pkl')

