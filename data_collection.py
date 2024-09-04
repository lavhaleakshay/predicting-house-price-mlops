from sklearn.datasets import load_boston
import pandas as pd

#load the dataset
boston = load_dataset()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

#save the dataset to csv file
data.to_csv('data.csv', index=False)

