



import pandas as pd
from sklearn.model_selection import train_test_split


# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../../input/test_iowa.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X



### begin for loop for features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]


### begin for loop for training sizes
# Split into validation and training data
X_train, X_valid, y_train, y_val = train_test_split(X, y, random_state=1)

# %% column drop

# find columns to get rid of beause of null values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

