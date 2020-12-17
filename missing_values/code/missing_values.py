



import pandas as pd
from sklearn.model_selection import train_test_split


# %% Gather Data

X_full = pd.read_csv('../../input/train_iowa.csv', index_col='Id')
X_test_full = pd.read_csv('../../input/test_iowa.csv', index_col='Id')

# %% Select Data

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)


# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])


# %% begin for loop for training sizes

# Split into validation and training data
# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# take a peak
print(X_train.head())

# %% column drop

# find columns to get rid of beause of null values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# %% Investigation

# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# Total missing values in X_train
print(missing_val_count_by_column[missing_val_count_by_column > 0].sum())