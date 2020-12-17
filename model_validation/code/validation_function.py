
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# %% Iterate and find best

def print_all_models(models,  X_train, X_valid, y_train, y_valid):
    for i in range(0, len(models)):
        print('I am on mae loop: ' + str(i))
        mae = score_model(models[i], X_train, X_valid, y_train, y_valid)
        print("Model %d MAE: %d" % (i+1, mae))
        print()

if __name__ == '__main__':
    
    # %% Gather Data
    # Path of the file to read. We changed the directory structure to simplify submitting to a competition
    
    X_full = pd.read_csv('../../input/train_iowa.csv', index_col='Id')
    X_test_full = pd.read_csv('../../input/test_iowa.csv', index_col='Id')
    
    #iowa_file_path = '../../input/train_iowa.csv'
    
    # %% Select Data
    
    # Create target object and call it y
    y = X_full.SalePrice
    # Create X
    
    
    
    # Obtain target and predictors
    y = X_full.SalePrice
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = X_full[features].copy()
    X_test = X_test_full[features].copy()
    
    # %% Split Data
    
    ### begin for loop for training sizes
    # Split into validation and training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                          random_state=0)
    
    # %% setup different models to validate
    
    # Define the models
    model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
    model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
    model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
    model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
    model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
    
    models = [model_1, model_2, model_3, model_4, model_5]
    
    # iterate models and print out mean absolute error
    print_all_models(models,  X_train, X_valid, y_train, y_valid)
        
    # %% Define optimized model from the data above
    
    my_model = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
    
    # Fit the model to the training data
    my_model.fit(X, y)
    
    # Generate test predictions
    preds_test = my_model.predict(X_test)
    
    # Save predictions in format used for competition scoring
    output = pd.DataFrame({'Id': X_test.index,
                           'SalePrice': preds_test})
    output.to_csv('../output/submission.csv', index=False)


