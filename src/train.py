"""
python 
"""

from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys
import pandas as pd

import utils  # for params access
from utils import load_config
from data_processing import get_dataset, add_features, split_df
from utils import create_dirs, get_output_path, save_params

def train(featured_path=None):
    create_dirs()
    output_path = get_output_path()
    if featured_path is None:
        df = get_dataset(True, False, filename=output_path + 'tgt_distrib.png')
        add_features(df)
    else:
        df = pd.read_csv(featured_path) 
    X_train, X_test, y_train, y_test = split_df(df)
    loss_after_train = train_catboost(X_train, X_test, y_train, y_test, hypoptim=False, save_dir=output_path)
    save_params(utils.params, output_path + 'params.txt', min_loss=loss_after_train)
    
    
def train_catboost(X_train, X_test, y_train, y_test, save_dir='./runs/train1/', hypoptim=True):
    config = load_config()
    train_config = config['train']
    
    if hypoptim:
        param_dict = { 
            'depth': [4, 8],
            'learning_rate': [0.03, 0.3],
            'l2_leaf_reg': [1, 5]
        }
    else:
        param_dict = { 
            'depth': [int(train_config['depth'])],
            'learning_rate': [float(train_config['learning_rate'])],
            'l2_leaf_reg': [int(train_config['l2_leaf_reg'])]
        }
        
    model = CatBoostRegressor(loss_function='RMSE', cat_features=[])

    grid_search = GridSearchCV(estimator=model, param_grid=param_dict, scoring=None, cv=3, verbose=100, n_jobs=-1)

    grid_search.fit(X_train, y_train, verbose=100)

    print(f"Best params: {grid_search.best_params_}")
    y_pred = grid_search.best_estimator_.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')

    os.makedirs(save_dir, exist_ok=True)

    best_model = grid_search.best_estimator_
    best_model.save_model(os.path.join(save_dir, 'model.cbm'))
    best_model.save_model(os.path.join('models', 'model.cbm'))  # for dvc

    print(f"Model save in {os.path.join(save_dir, 'model.cbm')} and in {os.path.join('models', 'model.cbm')}.")
    return mse
    
if __name__ == '__main__':
    argv = sys.argv
    if len(argv) == 2:
        featured_path = argv[1]
    else:
        featured_path = None
    train(featured_path=featured_path)
    