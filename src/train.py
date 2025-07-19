"""
Model training module with OOP interface
"""

import logging
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys
import pandas as pd

from utils import load_config, create_dirs, get_output_path, save_params, params
from data_processing import DataProcessor

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config_path="config.ini"):
        """Initialize with configuration"""
        self.logger = logging.getLogger(__name__)
        self.config = load_config(config_path)
        self.data_processor = DataProcessor(params)
        self.model = None
        self.best_params = None

    def prepare_data(self, featured_path=None):
        """Load and prepare training data"""
        try:
            create_dirs()
            output_path = get_output_path()
            
            if featured_path is None:
                self.logger.info("Loading and processing dataset from scratch")
                df = self.data_processor.get_dataset(True, False, filename=output_path + 'tgt_distrib.png')
                df = self.data_processor.add_features(df)
            else:
                self.logger.info(f"Loading pre-processed dataset from: {featured_path}")
                df = pd.read_csv(featured_path)
                
            return self.data_processor.split_data(df)
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise

    def train_model(self, X_train, X_test, y_train, y_test, hypoptim=False):
        """Train model with optional hyperparameter optimization"""
        try:
            train_config = self.config['train']
            
            if hypoptim:
                self.logger.info("Performing hyperparameter optimization")
                param_dict = {
                    'depth': [4, 8],
                    'learning_rate': [0.03, 0.3],
                    'l2_leaf_reg': [1, 5]
                }
            else:
                self.logger.info("Using fixed hyperparameters")
                param_dict = {
                    'depth': [int(train_config['depth'])],
                    'learning_rate': [float(train_config['learning_rate'])],
                    'l2_leaf_reg': [int(train_config['l2_leaf_reg'])]
                }
                
            model = CatBoostRegressor(loss_function='RMSE', cat_features=[])

            self.logger.info("Starting grid search")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_dict,
                scoring=None,
                cv=3,
                verbose=100,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train, verbose=100)

            self.best_params = grid_search.best_params_
            self.model = grid_search.best_estimator_
            self.logger.info(f"Best params found: {self.best_params}")

            return self.evaluate_model(X_test, y_test)
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        try:
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            self.logger.info(f'Mean Squared Error (MSE): {mse}')
            self.logger.info(f'Mean Absolute Error (MAE): {mae}')

            return mse
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise

    def save_model(self, save_dir='./runs/train1/'):
        """Save trained model to disk"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, 'model.cbm')
            self.model.save_model(model_path)
            self.model.save_model(os.path.join('models', 'model.cbm'))  # for dvc

            self.logger.info(f"Model saved to {model_path} and models/model.cbm")
            save_params(params, os.path.join(save_dir, 'params.txt'), min_loss=self.best_score)
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    try:
        logger = logging.getLogger(__name__)
        logger.info("Starting training script")
        
        argv = sys.argv
        featured_path = argv[1] if len(argv) == 2 else None
        
        trainer = ModelTrainer()
        X_train, X_test, y_train, y_test = trainer.prepare_data(featured_path)
        trainer.best_score = trainer.train_model(X_train, X_test, y_train, y_test)
        trainer.save_model()
        
        logger.info("Training script completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise