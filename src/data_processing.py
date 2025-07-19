import numpy as np
import os
import shutil
import pandas as pd
import kagglehub
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import params, cosine_sim, get_text_embedding
import logging

class DataProcessor:
    """Handles all data processing operations including loading, feature engineering and splitting"""
    
    def __init__(self, params):
        """Initialize with configuration parameters"""
        self.logger = logging.getLogger(__name__)
        self.params = params
        
        try:
            self.logger.info("Loading word vectors...")
            self.word_vectors = api.load("glove-wiki-gigaword-50")
            self.logger.info("Word vectors loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load word vectors: {str(e)}")
            raise

    def get_dataset(self, dataset_path=None, output=False, visualize=False, filename='distribution.png'):
        """Load and preprocess the dataset"""
        try:
            if not os.path.exists("./data"):
                self.logger.info("Creating data directory")
                os.makedirs("./data")
                
            path = kagglehub.dataset_download("arhamrumi/amazon-product-reviews") if dataset_path is None else dataset_path
            self.logger.info(f"Loading dataset from: {path}")
            df = pd.read_csv(os.path.join(path, 'Reviews.csv'))

            shutil.copy(path + '/Reviews.csv', "./data")
            
            df = df.sample(frac=1, random_state=self.params.random_seed).reset_index(drop=True)[:self.params.all_data_size]
            
            if output:
                self.logger.info("Common information:")
                self.logger.info(df.info())

            missing_values = df.isnull().sum()
            if output:
                self.logger.info("\nNulls:")
                self.logger.info(missing_values[missing_values > 0])

            if output:
                self.logger.info("\nData examples:")
                self.logger.info(df.head())

            df["ProfileName"] = df["ProfileName"].fillna("No text")
            df["Summary"] = df["Summary"].fillna("No text")


            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, palette="coolwarm", legend=False)
            plt.title("Score distribution")
            plt.xlabel("Score")
            plt.ylabel("N reviews")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            
            if output:
                self.logger.info(f"Saving distribution plot to: {filename}")
                plt.savefig(filename)
                
            if visualize:
                plt.show()
            
            return df
        except Exception as e:
            self.logger.error(f"Error in get_dataset: {str(e)}")
            raise

    def add_features(self, df):
        """Add text embedding features to the dataset"""
        try:
            self.logger.info("Adding features to dataset")
            
            good_emb = self.word_vectors["good"]
            bad_emb = self.word_vectors["bad"]

            self.logger.info("Calculating cosine similarities...")
            df["cos_sim_good_text"] = df["Text"].apply(lambda x: cosine_sim(get_text_embedding(x, self.word_vectors), good_emb))
            df["cos_sim_bad_text"] = df["Text"].apply(lambda x: cosine_sim(get_text_embedding(x, self.word_vectors), bad_emb))
            df["cos_sim_good_summary"] = df["Summary"].apply(lambda x: cosine_sim(get_text_embedding(x, self.word_vectors), good_emb))
            df["cos_sim_bad_summary"] = df["Summary"].apply(lambda x: cosine_sim(get_text_embedding(x, self.word_vectors), bad_emb))

            self.logger.info("Feature examples:")
            self.logger.info(df[["Text", "Summary", "cos_sim_good_text", "cos_sim_bad_text", "cos_sim_good_summary", "cos_sim_bad_summary"]].head())

            self.logger.info("Features added successfully")
            return df
        except Exception as e:
            self.logger.error(f"Error in add_features: {str(e)}")
            raise
        
    def split_data(self, df):
        """Split dataset into train and test sets"""
        try:
            self.logger.info("Splitting dataset")
            X = df.drop(columns=["Score"])
            y = df["Score"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.5,
                random_state=self.params.random_seed,
                shuffle=True
            )

            self.logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f"Error in split_data: {str(e)}")
            raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    try:
        logger = logging.getLogger(__name__)
        logger.info("Starting data processing")
        
        from utils import params
        processor = DataProcessor(params)
        
        df = processor.get_dataset(filename='data/tgt_distrib.png')
        df = processor.add_features(df)
        df.to_csv('data/Reviews_featurized.csv', index=False)
        
        logger.info("Data processing completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
