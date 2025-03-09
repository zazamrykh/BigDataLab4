import numpy as np
import os
import pandas as pd
import kagglehub
from sklearn.metrics.pairwise import cosine_similarity 
import gensim.downloader as api
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import utils

def get_dataset(output=False, visualize=False, filename='distribution.png'):
    path = kagglehub.dataset_download("arhamrumi/amazon-product-reviews")
    df = pd.read_csv(os.path.join(path, 'Reviews.csv')) 

    df = df.sample(frac=1, random_state=utils.params.random_seed).reset_index(drop=True)[:utils.params.all_data_size]
    
    if output: print("🔹 Common information:")
    if output: print(df.info())

    missing_values = df.isnull().sum()
    if output: print("\n🔍 Nulls:")
    if output: print(missing_values[missing_values > 0])

    if output: print("\n📊 Data examples:")
    if output: print(df.head())

    df["ProfileName"].fillna("No text", inplace=True)
    df["Summary"].fillna("No text", inplace=True)

    if output: print("\nNulls after filling:")
    null_sum_after_fill = df.isnull().sum()
    if output: print(null_sum_after_fill)

    assert null_sum_after_fill.sum() == 0

    plt.figure(figsize=(8, 5))
    sns.countplot(x=df["Score"], palette="coolwarm")
    plt.title("Score distribution")
    plt.xlabel("Score")
    plt.ylabel("N reviews")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    if output:
        plt.savefig(filename)
        
    if visualize:
        plt.show()
    
    return df


def get_text_embedding(text, model):
    words = text.lower().split() 
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(model.vector_size) 
    return np.mean(vectors, axis=0)


def cosine_sim(vec1, vec2):
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]


def add_features(df):
    print("Load embeddings (50-dim)...")
    word_vectors = api.load("glove-wiki-gigaword-50")
    
    good_emb = word_vectors["good"]
    bad_emb = word_vectors["bad"]

    df["cos_sim_good_text"] = df["Text"].apply(lambda x: cosine_sim(get_text_embedding(x, word_vectors), good_emb))
    df["cos_sim_bad_text"] = df["Text"].apply(lambda x: cosine_sim(get_text_embedding(x, word_vectors), bad_emb))
    df["cos_sim_good_summary"] = df["Summary"].apply(lambda x: cosine_sim(get_text_embedding(x, word_vectors), good_emb))
    df["cos_sim_bad_summary"] = df["Summary"].apply(lambda x: cosine_sim(get_text_embedding(x, word_vectors), bad_emb))

    print(df[["Text", "Summary", "cos_sim_good_text", "cos_sim_bad_text", "cos_sim_good_summary", "cos_sim_bad_summary"]].head())

    df.drop(columns=["Id", "UserId", "ProfileName", "ProductId", "Text", "Summary", 'Time'], inplace=True)
    
    
def split_df(df):
    X = df.drop(columns=["Score"])
    y = df["Score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=utils.params.random_seed, shuffle=True)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test