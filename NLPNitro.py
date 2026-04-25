import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def loadingDataFromCSV():
    print("Loading data from train_data.csv")
    train = pd.read_csv("train_data.csv")
    return train


def features(df):

    df['word'] = df['word'].fillna(""); #handling empty space in the csv file
    df['word'] = df['word'].astype(str); #making sure everything is a stirng
    df['word_len'] = df['word'].str.replace(r'[^\w\s]', '', regex=True).str.len() #new column for wordlen
    marks = ('.', ',', '!', '?', ';', ':')
    df['is_punctuation'] = df['word'].str.endswith(marks).astype(int) #checking for punctuation at the end
    counts = df['word'].value_counts() #storing word counts
    df['word_freq'] = df['word'].map(counts.to_dict())
    df['word_log_freq'] = np.log1p(df['word_freq'])
    df['syllable_count'] = df['word'].str.count(r'[aeiouăîâAEIOUĂÎÂ]') #more syllables = longer to read/prnounce
    df['word_index'] = df['word_id'].str.split('_').str[-1].fillna(0).astype(int);
    df['is_proper_noun'] = ((df['word'].str.istitle()) & (df['word_index'] > 0)).astype(int)
    return df;

if __name__ == "__main__":
    train_df = loadingDataFromCSV()
    train_df = features(train_df)

    print("\n--- FULL TRAINING DATA VIEW ---")
    print(train_df.head(10))

    print("\n--- COLUMN SUMMARY ---")
    print(train_df.info())
    print(train_df.groupby('is_punctuation')['answer'].mean())







