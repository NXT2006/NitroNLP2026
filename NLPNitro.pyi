import pandas as pd
import numpy as np

def loadingDataFromCSV():
    print("Loading data from train_data.csv")
    train =pd.read_csv("train_data.csv")
    print("Loading data from test_data.csv")
    test = pd.read_csv("test_data.csv")
    return train,test

def features(df):

    df['word'] = df['word'].fillna(""); #handling empty space in the csv file
    df['word'] = df['word'].astype(str); #making sure everything is a stirng
    df['word_len'] = df['word'].str.len() #new column for wordlen


    return df;

# if __name__ == "__main__":
#     train_df, test_df = loadingDataFromCSV()
#     train_df = features(train_df)
#     test_df = features(test_df)
#     print(train_df.head())





