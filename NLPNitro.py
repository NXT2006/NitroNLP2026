import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
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


def calculate_score(y_true, y_pred):
    # 1. Calculate R-squared (How close are the guesses?)
    r2 = max(0, r2_score(y_true, y_pred))

    # 2. Calculate Pearson Correlation (Do they go up and down together?)
    pearson = np.abs(pearsonr(y_true, y_pred)[0])
    if np.isnan(pearson): pearson = 0

    # 3. The Hackathon Formula
    return 100 * (r2 + pearson) / 2


if __name__ == "__main__":
    train_df = loadingDataFromCSV()
    train_df = features(train_df)

    print("\n--- FULL TRAINING DATA VIEW ---")
    print(train_df.head(10))

    print("\n--- COLUMN SUMMARY ---")
    print(train_df.info())
    print(train_df.groupby('is_punctuation')['answer'].mean())

    features_list = ['word_len', 'is_punctuation', 'word_freq', 'word_log_freq', 'syllable_count', 'word_index', 'is_proper_noun']
    X = train_df[features_list]
    y = train_df['answer']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"I have {len(X_train)} rows to study and {len(X_val)} rows for the exam!")
    print("Training the XGBoost brain... this might take a few seconds.")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

    model.fit(X_train, y_train)
    print("Training complete!")

    val_predictions = model.predict(X_val)

    final_score = calculate_score(y_val, val_predictions)
    print(f"\n YOUR CURRENT SCORE: {final_score:.2f} / 100")








