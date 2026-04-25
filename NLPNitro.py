import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def loadingDataFromCSV():
    print("Loading data from train_data.csv")
    train = pd.read_csv("train_data.csv")
    print("Loading data from test_data.csv")
    test = pd.read_csv("test_data.csv")
    return train, test


def features(df):

    df['word'] = df['word'].fillna(""); #handling empty space in the csv file
    df['word'] = df['word'].astype(str); #making sure everything is a stirng

    df['word_len'] = df['word'].str.len() #new column for wordlen
    marks = ('.', ',', '!', '?', ';', ':')
    df['is_punctuation'] = df['word'].str.endswith(marks).astype(int) #checking for punctuation at the end
    counts = df['word'].value_counts() #storing word counts
    df['word_freq'] = df['word'].map(counts.to_dict())
    df['word_log_freq'] = np.log1p(df['word_freq'])
    df['syllable_count'] = df['word'].str.count(r'[aeiouăîâAEIOUĂÎÂ]') #more syllables = longer to read/prnounce
    df['word_index'] = df['word_id'].str.split('_').str[-1].fillna(0).astype(int) #checking the word index on the page
    df['is_proper_noun'] = ((df['word'].str.istitle()) & (df['word_index'] > 0)).astype(int)
    return df;

if __name__ == "__main__":
    train_df, test_df = loadingDataFromCSV()
    train_df = features(train_df)
    test_df = features(test_df)

    print(test_df.head(10))

    features_list = ['word_len', 'is_punctuation', 'word_freq', 'word_log_freq', 'syllable_count', 'word_index', 'is_proper_noun']
    X = train_df[features_list]
    y = train_df['answer']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

    model.fit(X_train, y_train)

    print("Training completed!")

    X_test_final = test_df[features_list]
    real_test_predictions = model.predict(X_test_final)

    real_test_predictions = np.maximum(0, real_test_predictions)

    submission = pd.DataFrame({
        'subtaskID': 1,
        'datapointID': test_df['datapointID'],
        'answer': real_test_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("'submission.csv' is ready to upload.")








