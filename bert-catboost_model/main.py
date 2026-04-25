import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


# hackathon evaluation metric
def eval_metric(y_true, preds):
    y_true = y_true.astype(float)
    preds = preds.astype(float)
    r2 = r2_score(y_true, preds, sample_weight=None, force_finite=True)
    r2 = max(0, r2)
    pears = pearsonr(y_true, preds)[0]
    if np.isnan(pears):
        pears = 0.0
    pears = np.abs(pears)
    return 100*(pears + r2)/2


# load data and read train_data.csv
df = pd.read_csv('train_data.csv', encoding='utf-8')
print("Data loaded successfully")




# FEATURES


# word length
df['word_length'] = df['word'].astype(str).str.len()

# extract word position in text
df['word_index'] = df['word_id'].apply(lambda x: int(str(x).split('_')[-1]))

# punctuation
df['is_punctuation'] = df['word'].astype(str).str.match(r'^[\W_]+$').astype(int)



# TRAINING


features = ['word_length', 'word_index', 'is_punctuation', 'participant_id', 'text']
target = 'answer'

df = df.dropna(subset=target)

X = df[features]
y = df[target]

# training and validation split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# TRAINING MODEL
print("Starting training")
cat_features_indices = ['participant_id', 'text']

model = CatBoostRegressor(iterations=300, learning_rate=0.01, depth=6, cat_features=cat_features_indices, verbose=50)
model.fit(X_train, y_train, eval_set=(X_test, y_test))



# EVALUATE
predictions = model.predict(X_test)

score = eval_metric(y_test, predictions)
print(f"Score: {score:.2f}")



# Submission file creation
test_df = pd.read_csv("test_data.csv", encoding="utf-8")

test_df['word_length'] = test_df['word'].astype(str).str.len()
test_df['word_index'] = test_df['word_id'].apply(lambda x: int(str(x).split('_')[-1]))
test_df['is_punctuation'] = test_df['word'].astype(str).str.match(r'^[\W_]+$').astype(int)

# predictions
test_preds = model.predict(test_df[features])

# force negatives to zero
test_preds = np.maximum(0, test_preds)

# create dataframe in proper format
submission = pd.DataFrame({'subtaskID': 1, 'datapointID': test_df['datapointID'], 'answer': test_preds})

# save to csv
submission.to_csv("output.csv", index=False)
print("saved output")