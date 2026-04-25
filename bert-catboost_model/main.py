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
    r2 = max(0, r2_score(y_true, preds, force_finite=True))
    pears = pearsonr(y_true, preds)[0]
    pears = 0.0 if np.isnan(pears) else abs(pears)
    return 100 * (pears + r2) / 2


# load data
df = pd.read_csv('train_with_bert.csv', encoding='utf-8')
test_df = pd.read_csv("test_with_bert.csv", encoding="utf-8")
print("data loaded")

df = df.dropna(subset=['answer'])

# 1. FIX: Combine words BEFORE counting to prevent test-set NaNs
all_words = pd.concat([df['word'].astype(str), test_df['word'].astype(str)])
word_counts = all_words.value_counts().to_dict()

# 2. Gold Features (Calculate ONLY on train to prevent leakage)
word_means = df.groupby("word")["answer"].mean().to_dict()
word_skips = df.groupby("word")["answer"].apply(lambda x: (x == 0).mean()).to_dict()
global_mean_trt = df["answer"].mean()
global_skip_rate = (df["answer"] == 0).mean()


# 3. Apply features to BOTH dataframes cleanly
def apply_features(data):
    data['word_length'] = data['word'].str.replace(r'[^\w\s]', '', regex=True).str.len()
    data['word_index'] = data['word_id'].apply(lambda x: int(str(x).split('_')[-1]))
    data['is_punctuation'] = data['word'].astype(str).str.match(r'^[\W_]+$').astype(int)

    data['word_freq'] = data['word'].map(word_counts).fillna(1)
    data['word_log_freq'] = np.log1p(data['word_freq'])

    data['word_mean_trt'] = data['word'].map(word_means).fillna(global_mean_trt)
    data['word_skip_rate'] = data['word'].map(word_skips).fillna(global_skip_rate)
    return data


df = apply_features(df)
test_df = apply_features(test_df)

# TRAINING SETUP
N_COMPONENTS = 128  # Must match the change in add_bert.py!
bert_features = [f'bert_feature_{i}' for i in range(N_COMPONENTS)]

features = [
               'word_length',
               'word_index',
               'is_punctuation',
               'participant_id',
               'text',
               'word_log_freq',
               'word_mean_trt',
               'word_skip_rate',
               'surprisal',  # Crucial NLP metric added from your extraction script
           ] + bert_features

X = df[features]

# 4. FIX: Log-transform the target to handle massive outliers
y = np.log1p(df['answer'])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting training")
model = CatBoostRegressor(
    iterations=2000,  # More trees
    learning_rate=0.03,  # Slower learning for stability
    depth=6,
    cat_features=['participant_id', 'text'],
    early_stopping_rounds=100,  # Waits longer before giving up
    verbose=100
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))

# 5. FIX: Reverse the log-transform when predicting
val_preds = np.maximum(0, np.expm1(model.predict(X_test)))
y_test_true = np.expm1(y_test)

score = eval_metric(y_test_true, val_preds)
print(f"Validation Score: {score:.2f} / 100")

# SUBMISSION
test_preds = np.maximum(0, np.expm1(model.predict(test_df[features])))

submission = pd.DataFrame({
    'subtaskID': 1,
    'datapointID': test_df['datapointID'],
    'answer': test_preds
})
submission.to_csv("output.csv", index=False)
print("saved output!")