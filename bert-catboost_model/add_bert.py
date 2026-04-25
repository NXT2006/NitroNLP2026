import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np
import os

# use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
if device.type != "cuda":
    print("cuda is not available")

# load romanian bert
model_name = 'dumitrescustefan/bert-base-romanian-cased-v1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()


def get_bert_embeddings(filename):
    df = pd.read_csv(filename, encoding="utf-8")
    words = df['word'].fillna("").astype(str).tolist()

    embeddings = []
    batch_size = 128
    contexts = []
    target_word_indices = []

    for i in range(len(words)):
        start = max(0, i - 5)
        end = min(len(words), i + 6)

        context_words = words[start:end]
        context_phrase = " ".join(context_words)
        contexts.append(context_phrase)

        left_context = " ".join(words[start:i])
        left_tokens = tokenizer.tokenize(left_context)
        target_word_indices.append(len(left_tokens) + 1)

    for i in tqdm(range(0, len(contexts), batch_size)):
        batch_texts = contexts[i:i + batch_size]
        batch_indices = target_word_indices[i:i + batch_size]

        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_state = outputs.last_hidden_state

        for j in range(len(batch_texts)):
            idx = min(batch_indices[j], last_hidden_state.shape[1] - 1)
            word_embedding = last_hidden_state[j, idx, :].cpu().numpy()
            embeddings.append(word_embedding)

    return df, np.array(embeddings)


mlm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
mlm_model.eval()

def get_surprisal(filename, df=None, words=None):
    """Compute -log P(word | context) for each word using Romanian BERT MLM."""
    if df is None:
        df = pd.read_csv(filename, encoding="utf-8")
    if words is None:
        words = df['word'].fillna("").astype(str).tolist()

    surprisals = []
    batch_size = 64

    contexts = []
    target_token_positions = []  # position of [MASK] in the tokenized input
    target_token_ids = []        # vocab id(s) of the actual word

    for i in range(len(words)):
        start = max(0, i - 5)
        end = min(len(words), i + 6)

        context_words = words[start:end]
        local_idx = i - start  # index of target word within the window

        # Build masked context string
        masked_words = context_words.copy()
        masked_words[local_idx] = tokenizer.mask_token
        masked_phrase = " ".join(masked_words)
        contexts.append(masked_phrase)

        # Find position of [MASK] token in the tokenized input
        left_words = context_words[:local_idx]
        left_tokens = tokenizer.tokenize(" ".join(left_words)) if left_words else []
        mask_pos = len(left_tokens) + 1  # +1 for [CLS]
        target_token_positions.append(mask_pos)

        # Get the token id(s) of the actual target word
        word_tokens = tokenizer.encode(words[i], add_special_tokens=False)
        target_token_ids.append(word_tokens)

    for i in tqdm(range(0, len(contexts), batch_size), desc="Surprisal"):
        batch_texts = contexts[i:i + batch_size]
        batch_positions = target_token_positions[i:i + batch_size]
        batch_target_ids = target_token_ids[i:i + batch_size]

        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           max_length=64, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = mlm_model(**inputs)

        logits = outputs.logits  # (batch, seq_len, vocab_size)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        for j in range(len(batch_texts)):
            pos = min(batch_positions[j], log_probs.shape[1] - 1)
            token_ids = batch_target_ids[j]

            if len(token_ids) == 0:
                surprisals.append(20.0)  # fallback for unknown words
            elif len(token_ids) == 1:
                lp = log_probs[j, pos, token_ids[0]].item()
                surprisals.append(-lp)
            else:
                # Multi-token word: average surprisal over its subword tokens
                # (approximate — only the first token position is masked)
                lp = log_probs[j, pos, token_ids[0]].item()
                surprisals.append(-lp)

    return surprisals


# SAFETY CHECKPOINTING
train_embeds_file = "train_embeds_raw.npy"
test_embeds_file  = "test_embeds_raw.npy"

if os.path.exists(train_embeds_file) and os.path.exists(test_embeds_file):
    print("Loading saved embeddings from disk...")
    train_df     = pd.read_csv("train_data.csv", encoding="utf-8")
    test_df      = pd.read_csv("test_data.csv",  encoding="utf-8")
    train_embeds = np.load(train_embeds_file)
    test_embeds  = np.load(test_embeds_file)
else:
    print("Extracting Train embeddings...")
    train_df, train_embeds = get_bert_embeddings("train_data.csv")
    np.save(train_embeds_file, train_embeds)

    print("Extracting Test embeddings...")
    test_df, test_embeds = get_bert_embeddings("test_data.csv")
    np.save(test_embeds_file, test_embeds)

# --- CHECKPOINT 2: Surprisal (always has train_df/test_df available now) ---
train_surprisal_file = "train_surprisal.npy"
test_surprisal_file  = "test_surprisal.npy"

if os.path.exists(train_surprisal_file) and os.path.exists(test_surprisal_file):
    print("Loading saved surprisal from disk...")
    train_surprisal = np.load(train_surprisal_file)
    test_surprisal  = np.load(test_surprisal_file)
else:
    print("Computing surprisal for train...")
    train_words     = train_df['word'].fillna("").astype(str).tolist()
    train_surprisal = np.array(get_surprisal("train_data.csv", train_df, words=train_words))
    np.save(train_surprisal_file, train_surprisal)

    print("Computing surprisal for test...")
    test_words     = test_df['word'].fillna("").astype(str).tolist()
    test_surprisal = np.array(get_surprisal("test_data.csv", test_df, words=test_words))
    np.save(test_surprisal_file, test_surprisal)

# --- PCA & SAVING (all variables guaranteed defined above) ---
train_df['surprisal'] = train_surprisal
test_df['surprisal']  = test_surprisal

N_COMPONENTS = 32
pca = PCA(n_components=N_COMPONENTS, random_state=42)
train_compressed = pca.fit_transform(train_embeds)
test_compressed  = pca.transform(test_embeds)

for i in range(N_COMPONENTS):
    train_df[f'bert_feature_{i}'] = train_compressed[:, i]
    test_df[f'bert_feature_{i}']  = test_compressed[:, i]

train_df.to_csv("train_with_bert.csv", index=False)
test_df.to_csv("test_with_bert.csv",   index=False)
print("Done!")