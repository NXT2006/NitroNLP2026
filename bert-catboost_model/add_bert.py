import pandas as pd
import torch
from pygments.lexer import words
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.decomposition import PCA
import numpy as np



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


def process_file(filename, output_filename):
    df = pd.read_csv(filename, encoding="utf-8")

    #fill missing word with empty string to avoid crashing
    words = df['word'].fillna("").astype(str).tolist()

    bert_embeddings = []

    # process in batches of 128 to use gpu efficiently
    batch_size = 128

    contexts = []
    for i in range(len(words)):
        start = max(0, i - 2)
        end = min(len(words), i + 3)
        context_phrase = " ".join(words[start:end])
        contexts.append(context_phrase)

    for i in tqdm(range(0, len(contexts), batch_size)):
        batch_texts = contexts[i:i + batch_size]

        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        bert_embeddings.extend(cls_embeddings.cpu().numpy())

    bert_embeddings = np.array(bert_embeddings)

    pca = PCA(n_components=5, random_state=42)
    compressed_embeddings = pca.fit_transform(bert_embeddings)

    for i in range(5):
        df[f'bert_feature_{i}'] = compressed_embeddings[:, i]

    df.to_csv(output_filename, index=False)

process_file("train_data.csv", "train_with_bert.csv")
process_file("test_data.csv", "test_with_bert.csv")