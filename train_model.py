# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split

# 🔗 Download the dataset from Kaggle:
# Fake.csv and True.csv are available in “Fake and Real News Dataset”
# https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset :contentReference[oaicite:1]{index=1}

# ✅ Load fake and real datasets
df_fake = pd.read_csv("data/Fake.csv")   # ~23.5K fake news articles
df_true = pd.read_csv("data/True.csv")   # ~21.4K real news articles :contentReference[oaicite:2]{index=2}

# 🔧 Add labels (0 = Fake, 1 = Real)
df_fake["label"] = 0
df_true["label"] = 1

# ✂️ Keep only relevant columns: title, text, label
df_fake = df_fake[["title", "text", "label"]]
df_true = df_true[["title", "text", "label"]]

# 📦 Concatenate into a single DataFrame
df = pd.concat([df_fake, df_true], ignore_index=True)

# 🔤 Combine title and text into one column
df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")

# 💡 Sanity-check
print(f"Total samples: {len(df)}, Fake: {df.label.sum(axis=0)==0}, Real: {len(df)-df.label.sum()}")

# 🎯 Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

print("Train/Val counts:", y_train.value_counts(), y_test.value_counts())



from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_ds = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
test_ds = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./model/best_model',
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_total_limit=1,
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./model/best_model")
