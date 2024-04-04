import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from logger_config import configure_logger
import os
import sys

# Initialize logger
logger = configure_logger(__name__)

# Load data
df = pd.read_csv('data/labeled_data.csv')

# Initialize RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

# Tokenize and encode text data
tokenized_texts = [tokenizer.encode(tweet, add_special_tokens=True) for tweet in df['tweet']]

# Pad sequences to a fixed length
max_len = max(len(ids) for ids in tokenized_texts)
input_ids = [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids in tokenized_texts]

# Convert to PyTorch tensors
input_ids = torch.tensor(input_ids)
labels = torch.tensor(df['class'])

# Split data into train and validation sets
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)

# Fine-tune the RoBERTa model
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
epochs = 3
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(train_inputs, labels=train_labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(val_inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    accuracy = accuracy_score(val_labels.tolist(), predictions.tolist())
    logger.info("Validation Accuracy: %f", accuracy)

# Save the fine-tuned model
if model:
    model_dir = 'models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'fine_tuned_roberta_model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved fine-tuned RoBERTa model at: {model_path}")