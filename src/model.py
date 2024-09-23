import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AdamW
from transformers import BertTokenizer
import pandas as pd

class ReviewDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = self.data.iloc[idx]['Text']
        label = self.data.iloc[idx]['label']
        inputs = self.tokenizer(review, padding='max_length', truncation=True, return_tensors="pt")
        return inputs['input_ids'][0], inputs['attention_mask'][0], torch.tensor(label)

# Load train data
train_data = pd.read_csv('data/train.csv')

# Initialize tokenizer and dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = ReviewDataset(train_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
for epoch in range(3):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print("Training complete!")
