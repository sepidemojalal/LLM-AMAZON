import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Load dataset
data = pd.read_csv('data/reviews.csv')

# Clean the data (select the 'Text' and 'Score' columns)
data = data[['Text', 'Score']]

# Filter for positive (4 or 5) and negative (1 or 2) reviews
data = data[data['Score'] != 3]
data['label'] = data['Score'].apply(lambda x: 1 if x > 3 else 0)

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data.to_csv('data/train.csv', index=False)
test_data.to_csv('data/test.csv', index=False)