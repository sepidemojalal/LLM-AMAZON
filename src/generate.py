import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_sentiment(review):
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return "positive" if prediction == 1 else "negative"

# Example usage
review = "This product is great! I absolutely loved it."
result = predict_sentiment(review)
print(f"Review: {review}")
print(f"Prediction: {result}")