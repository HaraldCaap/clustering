import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel


# === Data Preparation ===

# Sample data
data = [
    ("On the leading edge of the platform", ["leading edge", "platform"]),
    ("On the leading edge", ["leading edge"]),
    ("Cutting-edge platform innovation", ["leading edge", "platform"]),
    ("A beginner's guide to programming", ["beginner guide"]),
]

# Convert to a DataFrame for easier handling
df = pd.DataFrame(data, columns=["text", "labels"])

# Multi-label binarization: Convert labels into a binary format
mlb = MultiLabelBinarizer()
df["binary_labels"] = list(mlb.fit_transform(df["labels"]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["binary_labels"], test_size=0.2, random_state=42
)

# === Tokenization ===

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define a helper function to tokenize the text
def encode_texts(texts, tokenizer, max_length=128):
    """
    Tokenizes a list of texts using the BERT tokenizer.
    Args:
        texts: List of strings (sentences or documents).
        tokenizer: BERT tokenizer object.
        max_length: Maximum length for padding/truncation.

    Returns:
        A dictionary containing tokenized input IDs and attention masks.
    """
    return tokenizer(
        list(texts),  # Tokenize the list of texts
        max_length=max_length,  # Truncate to max_length
        padding="max_length",  # Pad to max_length
        truncation=True,  # Truncate longer sequences
        return_tensors="pt",  # Return PyTorch tensors
    )

# Tokenize the training and testing texts
train_encodings = encode_texts(X_train, tokenizer)
test_encodings = encode_texts(X_test, tokenizer)

# === Data Preparation ===

# Sample data
data = [
    ("On the leading edge of the platform", ["leading edge", "platform"]),
    ("On the leading edge", ["leading edge"]),
    ("Cutting-edge platform innovation", ["leading edge", "platform"]),
    ("A beginner's guide to programming", ["beginner guide"]),
]

# Convert to a DataFrame for easier handling
df = pd.DataFrame(data, columns=["text", "labels"])

# Multi-label binarization: Convert labels into a binary format
mlb = MultiLabelBinarizer()
df["binary_labels"] = list(mlb.fit_transform(df["labels"]))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["binary_labels"], test_size=0.2, random_state=42
)

# === Tokenization ===

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define a helper function to tokenize the text
def encode_texts(texts, tokenizer, max_length=128):
    """
    Tokenizes a list of texts using the BERT tokenizer.
    Args:
        texts: List of strings (sentences or documents).
        tokenizer: BERT tokenizer object.
        max_length: Maximum length for padding/truncation.

    Returns:
        A dictionary containing tokenized input IDs and attention masks.
    """
    return tokenizer(
        list(texts),  # Tokenize the list of texts
        max_length=max_length,  # Truncate to max_length
        padding="max_length",  # Pad to max_length
        truncation=True,  # Truncate longer sequences
        return_tensors="pt",  # Return PyTorch tensors
    )

# Tokenize the training and testing texts
train_encodings = encode_texts(X_train, tokenizer)
test_encodings = encode_texts(X_test, tokenizer)

# === Model Definition ===

# Define the BERT-based multi-label classification model
class BertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        """
        Initializes the BERT model for multi-label classification.
        Args:
            num_labels: Number of output labels (multi-label classification).
        """
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)  # Regularization to prevent overfitting
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.
        Args:
            input_ids: Tokenized input IDs.
            attention_mask: Attention masks for input.
            labels: Optional, used for training.

        Returns:
            Logits (raw predictions) for each label.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # BERT's [CLS] token representation
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return logits

# Initialize the model
num_labels = len(mlb.classes_)
model = BertForMultiLabelClassification(num_labels)


# === Training ===

# Prepare data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for multi-label

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        # Move inputs and labels to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).float()

        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")


# === Training ===

# Prepare data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy for multi-label

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        # Move inputs and labels to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device).float()

        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")


# === Prediction ===

def predict_with_probabilities(texts, model, tokenizer, max_length=128):
    """
    Predict labels and their probabilities for new texts.
    Args:
        texts: List of input texts.
        model: Trained BERT model.
        tokenizer: BERT tokenizer.
        max_length: Maximum sequence length for tokenization.

    Returns:
        A dictionary where keys are input texts, and values are tuples of:
        - Predicted labels
        - Probabilities for each label
    """
    model.eval()

    encodings = encode_texts(texts, tokenizer, max_length)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)  # Raw logits
        probabilities = torch.sigmoid(logits)  # Convert logits to probabilities

    results = []
    for i, text in enumerate(texts):
        probs = probabilities[i].cpu().numpy()  # Extract probabilities for this example
        predicted_labels = [mlb.classes_[j] for j, p in enumerate(probs) if p > 0.5]  # Threshold 0.5
        label_probs = {mlb.classes_[j]: p for j, p in enumerate(probs)}  # Map labels to probabilities
        results.append({"text": text, "predicted_labels": predicted_labels, "probabilities": label_probs})

    return results

# Predict new texts
new_texts = ["Leading edge"]

# Predict labels and probabilities
predictions = predict_with_probabilities(new_texts, model, tokenizer)

# Display results
for prediction in predictions:
    print(f"Text: {prediction['text']}")
    print(f"Predicted Labels: {prediction['predicted_labels']}")
    print("Probabilities:")
    for label, prob in prediction['probabilities'].items():
        print(f"  {label}: {prob:.4f}")

# === Dataset Definition ===

# Create a custom PyTorch dataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        """
        Initializes the dataset with encodings and labels.
        Args:
            encodings: Tokenized input (input IDs and attention masks).
            labels: Binary labels for multi-label classification.
        """
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the data point at the given index.
        Args:
            idx: Index of the data point.

        Returns:
            A dictionary containing input IDs, attention masks, and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx])
        return item

# Create PyTorch datasets
train_dataset = TextDataset(train_encodings, y_train)
test_dataset = TextDataset(test_encodings, y_test)
