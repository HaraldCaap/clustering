import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import fcluster
import pandas as pd
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess_sentence(sentence):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(sentence.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Example sentences
sentences = [
    "This is the first sentence.",
    "This is the second sentence.",
    "And here is another sentence.",
    "Yet another different sentence here.",
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaps over a sleepy dog.",
    "A lazy dog is jumped over by a quick brown fox."
]

# Preprocess sentences
cleaned_sentences = [preprocess_sentence(sentence) for sentence in sentences]

# Convert sentences to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_sentences).toarray()

# Define the autoencoder model in PyTorch
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

input_dim = X.shape[1]
encoding_dim = 2  # Reduced dimensionality

model = Autoencoder(input_dim, encoding_dim)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)

# Train the autoencoder
num_epochs = 100
batch_size = 4
dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for data in dataloader:
        inputs, _ = data
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Encode the sentences to the lower-dimensional space
with torch.no_grad():
    X_encoded = model.encoder(X_tensor).numpy()

# Perform hierarchical clustering
linkage_matrix = sch.linkage(X_encoded, method='ward')

# Define the levels (distance thresholds) for clustering
levels = [0.1, 0.3, 0.5, 1.0, 1.5, 2]  # Adjust these based on the dendrogram

# Generate clusters at different levels
cluster_data = {}
for level in levels:
    clusters = fcluster(linkage_matrix, level, criterion='distance')
    cluster_data[f'Level {level}'] = clusters

# Create a DataFrame to show the clusters at different levels
df = pd.DataFrame(cluster_data, index=sentences)
df.index.name = 'Sentence'

print(df)

# Plot the dendrogram to visualize the clustering
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(linkage_matrix, labels=sentences, leaf_rotation=90, leaf_font_size=10)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sentences')
plt.ylabel('Distance')
plt.show()
