import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Dummy data generation with 1000 samples and 100 features (species)
np.random.seed(42)
data = np.random.rand(1000, 100).astype(np.float32)
species_names = [f'Species_{i}' for i in range(100)]
key_species = [0, 5, 10]  # Indexes of species to be preserved

# Function to perform hierarchical clustering while preserving key species
def preserve_key_species(data, key_species, method='ward', threshold=1.5):
    # Extract key species data to preserve them
    key_data = data[:, key_species]

    # Remove key species from the data used for clustering
    remaining_data = np.delete(data, key_species, axis=1)

    # Perform hierarchical clustering on the remaining data
    linked = linkage(remaining_data, method=method)

    # Plot the dendrogram to visualize the clustering
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.show()

    # Cut the dendrogram at the specified threshold to form clusters
    clusters = fcluster(linked, threshold, criterion='distance')

    # Combine key species data back with the clustered data
    combined_data = np.hstack((key_data, remaining_data))

    return combined_data, clusters

# Set the threshold for cutting the dendrogram
threshold = 1.5
# Perform clustering and get the combined data and clusters
combined_data, clusters = preserve_key_species(data, key_species, threshold=threshold)

# Map cluster assignments to species names
cluster_mapping = {}
for idx, cluster_id in enumerate(clusters):
    if cluster_id not in cluster_mapping:
        cluster_mapping[cluster_id] = []
    cluster_mapping[cluster_id].append(species_names[idx])

# Print the cluster assignments to see which species are grouped together
print("Cluster assignments:")
for cluster_id, species in cluster_mapping.items():
    print(f"Cluster {cluster_id}: {species}")

# Define the custom autoencoder class
class CustomAutoencoder(nn.Module):
    def __init__(self, input_dim, key_indexes):
        super(CustomAutoencoder, self).__init__()
        self.key_indexes = key_indexes
        self.input_dim = input_dim - len(key_indexes)

        # Define the encoder network
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Define the decoder network
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Separate key species and remaining data
        key_data = x[:, self.key_indexes]
        remaining_data = torch.cat([x[:, :i] for i in range(x.shape[1]) if i not in self.key_indexes], dim=1)

        # Encode and decode the remaining data
        encoded = self.encoder(remaining_data)
        decoded = self.decoder(encoded)

        # Reconstruct the data with key species intact
        reconstructed = torch.zeros_like(x)
        reconstructed[:, self.key_indexes] = key_data
        reconstructed[:, [i for i in range(x.shape[1]) if i not in self.key_indexes]] = decoded

        return reconstructed

# Set the input dimension for the autoencoder
input_dim = combined_data.shape[1]
# Initialize the autoencoder model
model = CustomAutoencoder(input_dim, key_species)

# Convert the combined data to a PyTorch tensor for training
dataset = torch.tensor(combined_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop for the autoencoder
num_epochs = 20
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        # Forward pass through the model
        outputs = model(batch)
        loss = criterion(outputs, batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Function to analyze the encoded representation
def analyze_encoded_representation(model, data, key_species):
    model.eval()
    with torch.no_grad():
        key_data = torch.tensor(data[:, key_species])
        remaining_data = torch.tensor(np.delete(data, key_species, axis=1))
        encoded = model.encoder(remaining_data)
    return encoded

# Analyze the encoded representation of the data
encoded_representation = analyze_encoded_representation(model, combined_data, key_species)

# Calculate importance scores for the features in the encoded representation
importance_scores = torch.mean(torch.abs(encoded_representation), dim=0).numpy()
important_features = np.argsort(importance_scores)[-len(key_species):][::-1]

# Print important features in the encoded representation
print(f"Important features in the encoded representation: {important_features}")

# Compare the original input data with the reconstructed output
model.eval()
with torch.no_grad():
    reconstructed_data = model(torch.tensor(combined_data)).numpy()

# Map original species to the reduced representation
reduced_species_mapping = {}
for i, idx in enumerate(key_species):
    reduced_species_mapping[species_names[idx]] = [species_names[idx]]
for i in important_features:
    species_list = [species_names[j] for j in np.where(clusters == (i + 1))[0]]
    reduced_species_mapping[f'Reduced_Species_{i}'] = species_list

# Print the mapping of reduced species to original species
print("Reduced species mapping:")
for key, value in reduced_species_mapping.items():
    print(f"{key}: {value}")
