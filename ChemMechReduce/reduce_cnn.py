import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def example():
    input_dim = 100  # Adjust based on your data
    model = Autoencoder(input_dim)

    # Dummy data for training
    data = np.random.rand(1000, input_dim).astype(np.float32)
    dataset = torch.tensor(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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
