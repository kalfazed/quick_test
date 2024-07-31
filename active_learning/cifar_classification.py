import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.metrics import accuracy_score


# Define the CNN model for cifar10 classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, criterion, optimizer, dataloader, epochs=1):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 1000:.3f}')
                running_loss = 0.0

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Function to get the uncertainty of the model's predictions
def get_uncertainty(model, dataloader):
    model.eval()
    uncertainties = []
    with torch.no_grad():
        for data in dataloader:
            images, _ = data
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            uncertainty = 1 - probs.max(dim=1)[0]
            uncertainties.extend(uncertainty.numpy())
    return np.array(uncertainties)



def main():
    # Define cifar dataset normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Define cifar dataset and data loaders
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)
    
    # Define a model, loss function, and optimizer to train cifar
    net = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Define the initial training dataset. Randomly pick 100 samples from dataset
    n_initial = 10
    initial_idx = np.random.choice(range(len(train_set)), size=n_initial, replace=False)
    initial_loader = DataLoader(Subset(train_set, initial_idx), batch_size=4, shuffle=True)

    # Train the initial model
    print("Training without active learning")
    train_model(net, criterion, optimizer, initial_loader, epochs=2)

    # Evaluate the initial model
    initial_accuracy = evaluate_model(net, test_loader)
    print(f'Initial accuracy: {initial_accuracy:.2f}')

    
    # Active learning loop
    n_queries = 10
    query_size = 10

    for i in range(n_queries):
        # Get the model's uncertainty on the training data
        train_loader = DataLoader(train_set, batch_size=4, shuffle=False)
        uncertainties = get_uncertainty(net, train_loader)
        
        # Select the most uncertain samples
        query_idx = np.argsort(-uncertainties)[:query_size]
        query_loader = DataLoader(Subset(train_set, query_idx), batch_size=4, shuffle=True)
        
        # Teach the model with the queried samples
        print("Training with active learning")
        train_model(net, criterion, optimizer, query_loader, epochs=12)
        
        # Remove queried samples from the training set
        train_set = Subset(train_set, [i for j, i in enumerate(range(len(train_set))) if j not in query_idx])

        # Evaluate the model
        accuracy = evaluate_model(net, test_loader)
        print(f'Accuracy after query {i + 1}: {accuracy:.2f}')

if __name__ == '__main__':
    main()
