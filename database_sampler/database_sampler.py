import torch
from torch.utils.data import Dataset, DataLoader
import random

class DatabaseSampler(Dataset):
    def __init__(self, database, transform=None):
        self.database = database
        self.transform = transform
        self.data = self.load_data_from_db()

    def load_data_from_db(self):
        # Placeholder for actual database loading logic
        return [self.database.get_sample(i) for i in range(len(self.database))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# Example usage
class FakeDatabase:
    def __init__(self, num_samples):
        self.samples = [(i, i*2) for i in range(num_samples)]

    def get_sample(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

# Simulating a database with 100 samples
fake_db = FakeDatabase(100)

# Creating a database sampler instance
sampler = DatabaseSampler(fake_db)

# Creating a DataLoader
dataloader = DataLoader(sampler, batch_size=10, shuffle=True)

for batch in dataloader:
    print(batch)

