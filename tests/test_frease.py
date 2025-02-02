from frEase.trainer import ProgressiveTrainer
from frEase.recipes import ProgressiveRecipes
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch

class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.ml = nn.ModuleList([
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        ])
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
    def forward(self, x):
        x = self.linear1(x)
        for module in self.ml:
            x = module(x)
        x = self.linear2(x)
        return x

class xyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

epochs = 10
lr = 0.001
group_size = 1
global_trainning = 2
scaling_factor = 2
batch_size = 99
mini_batch_size = 16
dim = 10

model = Model(dim)

X = torch.zeros((batch_size, mini_batch_size, dim))
y = torch.zeros((batch_size, mini_batch_size, dim))
dataset = xyDataset(X, y)
data_loader = DataLoader(dataset, batch_size=batch_size)
optimizer = optim.AdamW
optimizer(model.parameters(), lr)
criterion = nn.MSELoss()

training_recipe = ProgressiveRecipes(model)
training_recipe.progressive_simple(epochs, lr, group_size, global_trainning, scaling_factor)
trainer = ProgressiveTrainer(training_recipe)
trainer.train(data_loader, optimizer, criterion)