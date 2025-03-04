import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tests.transformer.transformer import Transformer
from tests.tiny_stories.preprocessing import get_data
from torch.utils.data import DataLoader
import torch
from frEase.recipes import ProgressiveRecipes
from frEase.trainer import ProgressiveTrainer
from torch import optim, nn
from tests.display_res import benchmark_display

seq_length = 5
vocab_size = 10000
train_batch_count = 2000
test_batch_count = 2000
sq_dim = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# On entraine notre transformer
model = Transformer(
    d_model=32,
    n_heads=2,
    d_ff=32,
    depth=4,
    vocab_size=vocab_size,
    max_context_size=seq_length,
).to(device)

train_dataset, validation_dataset, vocab = get_data(
    seq_length, vocab_size, train_batch_count, test_batch_count
)

train_dataloader = DataLoader(train_dataset, batch_size=16)
validation_dataloader = DataLoader(validation_dataset, batch_size=16)

# Constructive + Alternative training
frease_recipe = ProgressiveRecipes(model)
frease_recipe.base_recipe(
    epochs=40, iterations=10, global_trainning=0, scaling_factor=1
)
trainer = ProgressiveTrainer(frease_recipe)
trainer.train(
    train_dataloader,
    optim.AdamW,
    nn.CrossEntropyLoss(),
    validation_dataloader,
    results_saving_name="./results/frease.pkl",
)

# Traditional training
traditionnal_recipe = ProgressiveRecipes(model)
traditionnal_recipe.base_recipe(epochs=40, global_trainning=11, constructive=False)
trainer = ProgressiveTrainer(traditionnal_recipe)
trainer.train(
    train_dataloader,
    optim.AdamW,
    nn.CrossEntropyLoss(),
    validation_dataloader,
    results_saving_name="./results/traditionnal.pkl",
)

benchmark_display("./results/frease.pkl", "./results/traditionnal.pkl")
