from transformers import GraphormerConfig
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datamodule import GraphDataset
from model import GraphormerRegressionModule

# Example data and labels
dummy_data = [
    {
        "input_nodes": torch.randint(0, 4608, (512,)),
        "input_edges": torch.randint(0, 1536, (512, 512)),
        "attn_bias": torch.rand(512, 512),
        "in_degree": torch.randint(0, 512, (512,)),
        "out_degree": torch.randint(0, 512, (512,)),
        "spatial_pos": torch.randint(0, 1024, (512, 512)),
    }
    for _ in range(100)
]
dummy_labels = torch.rand(100)  # Regression targets

# Create Dataset and Dataloader
train_dataset = GraphDataset(dummy_data, dummy_labels)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Configuration for Graphormer
config = GraphormerConfig(
    num_classes=1,  # Regression task
    num_atoms=4608,
    num_edges=1536,
    num_in_degree=512,
    num_out_degree=512,
    num_attention_heads=32,
    num_hidden_layers=12,
    max_nodes=512,
    embedding_dim=768,
    ffn_embedding_dim=768,
)

# Initialize the Lightning Module
model = GraphormerRegressionModule(config)

# PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=10, gpus=1 if torch.cuda.is_available() else 0)

# Train the model
trainer.fit(model, train_loader)