from torch.utils.data import DataLoader, Dataset
import torch

class GraphDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # Preprocessed graph data (node features, edge features, etc.)
        self.labels = labels  # Target labels for regression

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        graph = self.data[idx]
        label = self.labels[idx]
        inputs = {
            "input_nodes": graph["input_nodes"],
            "input_edges": graph["input_edges"],
            "attn_bias": graph["attn_bias"],
            "in_degree": graph["in_degree"],
            "out_degree": graph["out_degree"],
            "spatial_pos": graph["spatial_pos"],
        }
        return inputs, torch.tensor(label, dtype=torch.float)