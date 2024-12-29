import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import Dataset as HFDataset
from transformers.models.graphormer.collating_graphormer import GraphormerDataCollator, preprocess_item
import pytorch_lightning as pl
import numpy as np
from torch_geometric.utils import from_smiles

class GraphDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        """
        Args:
            dataframe (pd.DataFrame): The DataFrame containing the dataset.
        """
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        smiles = row["smiles"]
        try:
            graph_data = from_smiles(smiles) # Convert SMILES to PyG Data object
            if graph_data.num_edges == 0: # Handle molecules with no edges
                print(f"Skipping molecule with no edges: {smiles}")
                return None

            return {
                "edge_index": graph_data["edge_index"],
                "num_nodes": graph_data.num_nodes,
                "labels": np.array([float(row["ri"])], dtype=np.float32),  # Retention Index to predict
                "node_feat": graph_data["x"], # Node features
                "edge_attr": graph_data["edge_attr"], # Edge features
            }
        except Exception as e:
            print(f"Error processing SMILES: {smiles}, Error: {e}")
            return None
class GraphormerDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, batch_size=32, train_split=0.8, val_split=0.1, test_split=0.1, num_workers=1):
        """
        Args:
            csv_path: Path to the CSV file containing the dataset
            batch_size: Batch size for training, validation, and test DataLoaders
            train_split: Proportion of data used for training
            val_split: Proportion of data used for validation
            test_split: Proportion of data used for testing
        """
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.data_collator = GraphormerDataCollator()

    def prepare_data(self):
        """
        Read the CSV file and initialize the PyTorch dataset.
        """
        # Step 1: Read CSV file
        self.data_df = pd.read_csv(self.csv_path)
        self.pytorch_dataset = GraphDataset(self.data_df)

    def setup(self, stage=None):
        """
        Convert PyTorch dataset to Hugging Face dataset, preprocess it, and split it.
        """
        # Step 2: Convert PyTorch Dataset to Hugging Face Dataset
        def pytorch_to_hf_dataset(pytorch_dataset):
            samples = [pytorch_dataset[i] for i in range(len(pytorch_dataset)) if pytorch_dataset[i] is not None]
            return HFDataset.from_list(samples)

        hf_dataset = pytorch_to_hf_dataset(self.pytorch_dataset)

        # Step 3: Preprocess the Hugging Face Dataset
        hf_dataset = hf_dataset.map(preprocess_item, batched=False)

        # Step 4: Split the Dataset
        dataset_length = len(hf_dataset)
        train_size = int(dataset_length * self.train_split)
        val_size = int(dataset_length * self.val_split)
        test_size = dataset_length - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            hf_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
        )