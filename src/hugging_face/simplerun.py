import pandas as pd
from torch_geometric.utils import from_smiles
from datasets import Dataset as HFDataset
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

# Step 1: Read the CSV file
csv_path = "data/NP-LRI-RAMP-G-C.csv"
data_df = pd.read_csv(csv_path)

# Step 2: Create a PyTorch Dataset
class GraphDataset(Dataset):
    def __init__(self, dataframe):
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
            graph_data = from_smiles(smiles)  # Convert SMILES to PyG Data object
            if graph_data.num_edges == 0:  # Skip molecules with no edges
                print(f"Skipping molecule with no edges: {smiles}")
                return None

            return {
                "edge_index": graph_data["edge_index"],
                "num_nodes": graph_data.num_nodes,
                "labels": np.array([float(row["ri"])], dtype=np.float32),  # Retention Index to predict
                "node_feat": graph_data["x"],  # Node features
                "edge_attr": graph_data["edge_attr"],  # Edge features
            }
        except Exception as e:
            print(f"Error processing SMILES: {smiles}, Error: {e}")
            return None  # Handle invalid SMILES gracefully

# Step 3: Instantiate the PyTorch Dataset
pytorch_dataset = GraphDataset(data_df)

# Step 4: Convert PyTorch Dataset to Hugging Face Dataset
def pytorch_to_hf_dataset(pytorch_dataset):
    # Convert each sample to a dictionary compatible with Hugging Face Dataset
    samples = [pytorch_dataset[i] for i in range(len(pytorch_dataset)) if pytorch_dataset[i] is not None]
    return HFDataset.from_list(samples)

hf_dataset = pytorch_to_hf_dataset(pytorch_dataset)

# Step 5: Preprocess the Hugging Face Dataset
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator

# Apply preprocessing
dataset_processed = hf_dataset.map(preprocess_item, batched=False)

#split the Dataset
dataset_length = len(dataset_processed)
train_size = int(dataset_length * 0.8)
val_size = int(dataset_length * 0.1)
test_size = dataset_length - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset_processed, [train_size, val_size, test_size]
)

from transformers import GraphormerConfig, GraphormerForGraphClassification

# # Define a new configuration
# config = GraphormerConfig(
#     num_classes=1,  # Number of output classes for the downstream task
#     num_hidden_layers=2,  # You can adjust the model architecture as needed
#     hidden_size=768,
#     num_attention_heads=1,
#     intermediate_size=768,
# )

# # Initialize a Graphormer model with the new configuration
# model = GraphormerForGraphClassification(config)

model = GraphormerForGraphClassification.from_pretrained(
    "clefourrier/pcqm4mv2_graphormer_base",
    num_classes=1, # num_classes for the downstream task 
    ignore_mismatched_sizes=True,
)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    "HuggingFaceTrainer", 
    logging_dir="huggingface_logs",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    auto_find_batch_size=True, # batch size can be changed automatically to prevent OOMs
    dataloader_num_workers=8, 
    num_train_epochs=20,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=GraphormerDataCollator(),
)

train_results = trainer.train()
#rainer.push_to_hub()

