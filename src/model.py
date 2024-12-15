import torch
import pytorch_lightning as pl
from transformers import GraphormerConfig, GraphormerForGraphClassification

# Lightning Module for Regression
class GraphormerRegressionModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = GraphormerForGraphClassification(config)
        self.criterion = torch.nn.MSELoss()  # Loss function for regression

    def forward(self, input_nodes, input_edges, attn_bias, in_degree, out_degree, spatial_pos):
        return self.model(
            input_nodes=input_nodes,
            input_edges=input_edges,
            attn_bias=attn_bias,
            in_degree=in_degree,
            out_degree=out_degree,
            spatial_pos=spatial_pos
        )

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs, labels=labels)
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(**inputs, labels=labels)
        loss = outputs.loss
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer