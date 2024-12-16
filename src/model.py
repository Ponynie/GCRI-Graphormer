import pytorch_lightning as pl
import torch
from transformers import GraphormerForGraphClassification

class GraphormerLightningModule(pl.LightningModule):
    def __init__(self, config, learning_rate=1e-3):
        super().__init__()
        self.model = GraphormerForGraphClassification(config)
        self.learning_rate = learning_rate

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss  # Graphormer automatically computes loss if "labels" are in batch
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)