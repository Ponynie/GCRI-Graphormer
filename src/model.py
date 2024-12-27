import pytorch_lightning as pl
import torch
from transformers import GraphormerForGraphClassification
from datamodule import GraphDataset, GraphDatasetScaled
import torch.nn.functional as F

class GraphormerLightningModule(pl.LightningModule):
    def __init__(self, config, learning_rate=1e-3, pretrain=False, model_name=None, pretrain_num_classes=1):
        super().__init__()
        if pretrain:
            if not model_name:
                raise ValueError("Model name must be provided when pretrain is True.")
            print(f"Loading pretrained model: {model_name}")
            self.model = GraphormerForGraphClassification.from_pretrained(
                model_name,
                num_classes=pretrain_num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            self.model = GraphormerForGraphClassification(config)
        
        self.learning_rate = learning_rate

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        mae = F.l1_loss(outputs.logits.squeeze(), batch['labels'].float())
        
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_mae", mae, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        mae = F.l1_loss(outputs.logits.squeeze(), batch['labels'].float())
        
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_mae", mae, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        mae = F.l1_loss(outputs.logits.squeeze(), batch['labels'].float())
        
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_mae", mae, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    