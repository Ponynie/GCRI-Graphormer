import pytorch_lightning as pl
import torch
from transformers import GraphormerForGraphClassification

class GraphormerLightningModule(pl.LightningModule):
    def __init__(self, config, learning_rate=1e-3, pretrain=False, model_name=None, pretrain_num_classes=1):
        """
        Args:
            config: Graphormer configuration.
            learning_rate: Learning rate for the optimizer.
            pretrain: Boolean, whether to use a pretrained model.
            model_name: Pretrained model name (required if pretrain=True).
            num_classes: Number of output classes for the classification task.
        """
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
        loss = outputs.loss  # Graphormer automatically computes loss if "labels" are in batch
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log("test_loss", loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)