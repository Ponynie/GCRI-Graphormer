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
    
class GraphormerLightningModuleScaled(pl.LightningModule):
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
        # We'll save these during setup
        self.max_ri = None
        self.min_ri = None
        self.mae_criterion = torch.nn.L1Loss()

    def forward(self, batch):
        return self.model(**batch)

    def setup(self, stage=None):
        """Called on every GPU"""
        # Get scaling factors from the dataset
        if isinstance(self.trainer.datamodule.pytorch_dataset, GraphDatasetScaled):
            self.max_ri = self.trainer.datamodule.pytorch_dataset.max_ri
            self.min_ri = self.trainer.datamodule.pytorch_dataset.min_ri

    def unscale_predictions(self, scaled_values):
        """Convert scaled predictions back to original range"""
        return scaled_values * (self.max_ri - self.min_ri) + self.min_ri

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        
        # Calculate scaled MAE (0-1 range)
        scaled_mae = self.mae_criterion(outputs.logits.squeeze(), batch['labels'].float())
        
        # Calculate unscaled MAE (original range)
        predictions_unscaled = self.unscale_predictions(outputs.logits.squeeze())
        labels_unscaled = self.unscale_predictions(batch['labels'].float())
        unscaled_mae = self.mae_criterion(predictions_unscaled, labels_unscaled)
        
        # Log both metrics
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_mae_scaled", scaled_mae, prog_bar=True, logger=True)
        self.log("train_mae", unscaled_mae, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        
        # Calculate scaled MAE (0-1 range)
        scaled_mae = self.mae_criterion(outputs.logits.squeeze(), batch['labels'].float())
        
        # Calculate unscaled MAE (original range)
        predictions_unscaled = self.unscale_predictions(outputs.logits.squeeze())
        labels_unscaled = self.unscale_predictions(batch['labels'].float())
        unscaled_mae = self.mae_criterion(predictions_unscaled, labels_unscaled)
        
        # Log both metrics
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_mae_scaled", scaled_mae, prog_bar=True, logger=True)
        self.log("val_mae", unscaled_mae, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        
        # Calculate scaled MAE (0-1 range)
        scaled_mae = self.mae_criterion(outputs.logits.squeeze(), batch['labels'].float())
        
        # Calculate unscaled MAE (original range)
        predictions_unscaled = self.unscale_predictions(outputs.logits.squeeze())
        labels_unscaled = self.unscale_predictions(batch['labels'].float())
        unscaled_mae = self.mae_criterion(predictions_unscaled, labels_unscaled)
        
        # Log both metrics
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_mae_scaled", scaled_mae, prog_bar=True, logger=True)
        self.log("test_mae", unscaled_mae, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)