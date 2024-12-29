import pytorch_lightning as pl
import torch
from transformers import GraphormerForGraphClassification
from datamodule import GraphDataset
import torch.nn.functional as F

class GraphormerLightningModule(pl.LightningModule):
    def __init__(self, config, learning_rate=1e-3, lr_patience=10, lr_factor=0.1, weight_decay=1e-3, pretrain=False, model_name=None, pretrain_num_classes=1):
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
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.weight_decay = weight_decay

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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay  # Adding weight decay here
        )

        # Cyclic Learning Rate Scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.learning_rate * self.lr_factor,  # Lower bound of the cycle
                max_lr=self.learning_rate,  # Upper bound of the cycle
                step_size_up=1000,  # Number of iterations to reach max_lr
                mode='triangular',  # Triangular LR schedule
                cycle_momentum=False,  # Set to False for AdamW optimizer
            ),
            'interval': 'step',  # Change learning rate after every step
            'frequency': 1  # Apply this scheduler at every step
        }
        
        return [optimizer], [scheduler]