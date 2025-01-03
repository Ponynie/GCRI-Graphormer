from model import GraphormerLightningModule
from datamodule import GraphormerDataModule
from pytorch_lightning import Trainer
from transformers import GraphormerConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from hparam import Hyperparameters
import torch

# torch.set_float32_matmul_precision('medium')
# Path to the dataset CSV file
csv_path = "data/syntatics.csv"
check_mode = Hyperparameters.check_mode

# Initialize the data module
data_module = GraphormerDataModule(
    csv_path=csv_path,
    batch_size=Hyperparameters.batch_size,
    train_split=Hyperparameters.train_size,
    val_split=Hyperparameters.val_size,
    test_split=Hyperparameters.test_size,
    num_workers=Hyperparameters.num_workers,
)

# Prepare and set up the data
data_module.prepare_data()
data_module.setup()

# Set up callbacks and logger
lr_monitor = LearningRateMonitor(logging_interval='epoch')
wandb_logger = WandbLogger(project='Graphormer-Molecule', save_dir='wandb_log')
check_point = ModelCheckpoint(monitor='val_loss')
    
# Model configuration
config = GraphormerConfig(
    num_classes=1, 
    num_layers=Hyperparameters.num_layers,
    embedding_dim=Hyperparameters.embedding_dim,
    ffn_embedding_dim=Hyperparameters.ffn_embedding_dim,
    num_attention_heads=Hyperparameters.num_attention_heads,
    dropout=Hyperparameters.dropout
)

# Instantiate model and trainer
base_params = {
    "config": None if Hyperparameters.pretrain else config,
    "learning_rate": Hyperparameters.learning_rate,
    "lr_patience": Hyperparameters.lr_patience,
    "lr_factor": Hyperparameters.lr_factor,
    "weight_decay": Hyperparameters.weight_decay,
    "pretrain": Hyperparameters.pretrain,
    "model_name": Hyperparameters.pretrain_model if Hyperparameters.pretrain else None,
    "pretrain_num_classes": 1
}

model = GraphormerLightningModule(**base_params)     

trainer = Trainer(devices='auto',
                  accelerator='auto',
                  max_epochs=Hyperparameters.max_epoch,
                  min_epochs=Hyperparameters.min_epoch,
                  logger=wandb_logger,
                  callbacks=[lr_monitor, check_point],
                  fast_dev_run=check_mode,
                  log_every_n_steps=25)

# Train and test the model
trainer.fit(model, datamodule=data_module)
if not check_mode:
    trainer.validate(model, datamodule=data_module, ckpt_path='best')