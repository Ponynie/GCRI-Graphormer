from model import GraphormerLightningModule
from datamodule import GraphormerDataModule
from pytorch_lightning import Trainer
from transformers import GraphormerConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from hparam import Hyperparameters
import torch

# torch.set_float32_matmul_precision('high')
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
)

# Prepare and set up the data
data_module.prepare_data()
data_module.setup()

# Set up callbacks and logger
lr_monitor = LearningRateMonitor(logging_interval='epoch')
wandb_logger = WandbLogger(project='Graphormer-Molecule', save_dir='wandb_log')
check_point = ModelCheckpoint(monitor='val_loss')
    
# Model configuration
config = GraphormerConfig(num_classes=1, 
                          num_hidden_layers=Hyperparameters.num_hidden_layers, 
                          hidden_size=Hyperparameters.hidden_size, 
                          num_attention_heads=Hyperparameters.num_attention_heads,)

# Instantiate model and trainer
if not Hyperparameters.pretrain:
    model = GraphormerLightningModule(config=config, learning_rate=Hyperparameters.learning_rate)
else:
    model = GraphormerLightningModule(config=None, learning_rate=Hyperparameters.learning_rate, 
                                      pretrain=True, 
                                      model_name=Hyperparameters.pretrain_model, 
                                      pretrain_num_classes=1)

trainer = Trainer(devices='auto',
                  accelerator='cpu',
                  max_epochs=Hyperparameters.max_epoch,
                  min_epochs=Hyperparameters.min_epoch,
                  logger=wandb_logger,
                  callbacks=[lr_monitor, check_point],
                  fast_dev_run=check_mode,
                  log_every_n_steps=50)

# Train and test the model
trainer.fit(model, datamodule=data_module)
if not check_mode:
    trainer.validate(model, datamodule=data_module, ckpt_path='best')