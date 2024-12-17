#hparam.py

import os

class Hyperparameters:
    pretrain = False
    max_epoch = 20
    min_epoch = 2
    batch_size = 2
    num_workers = int(os.cpu_count() / 2)
    train_size = 0.7
    val_size = 0.1
    test_size = 0.2
    random_state = 42
    patience = 10
    hidden_size = 768
    num_hidden_layers = 2
    num_attention_heads = 1
    learning_rate = 0.001
    weight_decay = 0.001 
    lr_patience = 10
    