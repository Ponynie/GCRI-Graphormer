#hparam.py

import os

class Hyperparameters:
    check_mode = False
    pretrain = True
    pretrain_model = "clefourrier/pcqm4mv2_graphormer_base"
    max_epoch = 50
    min_epoch = 5
    batch_size = 64
    num_workers = int(os.cpu_count()) if int(os.cpu_count()) <= 8 else 8
    train_size = 0.7
    val_size = 0.1
    test_size = 0.2
    random_state = 42
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 32
    learning_rate = 0.001
    weight_decay = 0.001 
    lr_patience = 10
    patience = 10
    