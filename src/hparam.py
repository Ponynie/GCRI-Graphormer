#hparam.py

import os

class Hyperparameters:
    check_mode = True
    pretrain = False
    pretrain_model = "clefourrier/pcqm4mv2_graphormer_base"
    max_epoch = 20
    min_epoch = 1
    batch_size = 3
    num_workers = int(os.cpu_count()) if int(os.cpu_count()) <= 4 else 4
    train_size = 0.7
    val_size = 0.1
    test_size = 0.2
    random_state = 42
    hidden_size = 768
    num_hidden_layers = 2
    num_attention_heads = 1
    learning_rate = 0.001
    weight_decay = 0.001 
    lr_patience = 10
    patience = 10
    