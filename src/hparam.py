#hparam.py

import os

class Hyperparameters:
    check_mode = False
    pretrain = False
    pretrain_model = "clefourrier/pcqm4mv2_graphormer_base"
    max_epoch = 50
    min_epoch = 5
    batch_size = 32
    num_workers = int(os.cpu_count()) if int(os.cpu_count()) <= 4 else 4
    train_size = 0.7
    val_size = 0.1
    test_size = 0.2
    random_state = 42
    num_layers = 8
    embedding_dim = 1024
    ffn_embedding_dim = 1024
    num_attention_heads = 1
    dropout = 0.2
    learning_rate = 1e-2
    weight_decay = 1e-3 
    lr_patience = 5
    lr_factor = 0.1
    