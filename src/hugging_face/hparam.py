#hparam.py

import os

class Hyperparameters:
    check_mode = True
    pretrain = False
    pretrain_model = "clefourrier/pcqm4mv2_graphormer_base"
    max_epoch = 50
    min_epoch = 5
    batch_size = 2
    num_workers = int(os.cpu_count()) if int(os.cpu_count()) <= 4 else 4
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1
    random_state = 42
    num_layers = 2
    embedding_dim = 768
    ffn_embedding_dim = 768
    num_attention_heads = 1
    dropout = 0.2
    weight_decay = 1e-2
    learning_rate = 1
    lr_factor = 1e-3
    lr_patience = 5

    