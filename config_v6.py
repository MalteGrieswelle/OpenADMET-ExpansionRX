"""Configuration file for Chemprop ensemble model training and prediction."""

from pathlib import Path
import pandas as pd

# Data paths
TRAINING_DATA = "/nfs/home/m_grie10/storage/02_projects/OpenADMET-ExpansionRX/training_data_v6.csv"
CHALLENGE_TEST_DATA = "/nfs/home/m_grie10/storage/02_projects/OpenADMET-ExpansionRX/expansion_data_test_blinded.csv"

# Model checkpoint directory
CHECKPOINT_DIR = Path("/nfs/home/m_grie10/storage/02_projects/OpenADMET-ExpansionRX/checkpoints_v6")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# Results directory
RESULTS_DIR = Path("/nfs/home/m_grie10/storage/02_projects/OpenADMET-ExpansionRX/results_v6")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Log directory
LOG_DIR = Path("/nfs/home/m_grie10/storage/02_projects/OpenADMET-ExpansionRX/logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Target endpoints (9 challenge endpoints)
CHALLENGE_ENDPOINTS = [
    "LogD",
    "KSOL",
    "HLM CLint",
    "MLM CLint",
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
    "MPPB",
    "MBPB",
    "MGMB"
]

SMILES_COL = "SMILES"

df = pd.read_csv(TRAINING_DATA)
TRAINING_TARGETS = df.columns[1:].tolist()


# Training configuration
TRAIN_TEST_SPLIT = 0.9
BATCH_SIZE = 128
EPOCHS = 1000
PATIENCE = 100
LOG_NAME = "chemprop_v6"

# Model architectures and aggregation methods to vary
MODEL_CONFIGS = [
    # --- Group 1 ---
    {"message_hidden_dim": 300, "depth": 4, "dropout": 0, "ff_layer": 3, "ff_hidden_dim": 512, "criterion": "MAE", "aggregation": "sum", "aggregation_norm": 100},    #0
    {"message_hidden_dim": 300, "depth": 4, "dropout": 0, "ff_layer": 3, "ff_hidden_dim": 512, "criterion": "MAE", "aggregation": "sum", "aggregation_norm": 100},    #
    {"message_hidden_dim": 300, "depth": 4, "dropout": 0, "ff_layer": 3, "ff_hidden_dim": 512, "criterion": "MAE", "aggregation": "sum", "aggregation_norm": 100},    #
    {"message_hidden_dim": 300, "depth": 4, "dropout": 0, "ff_layer": 3, "ff_hidden_dim": 512, "criterion": "MAE", "aggregation": "sum", "aggregation_norm": 100},    #
    {"message_hidden_dim": 300, "depth": 4, "dropout": 0, "ff_layer": 3, "ff_hidden_dim": 512, "criterion": "MAE", "aggregation": "sum", "aggregation_norm": 100},    #

    # --- Group 2 ---
    {"message_hidden_dim": 300, "depth": 3, "dropout": 0, "ff_layer": 4, "ff_hidden_dim": 300, "criterion": "MAE", "aggregation": "norm", "aggregation_norm": 100},   #5
    {"message_hidden_dim": 300, "depth": 3, "dropout": 0, "ff_layer": 4, "ff_hidden_dim": 300, "criterion": "MAE", "aggregation": "norm", "aggregation_norm": 100},   #
    {"message_hidden_dim": 300, "depth": 3, "dropout": 0, "ff_layer": 4, "ff_hidden_dim": 300, "criterion": "MAE", "aggregation": "norm", "aggregation_norm": 100},   #
    {"message_hidden_dim": 300, "depth": 3, "dropout": 0, "ff_layer": 4, "ff_hidden_dim": 300, "criterion": "MAE", "aggregation": "norm", "aggregation_norm": 100},   #
    {"message_hidden_dim": 300, "depth": 3, "dropout": 0, "ff_layer": 4, "ff_hidden_dim": 300, "criterion": "MAE", "aggregation": "norm", "aggregation_norm": 100},   #

    # --- Group 3 ---
    {"message_hidden_dim": 300, "depth": 6, "dropout": 0, "ff_layer": 2, "ff_hidden_dim": 512, "criterion": "MAE", "aggregation": "attentive", "aggregation_norm": 100},   #10
    {"message_hidden_dim": 300, "depth": 6, "dropout": 0, "ff_layer": 2, "ff_hidden_dim": 512, "criterion": "MAE", "aggregation": "attentive", "aggregation_norm": 100},   #
    {"message_hidden_dim": 300, "depth": 6, "dropout": 0, "ff_layer": 2, "ff_hidden_dim": 512, "criterion": "MAE", "aggregation": "attentive", "aggregation_norm": 100},   #
    {"message_hidden_dim": 300, "depth": 6, "dropout": 0, "ff_layer": 2, "ff_hidden_dim": 512, "criterion": "MAE", "aggregation": "attentive", "aggregation_norm": 100},   #
    {"message_hidden_dim": 300, "depth": 6, "dropout": 0, "ff_layer": 2, "ff_hidden_dim": 512, "criterion": "MAE", "aggregation": "attentive", "aggregation_norm": 100},   #

    # --- Group 4 ---
    {"message_hidden_dim": 512, "depth": 2, "dropout": 0, "ff_layer": 4, "ff_hidden_dim": 256, "criterion": "MAE", "aggregation": "sum", "aggregation_norm": 100},  #15
    {"message_hidden_dim": 512, "depth": 2, "dropout": 0, "ff_layer": 4, "ff_hidden_dim": 256, "criterion": "MAE", "aggregation": "sum", "aggregation_norm": 100},    #
    {"message_hidden_dim": 512, "depth": 2, "dropout": 0, "ff_layer": 4, "ff_hidden_dim": 256, "criterion": "MAE", "aggregation": "sum", "aggregation_norm": 100},    #
    {"message_hidden_dim": 512, "depth": 2, "dropout": 0, "ff_layer": 4, "ff_hidden_dim": 256, "criterion": "MAE", "aggregation": "sum", "aggregation_norm": 100},    #
    {"message_hidden_dim": 512, "depth": 2, "dropout": 0, "ff_layer": 4, "ff_hidden_dim": 256, "criterion": "MAE", "aggregation": "sum", "aggregation_norm": 100},    #

    # --- Group 5 ---
    {"message_hidden_dim": 512, "depth": 5, "dropout": 0, "ff_layer": 2, "ff_hidden_dim": 1024, "criterion": "MAE", "aggregation": "mean", "aggregation_norm": 100}, #20
    {"message_hidden_dim": 512, "depth": 5, "dropout": 0, "ff_layer": 2, "ff_hidden_dim": 1024, "criterion": "MAE", "aggregation": "mean", "aggregation_norm": 100}, #
    {"message_hidden_dim": 512, "depth": 5, "dropout": 0, "ff_layer": 2, "ff_hidden_dim": 1024, "criterion": "MAE", "aggregation": "mean", "aggregation_norm": 100}, #
    {"message_hidden_dim": 512, "depth": 5, "dropout": 0, "ff_layer": 2, "ff_hidden_dim": 1024, "criterion": "MAE", "aggregation": "mean", "aggregation_norm": 100}, #
    {"message_hidden_dim": 512, "depth": 5, "dropout": 0, "ff_layer": 2, "ff_hidden_dim": 1024, "criterion": "MAE", "aggregation": "mean", "aggregation_norm": 100}, #
]
