# config.py

CONFIG = {
    "filepath": r"C:\Users\97250\OneDrive\Desktop\Ann\Data\archive\stocks",
    "target_col": "Close",
    "batch_size": 64,
    "learning_rate": 0.0005,
    "epochs": 50,
    "input_dim": None,  # will be set dynamically based on data
    "model_layers": [256, 128, 64, 32],  # neurons in each hidden layer
    "dropout_rate": 0.3,
    }
