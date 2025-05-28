import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data_and_create_dataloaders(config):
    # 1. Load the CSV file
    df = pd.read_csv(config["filepath"])

    # 2. Split into features (X) and target (y)
    X = df.drop(columns=[config["target_col"]]).values
    y = df[config["target_col"]].values.reshape(-1, 1)

    # 3. Normalize the features and target using StandardScaler
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # 4. Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # 5. Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # 6. Create TensorDataset objects
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # 7. Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # 8. Set input dimension in the config (used for model initialization)
    config["input_dim"] = X.shape[1]

    return train_loader, val_loader
