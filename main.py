import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from models.modelAnn import DynamicANN
from train import train_loop, eval_loop
import torch.nn as nn
import torch.optim as optim

CONFIG = {
    "filepath": r"C:\Users\97250\OneDrive\Desktop\Ann\Data\archive",
    "target_col": "Close",
    "batch_size": 64,
    "learning_rate": 0.0005,
    "epochs": 50,
    "input_dim": None,
    "model_layers": [256, 128, 64, 32],
    "dropout_rate": 0.3,
}

def main():
    print("Loading data...")
    print("Script started...")
    df = pd.read_csv(CONFIG["filepath"])


    features = df.drop(columns=[CONFIG["target_col"]]).values
    targets = df[CONFIG["target_col"]].values.reshape(-1, 1)

    CONFIG["input_dim"] = features.shape[1]  #update to the number of features

    #translate to tensor PyTorch
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicANN(CONFIG["input_dim"], CONFIG["model_layers"], CONFIG["dropout_rate"]).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    for epoch in range(CONFIG["epochs"]):
        train_loss = train_loop(model, dataloader, criterion, optimizer, device)
        val_loss = eval_loop(model, dataloader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

if __name__ == "__main__":
    main()
