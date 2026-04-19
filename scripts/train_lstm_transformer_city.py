from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import math
import json
import sys
import joblib

# ======================
# ARGUMENT
# ======================
if len(sys.argv) < 2:
    CITY = "Karaganda"
else:
    CITY = sys.argv[1]

# ======================
# PATHS
# ======================
DATA_DIR = Path(r"C:\air_quality_project\processed\sequence_data_by_city") / CITY
MODEL_DIR = Path(r"C:\air_quality_project\processed\models_by_city") / CITY
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_FILE = DATA_DIR / "X_train.npy"
y_TRAIN_FILE = DATA_DIR / "y_train.npy"
X_VALID_FILE = DATA_DIR / "X_valid.npy"
y_VALID_FILE = DATA_DIR / "y_valid.npy"
X_TEST_FILE = DATA_DIR / "X_test.npy"
y_TEST_FILE = DATA_DIR / "y_test.npy"

BEST_MODEL_FILE = MODEL_DIR / "lstm_transformer_best.pt"
METRICS_FILE = MODEL_DIR / "lstm_transformer_metrics.json"
TARGET_SCALER_FILE = MODEL_DIR / "target_scaler.pkl"

# ======================
# CONFIG
# ======================
BATCH_SIZE = 256
D_MODEL = 128
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 8
SEED = 42

# ======================
# REPRODUCIBILITY
# ======================
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("City:", CITY)

# ======================
# LOAD DATA
# ======================
X_train = np.load(X_TRAIN_FILE).astype(np.float32)
y_train = np.load(y_TRAIN_FILE).astype(np.float32)

X_valid = np.load(X_VALID_FILE).astype(np.float32)
y_valid = np.load(y_VALID_FILE).astype(np.float32)

X_test = np.load(X_TEST_FILE).astype(np.float32)
y_test = np.load(y_TEST_FILE).astype(np.float32)

print("\nLoaded arrays:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_valid:", X_valid.shape, "y_valid:", y_valid.shape)
print("X_test :", X_test.shape,  "y_test :", y_test.shape)

# ======================
# TARGET SCALING
# ======================
target_scaler = StandardScaler()

y_train = target_scaler.fit_transform(y_train).astype(np.float32)
y_valid = target_scaler.transform(y_valid).astype(np.float32)
y_test = target_scaler.transform(y_test).astype(np.float32)

joblib.dump(target_scaler, TARGET_SCALER_FILE)

# ======================
# DATALOADERS
# ======================
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
valid_ds = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ======================
# MODEL
# ======================
class LSTMTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        d_model=128,
        lstm_hidden=128,
        lstm_layers=2,
        n_heads=8,
        transformer_layers=2,
        ff_dim=256,
        dropout=0.2,
        output_size=2
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_hidden,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        out = self.head(x)
        return out

input_size = X_train.shape[2]
model = LSTMTransformer(
    input_size=input_size,
    d_model=D_MODEL,
    lstm_hidden=LSTM_HIDDEN,
    lstm_layers=LSTM_LAYERS,
    n_heads=TRANSFORMER_HEADS,
    transformer_layers=TRANSFORMER_LAYERS,
    ff_dim=FF_DIM,
    dropout=DROPOUT,
    output_size=2
).to(device)

print("\nModel:")
print(model)

# ======================
# LOSS / OPTIMIZER
# ======================
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

# ======================
# HELPERS
# ======================
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    n_samples = 0

    all_preds = []
    all_targets = []

    with torch.set_grad_enabled(is_train):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(yb.detach().cpu().numpy())

    avg_loss = total_loss / max(n_samples, 1)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    return avg_loss, all_preds, all_targets

def compute_metrics(y_true, y_pred):
    metrics = {}
    names = ["pm25", "pm10"]

    for i, name in enumerate(names):
        rmse = math.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        metrics[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

    return metrics

# ======================
# TRAIN LOOP
# ======================
best_valid_loss = float("inf")
epochs_without_improvement = 0
history = []

print("\nStarting training...\n")

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, train_preds, train_targets = run_epoch(model, train_loader, criterion, optimizer)
    valid_loss, valid_preds, valid_targets = run_epoch(model, valid_loader, criterion, optimizer=None)

    scheduler.step(valid_loss)

    # inverse transform for readable metrics
    train_preds_real = target_scaler.inverse_transform(train_preds)
    train_targets_real = target_scaler.inverse_transform(train_targets)

    valid_preds_real = target_scaler.inverse_transform(valid_preds)
    valid_targets_real = target_scaler.inverse_transform(valid_targets)

    train_metrics = compute_metrics(train_targets_real, train_preds_real)
    valid_metrics = compute_metrics(valid_targets_real, valid_preds_real)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics
    })

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"LR: {current_lr:.6f} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Valid Loss: {valid_loss:.4f} | "
        f"Valid PM2.5 R2: {valid_metrics['pm25']['R2']:.4f} | "
        f"Valid PM10 R2: {valid_metrics['pm10']['R2']:.4f}"
    )

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        epochs_without_improvement = 0

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "valid_loss": valid_loss,
            "config": {
                "input_size": input_size,
                "d_model": D_MODEL,
                "lstm_hidden": LSTM_HIDDEN,
                "lstm_layers": LSTM_LAYERS,
                "n_heads": TRANSFORMER_HEADS,
                "transformer_layers": TRANSFORMER_LAYERS,
                "ff_dim": FF_DIM,
                "dropout": DROPOUT,
                "output_size": 2
            }
        }, BEST_MODEL_FILE)

        print(f"  -> Best model saved at epoch {epoch}")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping triggered after epoch {epoch}")
        break

# ======================
# LOAD BEST MODEL
# ======================
checkpoint = torch.load(BEST_MODEL_FILE, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

print("\nLoaded best model from epoch:", checkpoint["epoch"])
print("Best validation loss:", checkpoint["valid_loss"])

# ======================
# FINAL EVALUATION
# ======================
train_loss, train_preds, train_targets = run_epoch(model, train_loader, criterion, optimizer=None)
valid_loss, valid_preds, valid_targets = run_epoch(model, valid_loader, criterion, optimizer=None)
test_loss, test_preds, test_targets = run_epoch(model, test_loader, criterion, optimizer=None)

# inverse transform after final evaluation
train_preds_real = target_scaler.inverse_transform(train_preds)
train_targets_real = target_scaler.inverse_transform(train_targets)

valid_preds_real = target_scaler.inverse_transform(valid_preds)
valid_targets_real = target_scaler.inverse_transform(valid_targets)

test_preds_real = target_scaler.inverse_transform(test_preds)
test_targets_real = target_scaler.inverse_transform(test_targets)

train_metrics = compute_metrics(train_targets_real, train_preds_real)
valid_metrics = compute_metrics(valid_targets_real, valid_preds_real)
test_metrics = compute_metrics(test_targets_real, test_preds_real)

results = {
    "city": CITY,
    "best_epoch": checkpoint["epoch"],
    "best_valid_loss": checkpoint["valid_loss"],
    "train_loss": train_loss,
    "valid_loss": valid_loss,
    "test_loss": test_loss,
    "train_metrics": train_metrics,
    "valid_metrics": valid_metrics,
    "test_metrics": test_metrics,
    "history": history
}

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nFINAL RESULTS")
print("=" * 50)

for split_name, metrics in [
    ("TRAIN", train_metrics),
    ("VALID", valid_metrics),
    ("TEST", test_metrics)
]:
    print(f"\n{split_name}")
    print(f"PM2.5 -> RMSE: {metrics['pm25']['RMSE']:.4f}, "
          f"MAE: {metrics['pm25']['MAE']:.4f}, "
          f"R2: {metrics['pm25']['R2']:.4f}")
    print(f"PM10  -> RMSE: {metrics['pm10']['RMSE']:.4f}, "
          f"MAE: {metrics['pm10']['MAE']:.4f}, "
          f"R2: {metrics['pm10']['R2']:.4f}")

print("\nSaved files:")
print(BEST_MODEL_FILE)
print(METRICS_FILE)
print(TARGET_SCALER_FILE)