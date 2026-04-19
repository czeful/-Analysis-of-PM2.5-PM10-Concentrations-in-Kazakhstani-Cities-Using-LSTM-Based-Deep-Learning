from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

CITY = "Astana"

DATA_FILE = Path(r"C:\air_quality_project\processed\test_pm25.csv")
SEQ_DIR = Path(r"C:\air_quality_project\processed\sequence_data_pm25_by_city") / CITY
MODEL_DIR = Path(r"C:\air_quality_project\processed\models_pm25_by_city") / CITY
OUT_FILE = Path(r"C:\air_quality_project\processed\astana_pm25_error_analysis.csv")

X_TEST_FILE = SEQ_DIR / "X_test.npy"
y_TEST_FILE = SEQ_DIR / "y_test.npy"
MODEL_FILE = MODEL_DIR / "pm25_transformer_best.pt"

LOOKBACK = 72

# ======================
# LOAD RAW TEST TABLE
# ======================
df = pd.read_csv(DATA_FILE)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df[df["city"] == CITY].copy().sort_values("datetime").reset_index(drop=True)

# sequences start after lookback
df = df.iloc[LOOKBACK:].copy().reset_index(drop=True)

# ======================
# LOAD TEST ARRAYS
# ======================
X_test = np.load(X_TEST_FILE).astype(np.float32)
y_test = np.load(y_TEST_FILE).astype(np.float32)

# ======================
# MODEL DEF
# ======================
class PM25Transformer(nn.Module):
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
        output_size=1
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
        return self.head(x)

# ======================
# LOAD MODEL
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_FILE, map_location=device)
config = checkpoint["config"]

model = PM25Transformer(
    input_size=config["input_size"],
    d_model=config["d_model"],
    lstm_hidden=config["lstm_hidden"],
    lstm_layers=config["lstm_layers"],
    n_heads=config["n_heads"],
    transformer_layers=config["transformer_layers"],
    ff_dim=config["ff_dim"],
    dropout=config["dropout"],
    output_size=config["output_size"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ======================
# PREDICT
# ======================
with torch.no_grad():
    X_tensor = torch.from_numpy(X_test).to(device)
    preds_log = model(X_tensor).cpu().numpy().reshape(-1)
    y_true_log = y_test.reshape(-1)

preds = np.expm1(preds_log)
y_true = np.expm1(y_true_log)

# align with dataframe length
df = df.iloc[:len(preds)].copy()

df["pm25_true"] = y_true
df["pm25_pred"] = preds
df["abs_error"] = np.abs(df["pm25_true"] - df["pm25_pred"])
df["sq_error"] = (df["pm25_true"] - df["pm25_pred"]) ** 2

df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour

def get_season(m):
    if m in [12, 1, 2]:
        return "winter"
    elif m in [3, 4, 5]:
        return "spring"
    elif m in [6, 7, 8]:
        return "summer"
    return "autumn"

df["season"] = df["month"].apply(get_season)

df["pm25_regime"] = pd.cut(
    df["pm25_true"],
    bins=[-0.1, 15, 35, 75, 2000],
    labels=["low", "moderate", "high", "extreme"]
)

df.to_csv(OUT_FILE, index=False)

print("Saved detailed error file:", OUT_FILE)

print("\nOverall:")
rmse = math.sqrt(mean_squared_error(df["pm25_true"], df["pm25_pred"]))
mae = mean_absolute_error(df["pm25_true"], df["pm25_pred"])
r2 = r2_score(df["pm25_true"], df["pm25_pred"])
print("RMSE:", round(rmse, 4))
print("MAE :", round(mae, 4))
print("R2  :", round(r2, 4))

print("\nBy season:")
season_summary = df.groupby("season").apply(
    lambda x: pd.Series({
        "n": len(x),
        "RMSE": math.sqrt(mean_squared_error(x["pm25_true"], x["pm25_pred"])),
        "MAE": mean_absolute_error(x["pm25_true"], x["pm25_pred"]),
        "R2": r2_score(x["pm25_true"], x["pm25_pred"])
    })
)
print(season_summary)

print("\nBy regime:")
regime_summary = df.groupby("pm25_regime").apply(
    lambda x: pd.Series({
        "n": len(x),
        "RMSE": math.sqrt(mean_squared_error(x["pm25_true"], x["pm25_pred"])),
        "MAE": mean_absolute_error(x["pm25_true"], x["pm25_pred"]),
        "R2": r2_score(x["pm25_true"], x["pm25_pred"])
    })
)
print(regime_summary)

print("\nWorst 20 hours:")
print(
    df.sort_values("abs_error", ascending=False)[
        ["datetime", "pm25_true", "pm25_pred", "abs_error", "season", "pm25_regime"]
    ].head(20)
)