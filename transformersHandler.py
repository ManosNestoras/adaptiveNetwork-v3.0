import os
import pickle
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from dateutil.relativedelta import relativedelta
import torch
from sklearn.metrics import mean_absolute_error
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, data, win_len):
        self.X = []
        self.y = []
        for i in range(len(data) - win_len):
            self.X.append(data[i:i+win_len, :])
            self.y.append(data[i+win_len, 0])  # target is in the first column
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=3, d_model=128, nhead=8, num_layers=4, dropout=0.1, win_length=12):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(win_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.attn_pool = nn.Linear(d_model, 1)
        self.output = nn.Linear(d_model, 1)
        self.win_length = win_length

    def forward(self, x):
        B, T, _ = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.input_proj(x) + self.pos_embedding(pos)
        x = self.transformer(x)
        weights = torch.softmax(self.attn_pool(x), dim=1)
        pooled = (x * weights).sum(dim=1)
        return self.output(pooled).squeeze()

class TransformerTimeSeriesHandler:
    def __init__(self, model_dir, win_length=12, batch_size=64, epochs=50, lr=1e-4, name=None, device=None):
        self.model_dir = model_dir
        self.win_length = win_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model_cache = None
        self.scaler_cache = None
        self.name = name
        os.makedirs(self.model_dir, exist_ok=True)
        if self.name:
            self.load_all_models()

    def preprocess(self, df):
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        df = df.set_index('TimeStamp').sort_index()
        full_range = pd.date_range(df.index[0], df.index[-1], freq='5T')
        df_full = pd.DataFrame(index=full_range)
        df_full = df_full.join(df)

        df_full['day_of_week'] = df_full.index.dayofweek / 6
        df_full['hour'] = df_full.index.hour / 23

        for col in df.columns:
            df_full[col] = df_full[col].fillna(
                df_full.groupby([df_full.index.dayofweek, df_full.index.time])[col].transform('mean')
            )

            if df_full[col].isna().any():
                print(f"⚠️ {df_full[col].isna().sum()} NaNs remain in '{col}', applying fallback...")
                df_full[col] = df_full[col].fillna(method="ffill").fillna(method="bfill")
        return df_full

    def split_df(self, df, test_days):
        test_start = df.index[-1] - timedelta(days=test_days)
        split_point_val = df.index[int(len(df) * 0.85)]
        split_point_train = df.index[int(len(df) * 0.7)]
        train = df[df.index < split_point_train]
        val = df[(df.index >= split_point_train) & (df.index < split_point_val)]
        test = df[df.index >= test_start]
        return train, val, test

    def train_model(self, model, train_loader, val_loader, optimizer, loss_fn, col_name=""):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_loss = np.mean([
                    loss_fn(model(x.to(self.device)), y.to(self.device)).item()
                    for x, y in val_loader
                ])
            print(f"[{col_name}] Epoch {epoch:02d}/{self.epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")


    def train_on_dataframe(self, df, column, save_name, test_days=7, min_improvement=0.01):
        

        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        print(f"\n📦 Training model: {save_name}")

        processed_df = self.preprocess(df[['TimeStamp', column]].copy())
        train_df, val_df, test_df = self.split_df(processed_df, test_days)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train = scaler_X.fit_transform(train_df[['hour', 'day_of_week']])
        y_train = scaler_y.fit_transform(train_df[[column]])

        X_val = scaler_X.transform(val_df[['hour', 'day_of_week']])
        y_val = scaler_y.transform(val_df[[column]])

        X_test = scaler_X.transform(test_df[['hour', 'day_of_week']])
        y_test = scaler_y.transform(test_df[[column]])

        train_data = np.hstack([y_train, X_train])
        val_data = np.hstack([y_val, X_val])
        test_data = np.hstack([y_test, X_test])

        train_loader = DataLoader(TimeSeriesDataset(train_data, self.win_length), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TimeSeriesDataset(val_data, self.win_length), batch_size=self.batch_size)
        test_loader = DataLoader(TimeSeriesDataset(test_data, self.win_length), batch_size=self.batch_size)

        # ==== OLD MODEL EVALUATION (if exists) ====
        old_mae = np.inf
        model_path = os.path.join(self.model_dir, f"{save_name}_model.pt")
        scaler_path = os.path.join(self.model_dir, f"{save_name}_scaler.pkl")

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                print(f"🧠 Found existing model for {save_name}. Evaluating old MAE...")
                old_model = TransformerRegressor(input_dim=3, win_length=self.win_length).to(self.device)
                old_model.load_state_dict(torch.load(model_path, map_location=self.device))
                old_model.eval()

                with open(scaler_path, "rb") as f:
                    old_scaler_y = pickle.load(f)

                old_preds, old_actuals = [], []
                with torch.no_grad():
                    for x, y in test_loader:
                        x = x.to(self.device)
                        pred = old_model(x).cpu().numpy()
                        y = np.atleast_1d(y.numpy())
                        pred = np.atleast_1d(pred)
                        old_preds.append(pred)
                        old_actuals.append(y)

                old_preds = old_scaler_y.inverse_transform(np.concatenate(old_preds).reshape(-1, 1)).flatten()
                old_actuals = old_scaler_y.inverse_transform(np.concatenate(old_actuals).reshape(-1, 1)).flatten()
                old_mae = mean_absolute_error(old_actuals, old_preds)
                print(f"📉 Old MAE: {old_mae:.4f}")
            except Exception as e:
                print(f"[WARNING] Could not evaluate old model: {e}")

        # ==== TRAIN NEW MODEL ====
        model = TransformerRegressor(input_dim=3, win_length=self.win_length).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.L1Loss()

        self.train_model(model, train_loader, val_loader, optimizer, loss_fn, col_name=save_name)

        # ==== EVALUATE NEW MODEL ====
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                pred = model(x).cpu().numpy()
                pred = np.atleast_1d(pred)
                y = np.atleast_1d(y.numpy())
                preds.append(pred)
                actuals.append(y)

        preds = scaler_y.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()
        actuals = scaler_y.inverse_transform(np.concatenate(actuals).reshape(-1, 1)).flatten()
        new_mae = mean_absolute_error(actuals, preds)
        print(f"📊 New MAE for {save_name}: {new_mae:.4f}")

        # ==== SAVE IF BETTER OR FIRST TIME ====
        save_model = False

        if not os.path.exists(model_path):
            print(f"🆕 No previous model. Saving new model for {save_name}.")
            save_model = True
        elif new_mae < old_mae - min_improvement:
            print(f"✅ New model is better. Overwriting old model for {save_name}.")
            save_model = True
        else:
            print(f"⛔ New model not better. Keeping existing model for {save_name}.")

        if save_model:
            torch.save(model.state_dict(), model_path)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler_y, f)



    def load_all_models(self):
        """
        Preloads all models and scalers from self.model_dir into CPU memory.
        Assumes model filenames are in the format {column}_model.pt and {column}_scaler.pkl
        """

        model_path = os.path.join(self.model_dir, f"{self.name}_model.pt")
        scaler_path = os.path.join(self.model_dir, f"{self.name}_scaler.pkl")

        model = TransformerRegressor(input_dim=3, win_length=self.win_length)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        self.model_cache = model

        with open(scaler_path, "rb") as f:
            self.scaler_cache = pickle.load(f)


    def predict(self, df):
        if self.model_cache is None or self.scaler_cache is None:
            raise ValueError("Model and scaler must be loaded before prediction.")

        df = df.copy()
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        df = df.set_index('TimeStamp').sort_index()

        value_col = df.columns[0]
        df['day_of_week'] = df.index.dayofweek / 6
        df['hour'] = df.index.hour / 23

        if len(df) < self.win_length:
            raise ValueError(f"Not enough data to make prediction (need at least {self.win_length} rows).")

        df = df.iloc[-self.win_length:]

        y = df[[value_col]].values
        X = df[['hour', 'day_of_week']].values
        sequence = np.hstack([y, X])
        sequence[:, 0:1] = self.scaler_cache.transform(sequence[:, 0:1])

        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cache.to(device)
        self.model_cache.eval()

        with torch.no_grad():
            pred = self.model_cache(sequence_tensor).cpu().numpy().reshape(-1, 1)

        pred_original = self.scaler_cache.inverse_transform(pred).flatten()[0]

        # Move model back to CPU and clear GPU memory
        self.model_cache.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if pred_original < 0:
            return 0

        return int(round(pred_original))
