from src.utils import *
from settings import PROJECT_ROOT
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math

def run_dataset(case_study: str):
    """
    """
    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)

    output_dir = os.path.join(PROJECT_ROOT, "results", case_study, "ANN")
    os.makedirs(output_dir, exist_ok=True)

    levels = get_nodes_by_level(config["Load Tree"])
    first_level = levels[0]

    groups_path = os.path.join(PROJECT_ROOT, "results", case_study, "groups.csv")
    groups = pd.read_csv(groups_path, parse_dates=["timestamp"])
    groups["date"] = groups["timestamp"].dt.date
    context_ids = pd.read_csv(os.path.join(PROJECT_ROOT, "results", case_study, "time_windows.csv")).id.unique()
    cluster_cols = [col for col in groups.columns if col.startswith("Cluster_")]

    all_normals = []

    for leaf in first_level:
        print(f"{leaf}...", end="")
        for context in context_ids:
            for cluster_col in cluster_cols:
                cluster = int(cluster_col.split("_")[-1])
                df_normal = run_profile_power_temp(case_study, leaf, context, cluster)[0]
                all_normals.append(df_normal)
        print(f"    ok")
    print("\n")
    df = pd.concat(all_normals, ignore_index=False)
    df.to_csv(os.path.join(output_dir, "df_ANN.csv"))
    return df

class PowerProfileDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.df = df
        self.feature_cols = feature_cols
        feature_data = []
        for col in feature_cols:
            if df[col].apply(lambda x: isinstance(x, list)).all():
                expanded = pd.DataFrame(df[col].tolist(), index=df.index)
                feature_data.append(expanded)
            else:
                feature_data.append(df[[col]])
        features_df = pd.concat(feature_data, axis=1)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(features_df.values).astype(np.float32)
        self.targets = np.array(df["power_profile"].tolist(), dtype=np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.targets[idx])

class PowerNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(PowerNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

def run_ANN(case_study: str):
    df_path = os.path.join(PROJECT_ROOT, "results", case_study, "ANN", "df_ANN.csv")
    if not os.path.exists(df_path):
        print("\nCreating dataset for ANN training...\n")
        run_dataset(case_study)
    else:
        print("\nANN training dataset already exists.\n")

    df = pd.read_csv(df_path, index_col=0, parse_dates=True)
    df["power_profile"] = df["power_profile"].apply(eval)
    df["temp_profile"] = df["temp_profile"].apply(eval)



    context_ids = df["Context"].unique()
    base_model_dir = os.path.join(PROJECT_ROOT, "results", case_study, "ANN", "Ann_models")
    os.makedirs(base_model_dir, exist_ok=True)

    for thermal_sensitive in [True, False]:
        model_type = "ts" if thermal_sensitive else "nts"
        model_dir = os.path.join(base_model_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)

        for context in context_ids:
            print(f"Model {model_type.upper()} for context {context}...", end="")

            df_context = df[df["Context"] == context].copy()

            if thermal_sensitive:
                df_model = df_context.copy()
                feature_cols = [col for col in df_model.columns if col != "power_profile"]
            else:
                df_model = df_context.drop(columns=["temp_profile", "Mean_Temp"])
                feature_cols = [
                    col for col in df_model.columns
                    if col != "power_profile" and np.issubdtype(df_model[col].dtype, np.number)
                ]

            dataset = PowerProfileDataset(df_model, feature_cols)
            train_size = int(0.80 * len(dataset))
            test_size = len(dataset) - train_size
            train_ds, test_ds = random_split(dataset, [train_size, test_size])
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

            model = PowerNet(input_size=len(dataset[0][0]), output_size=len(dataset[0][1]))
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(50):
                model.train()
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()

            model.eval()
            test_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    preds = model(X_batch)
                    test_loss += criterion(preds, y_batch).item() * X_batch.size(0)
                    rmse = math.sqrt(test_loss / len(test_loader.dataset))
            print(f" Test RMSE: {rmse:.4f}")

            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': len(dataset[0][0]),
                'output_size': len(dataset[0][1]),
                'scaler': dataset.scaler,
                'feature_cols': feature_cols
            }, os.path.join(model_dir, f"model_ctx_{context}.pth"))


def predict_profile(case_study: str,
                    thermal_sensitive: bool,
                    context: int,
                    cluster: int,
                    subload_name: str,
                    weekday: int,
                    temp_profile: list = None,
                    mean_temp: float = None):
    """
    Predice il profilo di potenza in base alle feature fornite.
    """
    results_path = os.path.join(PROJECT_ROOT, "results", case_study)
    time_windows = pd.read_csv(os.path.join(results_path, "time_windows.csv"))
    print(f"\nContext {context} ->   predicting {time_windows['observations'][context-1]} timestep long profile "
          f"[{time_windows['from'][context-1]} - {time_windows['to'][context-1]})...\n")
    model_type = "ts" if thermal_sensitive else "nts"
    model_path = os.path.join(PROJECT_ROOT, "results", case_study, "ANN", "Ann_models", model_type, f"model_ctx_{context}.pth")

    if not os.path.exists(model_path):
        print("Creating ANN models...")
        run_ANN(case_study)
    else:
        print("ANN model already exists \n")

    with open(os.path.join(PROJECT_ROOT, "data", case_study, "config.json"), "r") as f:
        config = json.load(f)
    first_level = get_nodes_by_level(config["Load Tree"])[0]
    subload_map = map_subload(first_level, from_type="subload", to_type="number")
    subload = subload_map[subload_name]

    checkpoint = torch.load(model_path)
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    scaler = checkpoint['scaler']
    feature_cols = checkpoint['feature_cols']

    model = PowerNet(input_size=input_size, output_size=output_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepara le feature
    feature_data = []
    for col in feature_cols:
        if col == "temp_profile":
            if not thermal_sensitive:
                continue
            if temp_profile is None:
                raise ValueError("temp_profile richiesto per thermal_sensitive=True")
            feature_data.append(temp_profile)
        elif col == "Mean_Temp":
            if not thermal_sensitive:
                continue
            if mean_temp is None:
                raise ValueError("mean_temp richiesto per thermal_sensitive=True")
            feature_data.append([mean_temp])
        elif col == "weekday":
            feature_data.append([weekday])
        elif col == "Subload":
            feature_data.append([subload])
        elif col == "Context":
            feature_data.append([context])
        elif col == "Cluster":
            feature_data.append([cluster])
        else:
            raise ValueError(f"Colonna non gestita: {col}")

    x = np.concatenate(feature_data).reshape(1, -1)
    x_scaled = scaler.transform(x).astype(np.float32)
    x_tensor = torch.tensor(x_scaled)

    with torch.no_grad():
        y_pred = model(x_tensor).numpy().flatten()
    print(f"Power profile predicion [W]:\n{y_pred}")
    return y_pred




if __name__ == "__main__":
    predict_profile(
        case_study="Cabina",
        thermal_sensitive=False,
        context=1,
        cluster=1,
        subload_name="Rooftop 1",
        weekday=0
    )