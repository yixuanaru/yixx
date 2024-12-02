import pandas as pd
import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import yaml
from typing import List, Tuple, Dict

# 設定logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置參數
CONFIG = {
    'model_params': {
        'nn': {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.5,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'batch_size': 32,
            'epochs': 150
        },
        'xgb': {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'eval_metric': 'logloss'
        }
    },
    'ensemble_weights': {
        'nn': 0.3,
        'xgb': 0.7
    }
}

class NNModel(nn.Module):
    def __init__(self, input_shape: int, hidden_layers: List[int], dropout_rate: float):
        super(NNModel, self).__init__()
        
        layers = []
        prev_size = input_shape
        
        # 動態創建隱藏層
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
            
        # 輸出層
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class MLPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")

    def load_data(self, folder_path: Path) -> Tuple[List[str], List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
        dataset_names = []
        X_trains, y_trains, X_tests = [], [], []
        
        for folder_name in folder_path.iterdir():
            if folder_name.is_dir():
                try:
                    dataset_names.append(folder_name.name)
                    
                    paths = {
                        'X_train': folder_name / "X_train.csv",
                        'y_train': folder_name / "y_train.csv",
                        'X_test': folder_name / "X_test.csv"
                    }
                    
                    if not all(path.exists() for path in paths.values()):
                        logging.warning(f"Missing CSV files in {folder_name.name}")
                        continue
                        
                    X_trains.append(pd.read_csv(paths['X_train']))
                    y_trains.append(pd.read_csv(paths['y_train']))
                    X_tests.append(pd.read_csv(paths['X_test']))
                    
                except Exception as e:
                    logging.error(f"Error processing {folder_name.name}: {str(e)}")
                    
        return dataset_names, X_trains, y_trains, X_tests

    def train_nn_model(self, X_train: torch.Tensor, y_train: torch.Tensor, input_shape: int) -> NNModel:
        nn_params = self.config['model_params']['nn']
        
        model = NNModel(
            input_shape=input_shape,
            hidden_layers=nn_params['hidden_layers'],
            dropout_rate=nn_params['dropout_rate']
        ).to(self.device)
        
        # 創建DataLoader
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=nn_params['batch_size'], shuffle=True)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=nn_params['learning_rate'],
            weight_decay=nn_params['weight_decay']
        )
        
        # 使用tqdm顯示訓練進度
        progress_bar = tqdm(range(nn_params['epochs']), desc="Training NN")
        for epoch in progress_bar:
            model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
        return model

    def train_xgb_model(self, X_train: np.ndarray, y_train: np.ndarray) -> CalibratedClassifierCV:
        xgb_params = self.config['model_params']['xgb']
        
        base_model = xgb.XGBClassifier(**xgb_params)
        model = CalibratedClassifierCV(base_model, method='sigmoid')
        model.fit(X_train, y_train)
        
        return model

    def process_dataset(self, name: str, X_train: pd.DataFrame, y_train: pd.DataFrame, 
                       X_test: pd.DataFrame, output_path: Path) -> None:
        logging.info(f"Processing dataset: {name}")
        
        # 數據預處理
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 準備神經網絡的數據
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(self.device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.device)
        
        # 訓練模型
        nn_model = self.train_nn_model(X_train_tensor, y_train_tensor, X_train.shape[1])
        xgb_model = self.train_xgb_model(X_train_scaled, y_train.values.ravel())
        
        # 預測
        nn_model.eval()
        with torch.no_grad():
            nn_pred_proba = nn_model(X_test_tensor).cpu().numpy().flatten()
            
        xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
        
        # 集成預測
        weights = self.config['ensemble_weights']
        ensemble_pred_proba = (weights['nn'] * nn_pred_proba + 
                             weights['xgb'] * xgb_pred_proba)
        
        # 保存結果
        output_file = output_path / name / 'y_predict.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'y_predict': ensemble_pred_proba}).to_csv(output_file, index=False)
        logging.info(f"Saved predictions for {name}")

def main():
    folder_path = Path("C:/Users/user/Desktop/data science/cp1/Competition_data")
    
    pipeline = MLPipeline(CONFIG)
    dataset_names, X_trains, y_trains, X_tests = pipeline.load_data(folder_path)
    
    for name, X_train, y_train, X_test in zip(dataset_names, X_trains, y_trains, X_tests):
        pipeline.process_dataset(name, X_train, y_train, X_test, folder_path)

if __name__ == "__main__":
    main()