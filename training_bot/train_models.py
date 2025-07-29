"""
Professional Model Training Script for Forex Trading
Профессиональный скрипт обучения моделей для торговли на форекс
"""

import os
import sys
import asyncio
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

# ML Models
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Hyperparameter optimization
import optuna
from optuna.integration import TFKerasPruningCallback

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Custom imports
sys.path.append('..')
from trading_bot.mt5_connector.mt5_manager import MT5Manager
from loguru import logger

warnings.filterwarnings('ignore')


class ForexModelTrainer:
    """Профессиональный тренер моделей для форекс"""
    
    def __init__(self, config_path: str = "training_config.json"):
        self.config = self._load_config(config_path)
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Setup GPU if available
        self._setup_gpu()
        
        logger.info("ForexModelTrainer инициализирован")
    
    def _load_config(self, config_path: str) -> dict:
        """Загрузка конфигурации обучения"""
        default_config = {
            "data": {
                "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
                "timeframes": ["H1"],
                "start_date": "2020-01-01",
                "end_date": "2024-01-01",
                "test_size": 0.2,
                "validation_size": 0.2
            },
            "features": {
                "technical_indicators": True,
                "price_features": True,
                "volume_features": True,
                "lag_features": [1, 2, 3, 5, 10],
                "rolling_features": [10, 20, 50]
            },
            "models": {
                "xgboost": {
                    "enabled": True,
                    "optimize": True,
                    "trials": 100
                },
                "lightgbm": {
                    "enabled": True,
                    "optimize": True,
                    "trials": 100
                },
                "lstm": {
                    "enabled": True,
                    "optimize": True,
                    "trials": 50,
                    "sequence_length": 60,
                    "epochs": 100
                },
                "ensemble": {
                    "enabled": True,
                    "optimize_weights": True
                }
            },
            "training": {
                "cv_folds": 5,
                "early_stopping": True,
                "save_models": True,
                "plot_results": True
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return {**default_config, **config}
        except FileNotFoundError:
            logger.warning(f"Конфиг {config_path} не найден, используем стандартный")
            return default_config
    
    def _setup_gpu(self):
        """Настройка GPU для TensorFlow"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU настроен: {len(gpus)} устройств")
            except RuntimeError as e:
                logger.error(f"Ошибка настройки GPU: {e}")
        else:
            logger.warning("GPU не найден, используем CPU")
    
    async def collect_data(self) -> Dict[str, pd.DataFrame]:
        """Сбор данных для обучения"""
        logger.info("Начинаем сбор данных...")
        
        # В реальной ситуации здесь будет подключение к MT5
        # Для демонстрации создадим синтетические данные
        data = {}
        
        for symbol in self.config["data"]["symbols"]:
            logger.info(f"Сбор данных для {symbol}")
            
            # Создание синтетических данных (замените на реальные)
            dates = pd.date_range(
                start=self.config["data"]["start_date"],
                end=self.config["data"]["end_date"],
                freq='H'
            )
            
            # Симуляция цен (случайное блуждание)
            np.random.seed(42)
            returns = np.random.normal(0, 0.001, len(dates))
            prices = 1.1000 + np.cumsum(returns)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': prices * (1 + np.abs(np.random.normal(0, 0.0005, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.0005, len(dates)))),
                'close': prices,
                'volume': np.random.randint(1000, 10000, len(dates))
            })
            
            df.set_index('timestamp', inplace=True)
            data[symbol] = df
        
        logger.info(f"Собрано данных для {len(data)} символов")
        return data
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков для ML"""
        logger.info("Создание признаков...")
        
        features_df = df.copy()
        
        if self.config["features"]["technical_indicators"]:
            # Технические индикаторы
            features_df['sma_10'] = df['close'].rolling(10).mean()
            features_df['sma_20'] = df['close'].rolling(20).mean()
            features_df['sma_50'] = df['close'].rolling(50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features_df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            features_df['macd'] = ema_12 - ema_26
            features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            features_df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            features_df['bb_upper'] = features_df['bb_middle'] + (bb_std * 2)
            features_df['bb_lower'] = features_df['bb_middle'] - (bb_std * 2)
            features_df['bb_width'] = features_df['bb_upper'] - features_df['bb_lower']
            features_df['bb_position'] = (df['close'] - features_df['bb_lower']) / features_df['bb_width']
        
        if self.config["features"]["price_features"]:
            # Ценовые признаки
            features_df['returns'] = df['close'].pct_change()
            features_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            features_df['volatility'] = features_df['returns'].rolling(20).std()
            features_df['high_low_ratio'] = df['high'] / df['low']
            features_df['open_close_ratio'] = df['open'] / df['close']
        
        if self.config["features"]["volume_features"]:
            # Объемные признаки
            features_df['volume_sma'] = df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = df['volume'] / features_df['volume_sma']
        
        if self.config["features"]["lag_features"]:
            # Лаговые признаки
            for lag in self.config["features"]["lag_features"]:
                features_df[f'close_lag_{lag}'] = df['close'].shift(lag)
                features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
        
        # Скользящие статистики
        for window in self.config["features"]["rolling_features"]:
            features_df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            features_df[f'close_std_{window}'] = df['close'].rolling(window).std()
            features_df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
        
        # Создание целевой переменной (будущая цена)
        features_df['target'] = df['close'].shift(-1)  # Следующая цена
        features_df['target_direction'] = (features_df['target'] > df['close']).astype(int)
        
        # Удаление NaN
        features_df = features_df.dropna()
        
        logger.info(f"Создано {len(features_df.columns)} признаков")
        return features_df
    
    def prepare_lstm_data(self, df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для LSTM"""
        feature_columns = [col for col in df.columns if col not in ['target', 'target_direction']]
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_columns])
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(df['target'].iloc[i])
        
        self.scalers['lstm'] = scaler
        return np.array(X), np.array(y)
    
    def optimize_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBRegressor:
        """Оптимизация XGBoost с Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)], 
                     early_stopping_rounds=50, 
                     verbose=False)
            
            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config["models"]["xgboost"]["trials"])
        
        best_model = xgb.XGBRegressor(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train)
        
        logger.info(f"XGBoost оптимизирован. Лучший MSE: {study.best_value:.6f}")
        return best_model
    
    def optimize_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> lgb.LGBMRegressor:
        """Оптимизация LightGBM с Optuna"""
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'mse',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'random_state': 42,
                'verbosity': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            
            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config["models"]["lightgbm"]["trials"])
        
        best_model = lgb.LGBMRegressor(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train)
        
        logger.info(f"LightGBM оптимизирован. Лучший MSE: {study.best_value:.6f}")
        return best_model
    
    def create_lstm_model(self, input_shape: Tuple, trial=None) -> Model:
        """Создание LSTM модели"""
        if trial:
            # Оптимизация гиперпараметров
            units_1 = trial.suggest_int('units_1', 32, 128)
            units_2 = trial.suggest_int('units_2', 16, 64)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)
        else:
            # Стандартные параметры
            units_1, units_2, dropout, learning_rate = 64, 32, 0.2, 0.001
        
        model = Sequential([
            Bidirectional(LSTM(units_1, return_sequences=True), input_shape=input_shape),
            Dropout(dropout),
            Bidirectional(LSTM(units_2, return_sequences=True)),
            Dropout(dropout),
            Bidirectional(LSTM(units_2//2)),
            Dropout(dropout),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def optimize_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> Model:
        """Оптимизация LSTM с Optuna"""
        
        def objective(trial):
            model = self.create_lstm_model(X_train.shape[1:], trial)
            
            # Callbacks
            pruning_callback = TFKerasPruningCallback(trial, 'val_loss')
            early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
            
            # Обучение
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,  # Меньше эпох для оптимизации
                batch_size=32,
                callbacks=[pruning_callback, early_stopping],
                verbose=0
            )
            
            return min(history.history['val_loss'])
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config["models"]["lstm"]["trials"])
        
        # Создание финальной модели с лучшими параметрами
        best_model = self.create_lstm_model(X_train.shape[1:])
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint('best_lstm_model.h5', save_best_only=True)
        ]
        
        best_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config["models"]["lstm"]["epochs"],
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"LSTM оптимизирован. Лучший val_loss: {study.best_value:.6f}")
        return best_model
    
    async def train_all_models(self, symbol: str = "EURUSD"):
        """Обучение всех моделей для символа"""
        logger.info(f"Начинаем обучение моделей для {symbol}")
        
        # Сбор данных
        data = await self.collect_data()
        df = data[symbol]
        
        # Создание признаков
        features_df = self.create_features(df)
        
        # Подготовка данных для табличных моделей
        feature_columns = [col for col in features_df.columns 
                          if col not in ['target', 'target_direction']]
        X = features_df[feature_columns].values
        y = features_df['target'].values
        
        # Разделение данных (временное)
        split_idx = int(len(X) * (1 - self.config["data"]["test_size"]))
        X_train_full, X_test = X[:split_idx], X[split_idx:]
        y_train_full, y_test = y[:split_idx], y[split_idx:]
        
        # Разделение train на train и validation
        val_size = int(len(X_train_full) * self.config["data"]["validation_size"])
        X_train, X_val = X_train_full[:-val_size], X_train_full[-val_size:]
        y_train, y_val = y_train_full[:-val_size], y_train_full[-val_size:]
        
        # Нормализация данных
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        results = {}
        
        # 1. XGBoost
        if self.config["models"]["xgboost"]["enabled"]:
            logger.info("Обучение XGBoost...")
            if self.config["models"]["xgboost"]["optimize"]:
                xgb_model = self.optimize_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
            else:
                xgb_model = xgb.XGBRegressor(random_state=42)
                xgb_model.fit(X_train_scaled, y_train)
            
            xgb_pred = xgb_model.predict(X_test_scaled)
            xgb_mse = mean_squared_error(y_test, xgb_pred)
            results['xgboost'] = {'model': xgb_model, 'mse': xgb_mse, 'predictions': xgb_pred}
            self.models['xgboost'] = xgb_model
        
        # 2. LightGBM
        if self.config["models"]["lightgbm"]["enabled"]:
            logger.info("Обучение LightGBM...")
            if self.config["models"]["lightgbm"]["optimize"]:
                lgb_model = self.optimize_lightgbm(X_train_scaled, y_train, X_val_scaled, y_val)
            else:
                lgb_model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
                lgb_model.fit(X_train_scaled, y_train)
            
            lgb_pred = lgb_model.predict(X_test_scaled)
            lgb_mse = mean_squared_error(y_test, lgb_pred)
            results['lightgbm'] = {'model': lgb_model, 'mse': lgb_mse, 'predictions': lgb_pred}
            self.models['lightgbm'] = lgb_model
        
        # 3. LSTM
        if self.config["models"]["lstm"]["enabled"]:
            logger.info("Обучение LSTM...")
            seq_length = self.config["models"]["lstm"]["sequence_length"]
            X_lstm, y_lstm = self.prepare_lstm_data(features_df, seq_length)
            
            # Разделение LSTM данных
            lstm_split = int(len(X_lstm) * (1 - self.config["data"]["test_size"]))
            X_lstm_train_full, X_lstm_test = X_lstm[:lstm_split], X_lstm[lstm_split:]
            y_lstm_train_full, y_lstm_test = y_lstm[:lstm_split], y_lstm[lstm_split:]
            
            lstm_val_size = int(len(X_lstm_train_full) * self.config["data"]["validation_size"])
            X_lstm_train = X_lstm_train_full[:-lstm_val_size]
            X_lstm_val = X_lstm_train_full[-lstm_val_size:]
            y_lstm_train = y_lstm_train_full[:-lstm_val_size]
            y_lstm_val = y_lstm_train_full[-lstm_val_size:]
            
            if self.config["models"]["lstm"]["optimize"]:
                lstm_model = self.optimize_lstm(X_lstm_train, y_lstm_train, X_lstm_val, y_lstm_val)
            else:
                lstm_model = self.create_lstm_model(X_lstm_train.shape[1:])
                lstm_model.fit(X_lstm_train, y_lstm_train, 
                             validation_data=(X_lstm_val, y_lstm_val),
                             epochs=50, batch_size=32, verbose=1)
            
            lstm_pred = lstm_model.predict(X_lstm_test).flatten()
            lstm_mse = mean_squared_error(y_lstm_test, lstm_pred)
            results['lstm'] = {'model': lstm_model, 'mse': lstm_mse, 'predictions': lstm_pred}
            self.models['lstm'] = lstm_model
        
        # 4. Ensemble
        if self.config["models"]["ensemble"]["enabled"] and len(results) > 1:
            logger.info("Создание ансамбля...")
            ensemble_pred = self._create_ensemble(results, y_test)
            ensemble_mse = mean_squared_error(y_test, ensemble_pred)
            results['ensemble'] = {'mse': ensemble_mse, 'predictions': ensemble_pred}
        
        # Сохранение результатов
        self.results[symbol] = results
        
        # Вывод результатов
        logger.info("Результаты обучения:")
        for model_name, result in results.items():
            logger.info(f"{model_name}: MSE = {result['mse']:.6f}")
        
        # Сохранение моделей
        if self.config["training"]["save_models"]:
            self._save_models(symbol)
        
        # Построение графиков
        if self.config["training"]["plot_results"]:
            self._plot_results(symbol, y_test, results)
        
        return results
    
    def _create_ensemble(self, results: Dict, y_true: np.ndarray) -> np.ndarray:
        """Создание ансамблевого прогноза"""
        predictions = []
        weights = []
        
        for model_name, result in results.items():
            if 'predictions' in result:
                predictions.append(result['predictions'])
                # Вес обратно пропорционален MSE
                weights.append(1 / result['mse'])
        
        # Нормализация весов
        weights = np.array(weights) / sum(weights)
        
        # Взвешенное среднее
        ensemble_pred = np.average(predictions, weights=weights, axis=0)
        return ensemble_pred
    
    def _save_models(self, symbol: str):
        """Сохранение обученных моделей"""
        save_dir = f"trained_models/{symbol}"
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name == 'lstm':
                model.save(f"{save_dir}/{model_name}_model.h5")
            else:
                with open(f"{save_dir}/{model_name}_model.pkl", 'wb') as f:
                    pickle.dump(model, f)
        
        # Сохранение скейлеров
        with open(f"{save_dir}/scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        logger.info(f"Модели сохранены в {save_dir}")
    
    def _plot_results(self, symbol: str, y_true: np.ndarray, results: Dict):
        """Построение графиков результатов"""
        plt.figure(figsize=(15, 10))
        
        # График предсказаний
        plt.subplot(2, 2, 1)
        plt.plot(y_true, label='Истинные значения', alpha=0.7)
        
        for model_name, result in results.items():
            if 'predictions' in result:
                plt.plot(result['predictions'], label=f'{model_name} (MSE: {result["mse"]:.6f})', alpha=0.7)
        
        plt.title(f'Предсказания моделей - {symbol}')
        plt.legend()
        plt.grid(True)
        
        # График MSE
        plt.subplot(2, 2, 2)
        model_names = [name for name in results.keys() if 'predictions' in results[name]]
        mse_values = [results[name]['mse'] for name in model_names]
        
        plt.bar(model_names, mse_values)
        plt.title('Сравнение MSE моделей')
        plt.ylabel('MSE')
        plt.xticks(rotation=45)
        
        # График остатков
        plt.subplot(2, 2, 3)
        for model_name, result in results.items():
            if 'predictions' in result:
                residuals = y_true - result['predictions']
                plt.scatter(result['predictions'], residuals, alpha=0.5, label=model_name)
        
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('График остатков')
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Остатки')
        plt.legend()
        
        # Распределение остатков
        plt.subplot(2, 2, 4)
        for model_name, result in results.items():
            if 'predictions' in result:
                residuals = y_true - result['predictions']
                plt.hist(residuals, alpha=0.5, label=model_name, bins=30)
        
        plt.title('Распределение остатков')
        plt.xlabel('Остатки')
        plt.ylabel('Частота')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_results_{symbol}.png', dpi=300)
        plt.show()
        
        logger.info(f"Графики сохранены в training_results_{symbol}.png")


async def main():
    """Главная функция для запуска обучения"""
    trainer = ForexModelTrainer()
    
    # Обучение моделей
    results = await trainer.train_all_models("EURUSD")
    
    logger.info("Обучение завершено!")


if __name__ == "__main__":
    asyncio.run(main())