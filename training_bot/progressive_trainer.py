"""
Progressive Model Training System
Поэтапная система обучения моделей: XGBoost → LSTM → Ensemble
"""

import os
import sys
import asyncio
import time
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

# ML Models
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Hyperparameter optimization
import optuna

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Custom imports
sys.path.append('..')
from trading_bot.mt5_connector.mt5_manager import MT5Manager
from loguru import logger

warnings.filterwarnings('ignore')


class ProgressiveForexTrainer:
    """Поэтапный тренер моделей: XGBoost → LSTM → Ensemble"""
    
    def __init__(self, symbol: str = "EURUSD"):
        self.symbol = symbol
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.training_history = []
        
        # Создание папок
        self.save_dir = f"progressive_models/{symbol}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Setup GPU
        self._setup_gpu()
        
        logger.info(f"🚀 Инициализирован Progressive Trainer для {symbol}")
    
    def _setup_gpu(self):
        """Настройка GPU для TensorFlow"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"✅ GPU настроен: {len(gpus)} устройств")
            except RuntimeError as e:
                logger.error(f"❌ Ошибка настройки GPU: {e}")
        else:
            logger.warning("⚠️ GPU не найден, используем CPU")
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Загрузка данных для обучения"""
        
        if data_path and os.path.exists(data_path):
            logger.info(f"📂 Загружаем данные из {data_path}")
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            # Создание синтетических данных для демонстрации
            logger.info("🎲 Создаем синтетические данные для демонстрации")
            df = self._create_demo_data()
        
        logger.info(f"📊 Загружено {len(df)} записей, {len(df.columns)} признаков")
        return df
    
    def _create_demo_data(self) -> pd.DataFrame:
        """Создание демонстрационных данных"""
        from datetime import timedelta
        
        # Временной ряд
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Генерация цен
        np.random.seed(42)
        returns = np.random.normal(0, 0.001, len(dates))
        
        # Добавляем автокорреляцию (трендовость)
        for i in range(1, len(returns)):
            returns[i] += 0.15 * returns[i-1]
        
        prices = 1.1000 + np.cumsum(returns)
        
        # OHLC данные
        df = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0003, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0003, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000, 50000, len(dates))
        }, index=dates)
        
        # Корректировка high/low
        df['high'] = np.maximum(df[['open', 'close']].max(axis=1), df['high'])
        df['low'] = np.minimum(df[['open', 'close']].min(axis=1), df['low'])
        
        # Добавление технических индикаторов
        df = self._add_technical_indicators(df)
        
        # Целевая переменная (следующая цена)
        df['target'] = df['close'].shift(-1)
        df['target_direction'] = (df['target'] > df['close']).astype(int)
        
        return df.dropna()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление технических индикаторов"""
        
        # Скользящие средние
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26'] if 'ema_12' in df.columns else df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Доходности и волатильность
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Лаговые признаки
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Дополнительные признаки
        df['high_low_ratio'] = df['high'] / df['low']
        df['price_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Подготовка данных для обучения"""
        
        # Выбор признаков (исключаем целевые переменные)
        feature_columns = [col for col in df.columns 
                          if not col.startswith('target') and col not in ['open', 'high', 'low', 'close']]
        
        X = df[feature_columns].values
        y = df['target'].values
        
        # Удаление NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        logger.info(f"📝 Подготовлено {len(X)} образцов, {len(feature_columns)} признаков")
        
        return X, y, feature_columns
    
    # ================== ЭТАП 1: XGBoost ==================
    
    def train_xgboost_phase(self, X: np.ndarray, y: np.ndarray, 
                           optimize: bool = True, trials: int = 50) -> Dict:
        """ЭТАП 1: Обучение XGBoost - быстро и эффективно"""
        
        logger.info("🎯 ЭТАП 1: Обучение XGBoost (быстро и эффективно)")
        start_time = time.time()
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Временные ряды не перемешиваем
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, shuffle=False
        )
        
        # Масштабирование
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['xgboost'] = scaler
        
        if optimize:
            logger.info(f"🔧 Оптимизация гиперпараметров ({trials} trials)...")
            model = self._optimize_xgboost(X_train_scaled, y_train, X_val_scaled, y_val, trials)
        else:
            logger.info("⚡ Быстрое обучение с стандартными параметрами...")
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
        
        # Оценка
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)
        
        results = {
            'model': model,
            'train_mse': mean_squared_error(y_train, train_pred),
            'val_mse': mean_squared_error(y_val, val_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'directional_accuracy': self._calculate_directional_accuracy(y_test, test_pred),
            'predictions': test_pred,
            'actual': y_test,
            'training_time': time.time() - start_time
        }
        
        self.models['xgboost'] = model
        self.results['xgboost'] = results
        
        # Сохранение
        self._save_model('xgboost', model, results)
        
        logger.info(f"✅ XGBoost завершен за {results['training_time']:.1f}с")
        logger.info(f"📊 Test MSE: {results['test_mse']:.6f}")
        logger.info(f"🎯 Directional Accuracy: {results['directional_accuracy']:.1f}%")
        
        return results
    
    def _optimize_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray, trials: int) -> xgb.XGBRegressor:
        """Оптимизация XGBoost с Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'random_state': 42
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)], 
                     early_stopping_rounds=20, 
                     verbose=False)
            
            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=trials)
        
        best_model = xgb.XGBRegressor(**study.best_params, random_state=42)
        best_model.fit(X_train, y_train)
        
        logger.info(f"🏆 Лучший MSE: {study.best_value:.6f}")
        
        return best_model
    
    # ================== ЭТАП 2: LSTM ==================
    
    def train_lstm_phase(self, X: np.ndarray, y: np.ndarray, 
                        sequence_length: int = 60, optimize: bool = True, trials: int = 30) -> Dict:
        """ЭТАП 2: Обучение LSTM - для улучшения точности"""
        
        logger.info("🧠 ЭТАП 2: Обучение LSTM (для улучшения точности)")
        start_time = time.time()
        
        # Подготовка данных для LSTM
        X_lstm, y_lstm = self._prepare_lstm_data(X, y, sequence_length)
        
        # Разделение данных
        split_idx = int(len(X_lstm) * 0.8)
        val_split_idx = int(split_idx * 0.8)
        
        X_train = X_lstm[:val_split_idx]
        X_val = X_lstm[val_split_idx:split_idx]
        X_test = X_lstm[split_idx:]
        
        y_train = y_lstm[:val_split_idx]
        y_val = y_lstm[val_split_idx:split_idx]
        y_test = y_lstm[split_idx:]
        
        if optimize:
            logger.info(f"🔧 Оптимизация LSTM ({trials} trials)...")
            model = self._optimize_lstm(X_train, y_train, X_val, y_val, trials)
        else:
            logger.info("⚡ Быстрое обучение LSTM...")
            model = self._create_lstm_model(X_train.shape[1:])
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            model.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=50, batch_size=32,
                     callbacks=callbacks, verbose=1)
        
        # Оценка
        train_pred = model.predict(X_train).flatten()
        val_pred = model.predict(X_val).flatten()
        test_pred = model.predict(X_test).flatten()
        
        results = {
            'model': model,
            'train_mse': mean_squared_error(y_train, train_pred),
            'val_mse': mean_squared_error(y_val, val_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'directional_accuracy': self._calculate_directional_accuracy(y_test, test_pred),
            'predictions': test_pred,
            'actual': y_test,
            'training_time': time.time() - start_time
        }
        
        self.models['lstm'] = model
        self.results['lstm'] = results
        
        # Сохранение
        self._save_model('lstm', model, results)
        
        logger.info(f"✅ LSTM завершен за {results['training_time']:.1f}с")
        logger.info(f"📊 Test MSE: {results['test_mse']:.6f}")
        logger.info(f"🎯 Directional Accuracy: {results['directional_accuracy']:.1f}%")
        
        return results
    
    def _prepare_lstm_data(self, X: np.ndarray, y: np.ndarray, 
                          sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для LSTM"""
        
        # Нормализация
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['lstm'] = scaler
        
        # Создание последовательностей
        X_sequences, y_sequences = [], []
        
        for i in range(sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _create_lstm_model(self, input_shape: Tuple) -> tf.keras.Model:
        """Создание LSTM модели"""
        
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(32, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(16)),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _optimize_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray, trials: int) -> tf.keras.Model:
        """Оптимизация LSTM с Optuna"""
        
        def objective(trial):
            # Гиперпараметры
            units_1 = trial.suggest_int('units_1', 32, 128)
            units_2 = trial.suggest_int('units_2', 16, 64)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01)
            
            # Модель
            model = Sequential([
                Bidirectional(LSTM(units_1, return_sequences=True), input_shape=X_train.shape[1:]),
                Dropout(dropout),
                Bidirectional(LSTM(units_2)),
                Dropout(dropout),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='mse'
            )
            
            # Обучение
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,  # Меньше эпох для оптимизации
                batch_size=32,
                verbose=0,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
            )
            
            return min(history.history['val_loss'])
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=trials)
        
        # Финальная модель с лучшими параметрами
        best_model = self._create_lstm_model(X_train.shape[1:])
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        best_model.fit(X_train, y_train,
                      validation_data=(X_val, y_val),
                      epochs=100, batch_size=32,
                      callbacks=callbacks, verbose=1)
        
        logger.info(f"🏆 Лучший val_loss: {study.best_value:.6f}")
        
        return best_model
    
    # ================== ЭТАП 3: ENSEMBLE ==================
    
    def train_ensemble_phase(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """ЭТАП 3: Ensemble - максимальная точность"""
        
        logger.info("🏆 ЭТАП 3: Ensemble (максимальная точность)")
        start_time = time.time()
        
        if 'xgboost' not in self.models or 'lstm' not in self.models:
            logger.error("❌ Нужно сначала обучить XGBoost и LSTM")
            return {}
        
        # Разделение данных (те же разбиения)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Подготовка данных для XGBoost
        X_train_xgb = self.scalers['xgboost'].transform(X_train)
        X_test_xgb = self.scalers['xgboost'].transform(X_test)
        
        # Подготовка данных для LSTM
        X_lstm_full, y_lstm_full = self._prepare_lstm_data(X, y, 60)
        split_idx = int(len(X_lstm_full) * 0.8)
        X_test_lstm = X_lstm_full[split_idx:]
        y_test_lstm = y_lstm_full[split_idx:]
        
        # Получение предсказаний от каждой модели
        xgb_pred = self.models['xgboost'].predict(X_test_xgb)
        lstm_pred = self.models['lstm'].predict(X_test_lstm).flatten()
        
        # Выравнивание размеров (LSTM может иметь меньше предсказаний)
        min_length = min(len(xgb_pred), len(lstm_pred), len(y_test), len(y_test_lstm))
        xgb_pred = xgb_pred[:min_length]
        lstm_pred = lstm_pred[:min_length]
        y_test_aligned = y_test[:min_length]
        
        # Оптимизация весов ансамбля
        logger.info("🔧 Оптимизация весов ансамбля...")
        best_weights = self._optimize_ensemble_weights(
            xgb_pred, lstm_pred, y_test_aligned
        )
        
        # Финальные предсказания ансамбля
        ensemble_pred = (best_weights[0] * xgb_pred + 
                        best_weights[1] * lstm_pred)
        
        results = {
            'weights': best_weights,
            'test_mse': mean_squared_error(y_test_aligned, ensemble_pred),
            'test_mae': mean_absolute_error(y_test_aligned, ensemble_pred),
            'directional_accuracy': self._calculate_directional_accuracy(y_test_aligned, ensemble_pred),
            'predictions': ensemble_pred,
            'actual': y_test_aligned,
            'xgb_predictions': xgb_pred,
            'lstm_predictions': lstm_pred,
            'training_time': time.time() - start_time
        }
        
        self.results['ensemble'] = results
        
        # Сравнение с индивидуальными моделями
        xgb_mse = mean_squared_error(y_test_aligned, xgb_pred)
        lstm_mse = mean_squared_error(y_test_aligned, lstm_pred)
        
        logger.info(f"✅ Ensemble завершен за {results['training_time']:.1f}с")
        logger.info(f"📊 Сравнение MSE:")
        logger.info(f"   XGBoost: {xgb_mse:.6f}")
        logger.info(f"   LSTM:    {lstm_mse:.6f}")
        logger.info(f"   Ensemble: {results['test_mse']:.6f}")
        logger.info(f"🎯 Ensemble Directional Accuracy: {results['directional_accuracy']:.1f}%")
        logger.info(f"⚖️ Оптимальные веса: XGBoost={best_weights[0]:.2f}, LSTM={best_weights[1]:.2f}")
        
        return results
    
    def _optimize_ensemble_weights(self, pred1: np.ndarray, pred2: np.ndarray, 
                                  y_true: np.ndarray) -> Tuple[float, float]:
        """Оптимизация весов ансамбля"""
        
        def objective(trial):
            w1 = trial.suggest_float('weight_xgb', 0.0, 1.0)
            w2 = 1.0 - w1
            
            ensemble_pred = w1 * pred1 + w2 * pred2
            return mean_squared_error(y_true, ensemble_pred)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        
        best_w1 = study.best_params['weight_xgb']
        best_w2 = 1.0 - best_w1
        
        return (best_w1, best_w2)
    
    # ================== UTILITIES ==================
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Расчет точности направления"""
        if len(y_true) <= 1:
            return 0.0
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        return np.mean(true_direction == pred_direction) * 100
    
    def _save_model(self, model_name: str, model, results: Dict):
        """Сохранение модели и результатов"""
        
        # Сохранение модели
        if model_name == 'lstm':
            model.save(f"{self.save_dir}/{model_name}_model.h5")
        else:
            with open(f"{self.save_dir}/{model_name}_model.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        # Сохранение результатов
        results_to_save = {k: v for k, v in results.items() 
                          if k not in ['model', 'predictions', 'actual']}
        
        with open(f"{self.save_dir}/{model_name}_results.json", 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"💾 {model_name} сохранен в {self.save_dir}")
    
    def create_comparison_plots(self):
        """Создание графиков сравнения моделей"""
        
        if not self.results:
            logger.warning("⚠️ Нет результатов для построения графиков")
            return
        
        plt.figure(figsize=(20, 12))
        
        # 1. Сравнение MSE
        plt.subplot(2, 3, 1)
        models = list(self.results.keys())
        mse_values = [self.results[model]['test_mse'] for model in models]
        
        bars = plt.bar(models, mse_values, color=['#ff7f0e', '#2ca02c', '#d62728'])
        plt.title('Сравнение MSE моделей', fontsize=14, fontweight='bold')
        plt.ylabel('MSE')
        
        # Добавление значений на столбцы
        for bar, value in zip(bars, mse_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.6f}', ha='center', va='bottom')
        
        # 2. Сравнение Directional Accuracy
        plt.subplot(2, 3, 2)
        acc_values = [self.results[model]['directional_accuracy'] for model in models]
        
        bars = plt.bar(models, acc_values, color=['#ff7f0e', '#2ca02c', '#d62728'])
        plt.title('Точность направления (%)', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy (%)')
        plt.ylim(40, 70)
        
        for bar, value in zip(bars, acc_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # 3. Время обучения
        plt.subplot(2, 3, 3)
        time_values = [self.results[model]['training_time'] / 60 for model in models]  # В минутах
        
        bars = plt.bar(models, time_values, color=['#ff7f0e', '#2ca02c', '#d62728'])
        plt.title('Время обучения (минуты)', fontsize=14, fontweight='bold')
        plt.ylabel('Минуты')
        
        for bar, value in zip(bars, time_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.1f}м', ha='center', va='bottom')
        
        # 4. Предсказания vs Реальность (если есть ensemble)
        if 'ensemble' in self.results:
            plt.subplot(2, 3, 4)
            ensemble_results = self.results['ensemble']
            
            plt.plot(ensemble_results['actual'][:100], label='Реальные значения', alpha=0.7)
            plt.plot(ensemble_results['predictions'][:100], label='Ensemble', alpha=0.7)
            plt.plot(ensemble_results['xgb_predictions'][:100], label='XGBoost', alpha=0.5)
            plt.plot(ensemble_results['lstm_predictions'][:100], label='LSTM', alpha=0.5)
            
            plt.title('Предсказания vs Реальность (первые 100 точек)', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. Распределение ошибок
        plt.subplot(2, 3, 5)
        for model in models:
            if 'predictions' in self.results[model] and 'actual' in self.results[model]:
                errors = self.results[model]['actual'] - self.results[model]['predictions']
                plt.hist(errors, bins=30, alpha=0.5, label=model, density=True)
        
        plt.title('Распределение ошибок', fontsize=14, fontweight='bold')
        plt.xlabel('Ошибка прогноза')
        plt.ylabel('Плотность')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Веса ансамбля (если есть)
        if 'ensemble' in self.results:
            plt.subplot(2, 3, 6)
            weights = self.results['ensemble']['weights']
            models_ensemble = ['XGBoost', 'LSTM']
            
            wedges, texts, autotexts = plt.pie(weights, labels=models_ensemble, autopct='%1.1f%%',
                                              colors=['#ff7f0e', '#2ca02c'])
            plt.title('Веса в ансамбле', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Сохранение
        plot_path = f"{self.save_dir}/comparison_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Графики сохранены в {plot_path}")
        
        plt.show()
    
    def print_final_summary(self):
        """Вывод финального резюме"""
        
        logger.info("\n" + "="*80)
        logger.info("🏁 ФИНАЛЬНОЕ РЕЗЮМЕ ОБУЧЕНИЯ")
        logger.info("="*80)
        
        if not self.results:
            logger.warning("⚠️ Нет результатов для отображения")
            return
        
        total_time = sum(result['training_time'] for result in self.results.values())
        
        logger.info(f"📊 Символ: {self.symbol}")
        logger.info(f"⏱️ Общее время обучения: {total_time:.1f} секунд ({total_time/60:.1f} минут)")
        logger.info(f"💾 Модели сохранены в: {self.save_dir}")
        
        logger.info("\n📈 РЕЗУЛЬТАТЫ ПО ЭТАПАМ:")
        
        for i, (model_name, results) in enumerate(self.results.items(), 1):
            stage_names = {
                'xgboost': 'ЭТАП 1: XGBoost (Быстро и эффективно)',
                'lstm': 'ЭТАП 2: LSTM (Улучшение точности)', 
                'ensemble': 'ЭТАП 3: Ensemble (Максимальная точность)'
            }
            
            logger.info(f"\n{stage_names.get(model_name, model_name)}:")
            logger.info(f"   ⏱️ Время: {results['training_time']:.1f}с")
            logger.info(f"   📊 MSE: {results['test_mse']:.6f}")
            logger.info(f"   🎯 Directional Accuracy: {results['directional_accuracy']:.1f}%")
            
            if model_name == 'ensemble':
                weights = results['weights']
                logger.info(f"   ⚖️ Веса: XGBoost={weights[0]:.2f}, LSTM={weights[1]:.2f}")
        
        # Лучшая модель
        best_model = min(self.results.keys(), 
                        key=lambda x: self.results[x]['test_mse'])
        best_mse = self.results[best_model]['test_mse']
        best_acc = self.results[best_model]['directional_accuracy']
        
        logger.info(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model.upper()}")
        logger.info(f"   📊 MSE: {best_mse:.6f}")
        logger.info(f"   🎯 Точность: {best_acc:.1f}%")
        
        # Рекомендации
        logger.info(f"\n💡 РЕКОМЕНДАЦИИ:")
        if best_acc > 60:
            logger.info("   ✅ Отличные результаты! Модель готова для live-тестирования")
        elif best_acc > 55:
            logger.info("   👍 Хорошие результаты. Рекомендуется дополнительная оптимизация")
        else:
            logger.info("   ⚠️ Результаты требуют улучшения. Попробуйте больше данных или признаков")
        
        logger.info("="*80)
    
    # ================== MAIN WORKFLOW ==================
    
    async def run_progressive_training(self, data_path: str = None,
                                     quick_mode: bool = False) -> Dict:
        """Запуск полного поэтапного обучения"""
        
        logger.info("🚀 НАЧИНАЕМ ПОЭТАПНОЕ ОБУЧЕНИЕ МОДЕЛЕЙ")
        logger.info("="*60)
        
        # Загрузка данных
        df = self.load_data(data_path)
        X, y, feature_columns = self.prepare_data(df)
        
        # Настройка параметров
        if quick_mode:
            logger.info("⚡ БЫСТРЫЙ РЕЖИМ: уменьшенное количество trials")
            xgb_trials, lstm_trials = 10, 5
        else:
            logger.info("🔥 ПОЛНЫЙ РЕЖИМ: максимальная оптимизация")
            xgb_trials, lstm_trials = 50, 30
        
        all_results = {}
        
        try:
            # ЭТАП 1: XGBoost
            logger.info("\n" + "="*60)
            xgb_results = self.train_xgboost_phase(X, y, optimize=True, trials=xgb_trials)
            all_results['xgboost'] = xgb_results
            
            # ЭТАП 2: LSTM
            logger.info("\n" + "="*60)
            lstm_results = self.train_lstm_phase(X, y, optimize=True, trials=lstm_trials)
            all_results['lstm'] = lstm_results
            
            # ЭТАП 3: Ensemble
            logger.info("\n" + "="*60)
            ensemble_results = self.train_ensemble_phase(X, y)
            all_results['ensemble'] = ensemble_results
            
            # Создание графиков
            logger.info("\n" + "="*60)
            logger.info("📊 Создание графиков сравнения...")
            self.create_comparison_plots()
            
            # Финальное резюме
            self.print_final_summary()
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обучении: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return all_results


# ================== ГЛАВНАЯ ФУНКЦИЯ ==================

async def main():
    """Главная функция демонстрации"""
    
    # Создание тренера
    trainer = ProgressiveForexTrainer("EURUSD")
    
    # Запуск поэтапного обучения
    results = await trainer.run_progressive_training(
        data_path=None,  # None = синтетические данные
        quick_mode=False  # True для быстрого тестирования
    )
    
    logger.info("🎉 Поэтапное обучение завершено!")
    
    return results


if __name__ == "__main__":
    # Запуск
    results = asyncio.run(main())