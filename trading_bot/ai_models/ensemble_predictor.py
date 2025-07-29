"""
Ensemble Predictor - система ансамблевых моделей для прогнозирования цен
Включает LSTM, XGBoost, LightGBM, Prophet и другие модели
"""

import asyncio
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet

# TensorFlow и Keras для нейронных сетей
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from loguru import logger

warnings.filterwarnings('ignore')


class LSTMModel:
    """LSTM модель для временных рядов"""
    
    def __init__(self, sequence_length: int = 60, features: int = 10):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
    
    def create_model(self):
        """Создание архитектуры LSTM"""
        model = Sequential([
            Bidirectional(LSTM(50, return_sequences=True), 
                         input_shape=(self.sequence_length, self.features)),
            Dropout(0.2),
            Bidirectional(LSTM(50, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(25)),
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
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для LSTM"""
        # Выбираем основные признаки
        feature_columns = [
            'open', 'high', 'low', 'close', 'tick_volume',
            'sma_20', 'sma_50', 'rsi', 'macd', 'atr'
        ]
        
        # Проверяем наличие колонок
        available_features = [col for col in feature_columns if col in df.columns]
        
        if len(available_features) < 5:
            logger.warning(f"Недостаточно признаков для LSTM: {len(available_features)}")
            return None, None
        
        data = df[available_features].values
        
        # Нормализация данных
        scaled_data = self.scaler.fit_transform(data)
        
        # Создание последовательностей
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 3])  # close price
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame) -> bool:
        """Обучение LSTM модели"""
        try:
            X, y = self.prepare_data(df)
            if X is None:
                return False
            
            # Разделение на train/validation
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Создание модели
            self.model = self.create_model()
            
            # Callbacks
            early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.0001)
            
            # Обучение
            history = self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            self.is_trained = True
            logger.info("LSTM модель обучена успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения LSTM: {e}")
            return False
    
    def predict(self, df: pd.DataFrame, steps: int = 1) -> Optional[np.ndarray]:
        """Прогнозирование с помощью LSTM"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            # Подготавливаем данные
            feature_columns = [
                'open', 'high', 'low', 'close', 'tick_volume',
                'sma_20', 'sma_50', 'rsi', 'macd', 'atr'
            ]
            available_features = [col for col in feature_columns if col in df.columns]
            
            if len(available_features) < 5:
                return None
            
            data = df[available_features].tail(self.sequence_length).values
            scaled_data = self.scaler.transform(data)
            
            # Прогнозирование
            predictions = []
            current_sequence = scaled_data.reshape(1, self.sequence_length, -1)
            
            for _ in range(steps):
                pred = self.model.predict(current_sequence, verbose=0)[0, 0]
                predictions.append(pred)
                
                # Обновляем последовательность для следующего прогноза
                new_row = current_sequence[0, -1, :].copy()
                new_row[3] = pred  # обновляем close price
                current_sequence = np.append(
                    current_sequence[:, 1:, :], 
                    new_row.reshape(1, 1, -1), 
                    axis=1
                )
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Ошибка прогнозирования LSTM: {e}")
            return None


class XGBoostModel:
    """XGBoost модель для прогнозирования"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков для XGBoost"""
        features_df = df.copy()
        
        # Лаговые признаки
        for lag in [1, 2, 3, 5, 10]:
            features_df[f'close_lag_{lag}'] = df['close'].shift(lag)
            features_df[f'volume_lag_{lag}'] = df['tick_volume'].shift(lag)
        
        # Технические индикаторы (уже есть в df)
        # Добавим дополнительные
        
        # Волатильность
        features_df['volatility'] = df['close'].rolling(20).std()
        
        # Ценовые изменения
        features_df['price_change'] = df['close'].pct_change()
        features_df['price_change_lag1'] = features_df['price_change'].shift(1)
        
        # Относительная позиция цены
        features_df['price_position'] = (df['close'] - df['close'].rolling(20).min()) / \
                                       (df['close'].rolling(20).max() - df['close'].rolling(20).min())
        
        # Удаляем NaN
        features_df = features_df.dropna()
        
        return features_df
    
    def train(self, df: pd.DataFrame) -> bool:
        """Обучение XGBoost модели"""
        try:
            # Создание признаков
            features_df = self.create_features(df)
            
            # Подготовка данных
            target = features_df['close'].shift(-1).dropna()  # следующая цена
            features_df = features_df[:-1]  # убираем последнюю строку
            
            # Выбираем численные колонки
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            X = features_df[numeric_columns]
            
            # Удаляем target из признаков
            if 'close' in X.columns:
                X = X.drop('close', axis=1)
            
            self.feature_names = X.columns.tolist()
            
            # Нормализация
            X_scaled = self.scaler.fit_transform(X)
            
            # Разделение на train/test
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = target[:split_idx], target[split_idx:]
            
            # Обучение модели
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Оценка качества
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            self.is_trained = True
            logger.info(f"XGBoost модель обучена. MSE: {mse:.6f}, MAE: {mae:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения XGBoost: {e}")
            return False
    
    def predict(self, df: pd.DataFrame, steps: int = 1) -> Optional[np.ndarray]:
        """Прогнозирование с помощью XGBoost"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            features_df = self.create_features(df)
            
            # Берем последние данные
            X = features_df[self.feature_names].tail(1)
            X_scaled = self.scaler.transform(X)
            
            # Прогнозирование
            predictions = []
            for _ in range(steps):
                pred = self.model.predict(X_scaled)[0]
                predictions.append(pred)
                
                # Для многошагового прогноза нужно обновить признаки
                # Упрощенная версия - повторяем последний прогноз
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Ошибка прогнозирования XGBoost: {e}")
            return None


class EnsemblePredictor:
    """Ансамблевый предиктор, объединяющий несколько моделей"""
    
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.weights = {}
        self.last_retrain = None
        self.retrain_interval = config.get("retrain_interval", 24)  # часы
        
        # Инициализация моделей
        if "lstm" in config.get("models", []):
            self.models["lstm"] = LSTMModel()
            self.weights["lstm"] = 0.4
        
        if "xgboost" in config.get("models", []):
            self.models["xgboost"] = XGBoostModel()
            self.weights["xgboost"] = 0.3
        
        if "lightgbm" in config.get("models", []):
            self.models["lightgbm"] = self._create_lightgbm_model()
            self.weights["lightgbm"] = 0.3
    
    def _create_lightgbm_model(self):
        """Создание LightGBM модели"""
        # Упрощенная версия - можно расширить аналогично XGBoost
        return lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    
    async def load_models(self):
        """Загрузка предобученных моделей"""
        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name in self.models.keys():
            model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
            
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    logger.info(f"Модель {model_name} загружена")
                except Exception as e:
                    logger.error(f"Ошибка загрузки модели {model_name}: {e}")
            else:
                logger.info(f"Модель {model_name} не найдена, требуется обучение")
    
    async def save_models(self):
        """Сохранение обученных моделей"""
        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if hasattr(model, 'is_trained') and model.is_trained:
                model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
                try:
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    logger.info(f"Модель {model_name} сохранена")
                except Exception as e:
                    logger.error(f"Ошибка сохранения модели {model_name}: {e}")
    
    async def train_models(self, market_data: Dict[str, pd.DataFrame]):
        """Обучение всех моделей"""
        logger.info("Начинаем обучение моделей ИИ")
        
        for symbol, df in market_data.items():
            if len(df) < 200:  # Недостаточно данных
                logger.warning(f"Недостаточно данных для {symbol}: {len(df)}")
                continue
            
            logger.info(f"Обучение моделей для {symbol}")
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'train'):
                        success = model.train(df)
                        if success:
                            logger.info(f"Модель {model_name} для {symbol} обучена")
                        else:
                            logger.error(f"Ошибка обучения {model_name} для {symbol}")
                except Exception as e:
                    logger.error(f"Исключение при обучении {model_name}: {e}")
        
        self.last_retrain = datetime.now()
        await self.save_models()
        
        logger.info("Обучение моделей завершено")
    
    async def predict(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Получение прогнозов от ансамбля моделей"""
        predictions = {}
        
        # Проверяем, нужно ли переобучение
        if self._need_retrain():
            await self.train_models(market_data)
        
        for symbol, df in market_data.items():
            symbol_predictions = {}
            
            # Получаем прогнозы от каждой модели
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict(df, steps=1)
                        if pred is not None and len(pred) > 0:
                            symbol_predictions[model_name] = float(pred[0])
                except Exception as e:
                    logger.error(f"Ошибка прогноза {model_name} для {symbol}: {e}")
            
            # Вычисляем ансамблевый прогноз
            if symbol_predictions:
                ensemble_pred = self._calculate_ensemble_prediction(symbol_predictions)
                
                # Определяем направление тренда
                current_price = df['close'].iloc[-1]
                trend_direction = "buy" if ensemble_pred > current_price else "sell"
                confidence = abs(ensemble_pred - current_price) / current_price
                
                predictions[symbol] = {
                    "predicted_price": ensemble_pred,
                    "current_price": current_price,
                    "trend_direction": trend_direction,
                    "confidence": min(confidence * 100, 100),  # в процентах
                    "individual_predictions": symbol_predictions
                }
        
        return predictions
    
    def _calculate_ensemble_prediction(self, predictions: Dict[str, float]) -> float:
        """Вычисление ансамблевого прогноза с весами"""
        weighted_sum = 0
        total_weight = 0
        
        for model_name, prediction in predictions.items():
            weight = self.weights.get(model_name, 1.0)
            weighted_sum += prediction * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _need_retrain(self) -> bool:
        """Проверка необходимости переобучения"""
        if self.last_retrain is None:
            return True
        
        time_since_retrain = datetime.now() - self.last_retrain
        return time_since_retrain.total_seconds() > (self.retrain_interval * 3600)
    
    async def get_status(self) -> Dict:
        """Получение статуса моделей"""
        status = {
            "models_count": len(self.models),
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "next_retrain": (self.last_retrain + timedelta(hours=self.retrain_interval)).isoformat() 
                           if self.last_retrain else None,
            "models_status": {}
        }
        
        for model_name, model in self.models.items():
            status["models_status"][model_name] = {
                "trained": hasattr(model, 'is_trained') and model.is_trained,
                "weight": self.weights.get(model_name, 1.0)
            }
        
        return status