#!/usr/bin/env python3
"""
Advanced AI Models for Professional Trading
Продвинутые AI модели для профессиональной торговли
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Продвинутый инженер признаков"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание технических индикаторов"""
        # Базовые индикаторы
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # EMA
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(periods=5)
        df['price_acceleration'] = df['price_momentum'].diff()
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['close']
        
        # Support and Resistance
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()
        df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        
        return df
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание продвинутых признаков"""
        # Fractal features
        df['fractal_high'] = self._fractal_high(df)
        df['fractal_low'] = self._fractal_low(df)
        
        # Market structure
        df['higher_high'] = self._higher_high(df)
        df['lower_low'] = self._lower_low(df)
        
        # Divergence indicators
        df['price_rsi_divergence'] = self._calculate_divergence(df, 'close', 'rsi')
        
        # Multi-timeframe features
        df['mtf_trend'] = self._multi_timeframe_trend(df)
        
        # Market regime detection
        df['market_regime'] = self._detect_market_regime(df)
        
        # Volatility regime
        df['volatility_regime'] = self._detect_volatility_regime(df)
        
        return df
        
    def _fractal_high(self, df: pd.DataFrame) -> pd.Series:
        """Определение фрактальных максимумов"""
        fractal_high = pd.Series(0, index=df.index)
        for i in range(2, len(df) - 2):
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                fractal_high.iloc[i] = 1
        return fractal_high
        
    def _fractal_low(self, df: pd.DataFrame) -> pd.Series:
        """Определение фрактальных минимумов"""
        fractal_low = pd.Series(0, index=df.index)
        for i in range(2, len(df) - 2):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                fractal_low.iloc[i] = 1
        return fractal_low
        
    def _higher_high(self, df: pd.DataFrame) -> pd.Series:
        """Определение более высоких максимумов"""
        higher_high = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['high'].iloc[i] > df['high'].iloc[i-1]:
                higher_high.iloc[i] = 1
        return higher_high
        
    def _lower_low(self, df: pd.DataFrame) -> pd.Series:
        """Определение более низких минимумов"""
        lower_low = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if df['low'].iloc[i] < df['low'].iloc[i-1]:
                lower_low.iloc[i] = 1
        return lower_low
        
    def _calculate_divergence(self, df: pd.DataFrame, price_col: str, indicator_col: str) -> pd.Series:
        """Расчет дивергенции"""
        divergence = pd.Series(0, index=df.index)
        
        # Простая дивергенция
        for i in range(20, len(df)):
            price_trend = df[price_col].iloc[i] - df[price_col].iloc[i-20]
            indicator_trend = df[indicator_col].iloc[i] - df[indicator_col].iloc[i-20]
            
            if price_trend > 0 and indicator_trend < 0:
                divergence.iloc[i] = -1  # Bearish divergence
            elif price_trend < 0 and indicator_trend > 0:
                divergence.iloc[i] = 1   # Bullish divergence
                
        return divergence
        
    def _multi_timeframe_trend(self, df: pd.DataFrame) -> pd.Series:
        """Мультитаймфреймный тренд"""
        # Комбинация трендов разных таймфреймов
        trend_5 = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        trend_15 = np.where(df['sma_50'] > df['sma_200'], 1, -1)
        
        # Взвешенная комбинация
        mtf_trend = (trend_5 * 0.6 + trend_15 * 0.4)
        return pd.Series(mtf_trend, index=df.index)
        
    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Определение режима рынка"""
        # Трендовый vs боковой рынок
        volatility = df['close'].rolling(window=20).std()
        trend_strength = abs(df['sma_20'] - df['sma_200']) / df['sma_200']
        
        regime = pd.Series('sideways', index=df.index)
        regime[(trend_strength > 0.02) & (volatility < volatility.quantile(0.7))] = 'trending'
        regime[(volatility > volatility.quantile(0.8))] = 'volatile'
        
        return regime
        
    def _detect_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Определение режима волатильности"""
        atr = df['atr']
        atr_ma = atr.rolling(window=20).mean()
        
        regime = pd.Series('normal', index=df.index)
        regime[atr > atr_ma * 1.5] = 'high'
        regime[atr < atr_ma * 0.5] = 'low'
        
        return regime
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Подготовка признаков для модели"""
        # Создание всех индикаторов
        df = self.create_technical_indicators(df)
        df = self.create_advanced_features(df)
        
        # Выбор признаков
        feature_columns = [
            'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_histogram', 'rsi',
            'bb_position', 'bb_width', 'stoch_k', 'stoch_d',
            'atr', 'volume_ratio', 'price_momentum', 'price_acceleration',
            'volatility_ratio', 'price_position', 'fractal_high', 'fractal_low',
            'higher_high', 'lower_low', 'price_rsi_divergence',
            'mtf_trend', 'market_regime', 'volatility_regime'
        ]
        
        # Кодирование категориальных признаков
        df['market_regime_code'] = df['market_regime'].map({'trending': 1, 'sideways': 0, 'volatile': 2})
        df['volatility_regime_code'] = df['volatility_regime'].map({'low': 0, 'normal': 1, 'high': 2})
        
        feature_columns.extend(['market_regime_code', 'volatility_regime_code'])
        
        # Удаление NaN значений
        df = df.dropna()
        
        # Нормализация
        features = df[feature_columns].values
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled, feature_columns

class AdvancedLSTMModel:
    """Продвинутая LSTM модель"""
    
    def __init__(self, sequence_length: int = 60, n_features: int = 30):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_model(self) -> Model:
        """Создание продвинутой LSTM модели"""
        model = Sequential([
            # Первый LSTM слой с возвратом последовательности
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            
            # Второй LSTM слой
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            
            # Третий LSTM слой
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Dense слои
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')  # 3 класса: BUY, SELL, HOLD
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def prepare_sequences(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка последовательностей для LSTM"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(labels[i])
            
        return np.array(X), np.array(y)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Обучение модели"""
        self.model = self.create_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание"""
        return self.model.predict(X)
        
    def save_model(self, filepath: str):
        """Сохранение модели"""
        self.model.save(filepath)
        
    def load_model(self, filepath: str):
        """Загрузка модели"""
        self.model = tf.keras.models.load_model(filepath)

class AdvancedTransformerModel:
    """Продвинутая Transformer модель"""
    
    def __init__(self, sequence_length: int = 60, n_features: int = 30, n_heads: int = 8):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_heads = n_heads
        self.model = None
        
    def create_transformer_block(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Создание Transformer блока"""
        # Multi-head attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size, dropout=dropout
        )(inputs, inputs)
        attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
        attention_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(attention_output)
        ffn_output = tf.keras.layers.Dense(self.n_features)(ffn_output)
        ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)
        ffn_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
        
        return ffn_output
        
    def create_model(self) -> Model:
        """Создание Transformer модели"""
        inputs = tf.keras.layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Positional encoding
        pos_encoding = self._positional_encoding(self.sequence_length, self.n_features)
        x = inputs + pos_encoding
        
        # Transformer blocks
        for _ in range(4):  # 4 transformer blocks
            x = self.create_transformer_block(x, 64, self.n_heads, 128, dropout=0.1)
            
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def _positional_encoding(self, position, d_model):
        """Positional encoding для Transformer"""
        angle_rads = self._get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
        
    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Обучение модели"""
        self.model = self.create_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history

class AdvancedEnsembleModel:
    """Продвинутый ансамбль моделей"""
    
    def __init__(self):
        self.models = {}
        self.feature_engineer = AdvancedFeatureEngineer()
        self.scaler = StandardScaler()
        
    def add_model(self, name: str, model):
        """Добавление модели в ансамбль"""
        self.models[name] = model
        
    def train_models(self, df: pd.DataFrame, target: pd.Series):
        """Обучение всех моделей"""
        # Подготовка данных
        features, feature_names = self.feature_engineer.prepare_features(df)
        
        # Создание целевой переменной
        y = self._create_target_variable(target)
        
        # Разделение данных
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Обучение LSTM
        if 'lstm' in self.models:
            X_lstm_train, y_lstm_train = self.models['lstm'].prepare_sequences(X_train, y_train)
            X_lstm_test, y_lstm_test = self.models['lstm'].prepare_sequences(X_test, y_test)
            self.models['lstm'].train(X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test)
            
        # Обучение Transformer
        if 'transformer' in self.models:
            X_transformer_train, y_transformer_train = self.models['transformer'].prepare_sequences(X_train, y_train)
            X_transformer_test, y_transformer_test = self.models['transformer'].prepare_sequences(X_test, y_test)
            self.models['transformer'].train(X_transformer_train, y_transformer_train, X_transformer_test, y_transformer_test)
            
        # Обучение XGBoost
        if 'xgboost' in self.models:
            self.models['xgboost'].fit(X_train, y_train)
            
        # Обучение LightGBM
        if 'lightgbm' in self.models:
            self.models['lightgbm'].fit(X_train, y_train)
            
        # Обучение Random Forest
        if 'random_forest' in self.models:
            self.models['random_forest'].fit(X_train, y_train)
            
        # Обучение Gradient Boosting
        if 'gradient_boosting' in self.models:
            self.models['gradient_boosting'].fit(X_train, y_train)
            
    def predict_ensemble(self, df: pd.DataFrame) -> Dict:
        """Предсказание ансамбля"""
        features, _ = self.feature_engineer.prepare_features(df)
        
        predictions = {}
        
        # Получение предсказаний от каждой модели
        for name, model in self.models.items():
            if name == 'lstm':
                X_seq = self.models['lstm'].prepare_sequences(features, np.zeros(len(features)))
                pred = model.predict(X_seq)
                predictions[name] = pred
                
            elif name == 'transformer':
                X_seq = self.models['transformer'].prepare_sequences(features, np.zeros(len(features)))
                pred = model.predict(X_seq)
                predictions[name] = pred
                
            else:
                pred = model.predict_proba(features)
                predictions[name] = pred
                
        # Взвешенное голосование
        ensemble_prediction = self._weighted_voting(predictions)
        
        return {
            'ensemble': ensemble_prediction,
            'individual': predictions
        }
        
    def _create_target_variable(self, price_series: pd.Series) -> np.ndarray:
        """Создание целевой переменной"""
        # Процентное изменение цены
        returns = price_series.pct_change()
        
        # Создание классов
        y = np.zeros(len(returns))
        y[returns > 0.001] = 1  # BUY
        y[returns < -0.001] = 2  # SELL
        # 0 = HOLD
        
        return y
        
    def _weighted_voting(self, predictions: Dict) -> np.ndarray:
        """Взвешенное голосование"""
        weights = {
            'lstm': 0.25,
            'transformer': 0.25,
            'xgboost': 0.20,
            'lightgbm': 0.15,
            'random_forest': 0.10,
            'gradient_boosting': 0.05
        }
        
        ensemble_pred = np.zeros((len(list(predictions.values())[0]), 3))
        
        for name, pred in predictions.items():
            if name in weights:
                ensemble_pred += weights[name] * pred
                
        return ensemble_pred
        
    def save_models(self, directory: str):
        """Сохранение всех моделей"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            if name in ['lstm', 'transformer']:
                model.save_model(f"{directory}/{name}_model.h5")
            else:
                joblib.dump(model, f"{directory}/{name}_model.pkl")
                
        # Сохранение feature engineer
        joblib.dump(self.feature_engineer, f"{directory}/feature_engineer.pkl")
        
    def load_models(self, directory: str):
        """Загрузка всех моделей"""
        import os
        
        for name in self.models.keys():
            if name in ['lstm', 'transformer']:
                self.models[name].load_model(f"{directory}/{name}_model.h5")
            else:
                self.models[name] = joblib.load(f"{directory}/{name}_model.pkl")
                
        # Загрузка feature engineer
        self.feature_engineer = joblib.load(f"{directory}/feature_engineer.pkl")

# Создание и настройка моделей
def create_advanced_models():
    """Создание продвинутых моделей"""
    ensemble = AdvancedEnsembleModel()
    
    # LSTM модель
    lstm_model = AdvancedLSTMModel(sequence_length=60, n_features=30)
    ensemble.add_model('lstm', lstm_model)
    
    # Transformer модель
    transformer_model = AdvancedTransformerModel(sequence_length=60, n_features=30)
    ensemble.add_model('transformer', transformer_model)
    
    # XGBoost модель
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    ensemble.add_model('xgboost', xgb_model)
    
    # LightGBM модель
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    ensemble.add_model('lightgbm', lgb_model)
    
    # Random Forest модель
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    ensemble.add_model('random_forest', rf_model)
    
    # Gradient Boosting модель
    gb_model = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        random_state=42
    )
    ensemble.add_model('gradient_boosting', gb_model)
    
    return ensemble

# Пример использования
if __name__ == "__main__":
    # Создание тестовых данных
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='1H')
    np.random.seed(42)
    
    # Симуляция рыночных данных
    price = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
    volume = np.random.randint(1000, 10000, len(dates))
    
    df = pd.DataFrame({
        'open': price * (1 + np.random.randn(len(dates)) * 0.001),
        'high': price * (1 + abs(np.random.randn(len(dates)) * 0.002)),
        'low': price * (1 - abs(np.random.randn(len(dates)) * 0.002)),
        'close': price,
        'volume': volume
    }, index=dates)
    
    # Создание и обучение моделей
    ensemble = create_advanced_models()
    ensemble.train_models(df, df['close'])
    
    # Предсказание
    predictions = ensemble.predict_ensemble(df.tail(100))
    
    print("=== ПРОДВИНУТЫЕ AI МОДЕЛИ ===")
    print(f"Ансамбль предсказаний: {predictions['ensemble'].shape}")
    print(f"Индивидуальные предсказания: {len(predictions['individual'])} моделей")
    
    # Сохранение моделей
    ensemble.save_models('models/')
    print("Модели сохранены в директории 'models/'")