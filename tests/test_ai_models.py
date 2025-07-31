#!/usr/bin/env python3
"""
Unit Tests for AI Models
Unit тесты для AI моделей
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Добавление пути к модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_ai_models import (
    AdvancedFeatureEngineer,
    AdvancedLSTMModel,
    AdvancedTransformerModel,
    AdvancedEnsembleModel,
    create_advanced_models
)

class TestAdvancedFeatureEngineer:
    """Тесты для инженера признаков"""
    
    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='1H')
        np.random.seed(42)
        
        price = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
        volume = np.random.randint(1000, 10000, len(dates))
        
        return pd.DataFrame({
            'open': price * (1 + np.random.randn(len(dates)) * 0.001),
            'high': price * (1 + abs(np.random.randn(len(dates)) * 0.002)),
            'low': price * (1 - abs(np.random.randn(len(dates)) * 0.002)),
            'close': price,
            'volume': volume
        }, index=dates)
    
    @pytest.fixture
    def feature_engineer(self):
        """Создание инженера признаков"""
        return AdvancedFeatureEngineer()
    
    def test_create_technical_indicators(self, feature_engineer, sample_data):
        """Тест создания технических индикаторов"""
        result = feature_engineer.create_technical_indicators(sample_data.copy())
        
        # Проверка наличия индикаторов
        assert 'sma_20' in result.columns
        assert 'sma_50' in result.columns
        assert 'sma_200' in result.columns
        assert 'ema_12' in result.columns
        assert 'ema_26' in result.columns
        assert 'macd' in result.columns
        assert 'rsi' in result.columns
        assert 'bb_upper' in result.columns
        assert 'bb_lower' in result.columns
        assert 'stoch_k' in result.columns
        assert 'atr' in result.columns
        
        # Проверка корректности значений
        assert not result['sma_20'].isna().all()
        assert not result['rsi'].isna().all()
        assert (result['rsi'] >= 0).all() and (result['rsi'] <= 100).all()
    
    def test_create_advanced_features(self, feature_engineer, sample_data):
        """Тест создания продвинутых признаков"""
        # Сначала создаем технические индикаторы
        df_with_indicators = feature_engineer.create_technical_indicators(sample_data.copy())
        result = feature_engineer.create_advanced_features(df_with_indicators)
        
        # Проверка наличия продвинутых признаков
        assert 'fractal_high' in result.columns
        assert 'fractal_low' in result.columns
        assert 'higher_high' in result.columns
        assert 'lower_low' in result.columns
        assert 'price_rsi_divergence' in result.columns
        assert 'mtf_trend' in result.columns
        assert 'market_regime' in result.columns
        assert 'volatility_regime' in result.columns
    
    def test_prepare_features(self, feature_engineer, sample_data):
        """Тест подготовки признаков"""
        features, feature_names = feature_engineer.prepare_features(sample_data)
        
        # Проверка корректности результата
        assert isinstance(features, np.ndarray)
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert features.shape[1] == len(feature_names)
        assert not np.isnan(features).any()

class TestAdvancedLSTMModel:
    """Тесты для LSTM модели"""
    
    @pytest.fixture
    def lstm_model(self):
        """Создание LSTM модели"""
        return AdvancedLSTMModel(sequence_length=10, n_features=5)
    
    def test_create_model(self, lstm_model):
        """Тест создания модели"""
        model = lstm_model.create_model()
        
        # Проверка архитектуры
        assert model is not None
        assert len(model.layers) > 0
        assert model.output_shape == (None, 3)  # 3 класса: HOLD, BUY, SELL
    
    def test_prepare_sequences(self, lstm_model):
        """Тест подготовки последовательностей"""
        data = np.random.randn(100, 5)
        labels = np.random.randint(0, 3, 100)
        
        X, y = lstm_model.prepare_sequences(data, labels)
        
        # Проверка размеров
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 10  # sequence_length
        assert X.shape[2] == 5   # n_features
        assert y.shape[1] == 3   # количество классов

class TestAdvancedTransformerModel:
    """Тесты для Transformer модели"""
    
    @pytest.fixture
    def transformer_model(self):
        """Создание Transformer модели"""
        return AdvancedTransformerModel(sequence_length=10, n_features=5)
    
    def test_create_model(self, transformer_model):
        """Тест создания модели"""
        model = transformer_model.create_model()
        
        # Проверка архитектуры
        assert model is not None
        assert len(model.layers) > 0
        assert model.output_shape == (None, 3)  # 3 класса
    
    def test_positional_encoding(self, transformer_model):
        """Тест positional encoding"""
        pos_encoding = transformer_model._positional_encoding(10, 5)
        
        # Проверка размеров
        assert pos_encoding.shape == (1, 10, 5)
        assert not np.isnan(pos_encoding).any()

class TestAdvancedEnsembleModel:
    """Тесты для ансамбля моделей"""
    
    @pytest.fixture
    def ensemble_model(self):
        """Создание ансамбля моделей"""
        return create_advanced_models()
    
    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных"""
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='1H')
        np.random.seed(42)
        
        price = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
        volume = np.random.randint(1000, 10000, len(dates))
        
        return pd.DataFrame({
            'open': price * (1 + np.random.randn(len(dates)) * 0.001),
            'high': price * (1 + abs(np.random.randn(len(dates)) * 0.002)),
            'low': price * (1 - abs(np.random.randn(len(dates)) * 0.002)),
            'close': price,
            'volume': volume
        }, index=dates)
    
    def test_add_model(self, ensemble_model):
        """Тест добавления модели"""
        initial_count = len(ensemble_model.models)
        ensemble_model.add_model('test_model', 'test_object')
        
        assert len(ensemble_model.models) == initial_count + 1
        assert 'test_model' in ensemble_model.models
    
    def test_create_target_variable(self, ensemble_model, sample_data):
        """Тест создания целевой переменной"""
        target = ensemble_model._create_target_variable(sample_data['close'])
        
        # Проверка корректности
        assert isinstance(target, np.ndarray)
        assert len(target) == len(sample_data)
        assert set(np.unique(target)).issubset({0, 1, 2})  # HOLD, BUY, SELL
    
    def test_weighted_voting(self, ensemble_model):
        """Тест взвешенного голосования"""
        predictions = {
            'lstm': np.array([[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]]),
            'xgboost': np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3]]),
            'lightgbm': np.array([[0.5, 0.2, 0.3], [0.4, 0.3, 0.3]])
        }
        
        ensemble_pred = ensemble_model._weighted_voting(predictions)
        
        # Проверка корректности
        assert isinstance(ensemble_pred, np.ndarray)
        assert ensemble_pred.shape == (2, 3)
        assert np.allclose(ensemble_pred.sum(axis=1), 1.0, atol=1e-6)

class TestModelIntegration:
    """Интеграционные тесты"""
    
    def test_full_pipeline(self):
        """Тест полного пайплайна"""
        # Создание данных
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='1H')
        np.random.seed(42)
        
        price = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
        volume = np.random.randint(1000, 10000, len(dates))
        
        df = pd.DataFrame({
            'open': price * (1 + np.random.randn(len(dates)) * 0.001),
            'high': price * (1 + abs(np.random.randn(len(dates)) * 0.002)),
            'low': price * (1 - abs(np.random.randn(len(dates)) * 0.002)),
            'close': price,
            'volume': volume
        }, index=dates)
        
        # Создание ансамбля
        ensemble = create_advanced_models()
        
        # Подготовка признаков
        features, feature_names = ensemble.feature_engineer.prepare_features(df)
        
        # Создание целевой переменной
        target = ensemble._create_target_variable(df['close'])
        
        # Проверка корректности
        assert features.shape[0] == len(target)
        assert features.shape[1] == len(feature_names)
        assert not np.isnan(features).any()
        assert set(np.unique(target)).issubset({0, 1, 2})

if __name__ == "__main__":
    pytest.main([__file__, "-v"])