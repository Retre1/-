#!/usr/bin/env python3
"""
Unit Tests for Backtesting System
Unit тесты для системы backtesting
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Добавление пути к модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_backtesting import (
    AdvancedBacktester,
    ModelOptimizer,
    PerformanceAnalyzer
)

class TestAdvancedBacktester:
    """Тесты для продвинутого backtester"""
    
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
    def sample_predictions(self):
        """Создание тестовых предсказаний"""
        # Создаем предсказания для 100 точек
        predictions = []
        for i in range(100):
            # Случайные предсказания с нормализацией
            pred = np.random.rand(3)
            pred = pred / pred.sum()  # Нормализация
            predictions.append(pred)
        return np.array(predictions)
    
    @pytest.fixture
    def backtester(self):
        """Создание backtester"""
        return AdvancedBacktester(initial_capital=10000, commission=0.001)
    
    def test_initialization(self, backtester):
        """Тест инициализации"""
        assert backtester.initial_capital == 10000
        assert backtester.commission == 0.001
        assert backtester.capital == 10000
        assert len(backtester.positions) == 0
        assert len(backtester.trades) == 0
    
    def test_reset(self, backtester):
        """Тест сброса состояния"""
        # Изменяем состояние
        backtester.capital = 5000
        backtester.positions['EURUSD'] = {'type': 'LONG', 'size': 100}
        backtester.trades.append({'profit': 100})
        
        # Сбрасываем
        backtester.reset()
        
        assert backtester.capital == 10000
        assert len(backtester.positions) == 0
        assert len(backtester.trades) == 0
    
    def test_get_signal(self, backtester):
        """Тест получения сигнала"""
        # Тест HOLD сигнала
        pred = np.array([0.8, 0.1, 0.1])  # Высокая вероятность HOLD
        signal = backtester._get_signal(pred, confidence_threshold=0.6)
        assert signal == 'HOLD'
        
        # Тест BUY сигнала
        pred = np.array([0.1, 0.8, 0.1])  # Высокая вероятность BUY
        signal = backtester._get_signal(pred, confidence_threshold=0.6)
        assert signal == 'BUY'
        
        # Тест SELL сигнала
        pred = np.array([0.1, 0.1, 0.8])  # Высокая вероятность SELL
        signal = backtester._get_signal(pred, confidence_threshold=0.6)
        assert signal == 'SELL'
        
        # Тест низкой уверенности
        pred = np.array([0.4, 0.3, 0.3])  # Низкая уверенность
        signal = backtester._get_signal(pred, confidence_threshold=0.6)
        assert signal == 'HOLD'
    
    def test_execute_trade_buy(self, backtester):
        """Тест выполнения покупки"""
        price = 1.0850
        timestamp = datetime.now()
        
        # Открытие позиции
        backtester._execute_trade('BUY', price, timestamp)
        
        assert 'EURUSD' in backtester.positions
        assert backtester.positions['EURUSD']['type'] == 'LONG'
        assert backtester.positions['EURUSD']['entry_price'] == price
        assert backtester.capital < 10000  # Капитал уменьшился
    
    def test_execute_trade_sell(self, backtester):
        """Тест выполнения продажи"""
        price = 1.0850
        timestamp = datetime.now()
        
        # Сначала открываем позицию
        backtester._execute_trade('BUY', price, timestamp)
        initial_capital = backtester.capital
        
        # Затем закрываем позицию
        sell_price = 1.0900  # Прибыльная сделка
        backtester._execute_trade('SELL', sell_price, timestamp)
        
        assert 'EURUSD' not in backtester.positions
        assert len(backtester.trades) == 1
        assert backtester.capital > initial_capital  # Капитал увеличился
    
    def test_update_equity_curve(self, backtester):
        """Тест обновления кривой доходности"""
        price = 1.0850
        
        # Открываем позицию
        backtester.positions['EURUSD'] = {
            'type': 'LONG',
            'size': 1000,
            'entry_price': 1.0800,
            'entry_time': datetime.now()
        }
        backtester.capital = 9000  # Уменьшаем капитал
        
        # Обновляем кривую доходности
        backtester._update_equity_curve(price)
        
        assert len(backtester.equity_curve) > 0
        assert backtester.equity_curve[-1] > 9000  # Прибыль от позиции
    
    def test_generate_results(self, backtester):
        """Тест генерации результатов"""
        # Добавляем тестовые сделки
        backtester.trades = [
            {
                'entry_time': datetime.now() - timedelta(hours=1),
                'exit_time': datetime.now(),
                'entry_price': 1.0800,
                'exit_price': 1.0850,
                'size': 1000,
                'profit': 50,
                'return': 0.05
            },
            {
                'entry_time': datetime.now() - timedelta(hours=2),
                'exit_time': datetime.now() - timedelta(hours=1),
                'entry_price': 1.0900,
                'exit_price': 1.0850,
                'size': 1000,
                'profit': -50,
                'return': -0.05
            }
        ]
        
        backtester.equity_curve = [10000, 10050, 10000]
        backtester.drawdown_curve = [0, 0, 0.5]
        backtester.max_drawdown = 0.5
        
        results = backtester._generate_results()
        
        assert results['total_trades'] == 2
        assert results['winning_trades'] == 1
        assert results['losing_trades'] == 1
        assert results['win_rate'] == 50.0
        assert results['total_profit'] == 0
        assert results['max_drawdown'] == 0.5
    
    def test_empty_results(self, backtester):
        """Тест пустых результатов"""
        results = backtester._empty_results()
        
        assert results['total_trades'] == 0
        assert results['winning_trades'] == 0
        assert results['losing_trades'] == 0
        assert results['win_rate'] == 0
        assert results['total_profit'] == 0
        assert results['max_drawdown'] == 0
    
    def test_run_backtest(self, backtester, sample_data, sample_predictions):
        """Тест запуска backtesting"""
        # Обрезаем данные до размера предсказаний
        df = sample_data.head(len(sample_predictions))
        
        results = backtester.run_backtest(df, sample_predictions, confidence_threshold=0.6)
        
        assert isinstance(results, dict)
        assert 'total_trades' in results
        assert 'win_rate' in results
        assert 'total_profit' in results
        assert 'max_drawdown' in results
        assert 'sharpe_ratio' in results

class TestModelOptimizer:
    """Тесты для оптимизатора моделей"""
    
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
    
    def test_initialization(self):
        """Тест инициализации оптимизатора"""
        # Создаем мок ансамбля
        class MockEnsemble:
            def __init__(self):
                self.models = {'xgboost': None, 'lightgbm': None}
                self.feature_engineer = None
        
        ensemble = MockEnsemble()
        optimizer = ModelOptimizer(ensemble)
        
        assert optimizer.ensemble_model == ensemble
        assert isinstance(optimizer.best_params, dict)
        assert isinstance(optimizer.optimization_history, list)

class TestPerformanceAnalyzer:
    """Тесты для анализатора производительности"""
    
    @pytest.fixture
    def analyzer(self):
        """Создание анализатора"""
        return PerformanceAnalyzer()
    
    @pytest.fixture
    def sample_results(self):
        """Создание тестовых результатов"""
        return {
            'total_trades': 100,
            'winning_trades': 60,
            'losing_trades': 40,
            'win_rate': 60.0,
            'total_profit': 1000.0,
            'total_return': 0.1,
            'avg_win': 50.0,
            'avg_loss': -30.0,
            'profit_factor': 1.67,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.05,
            'calmar_ratio': 2.0,
            'sortino_ratio': 1.5,
            'equity_curve': [10000, 10100, 10200, 10150, 10300],
            'drawdown_curve': [0, 0, 0, 0.02, 0],
            'trades': [
                {'profit': 50, 'return': 0.05},
                {'profit': -30, 'return': -0.03},
                {'profit': 40, 'return': 0.04}
            ]
        }
    
    def test_initialization(self, analyzer):
        """Тест инициализации"""
        assert isinstance(analyzer.results, dict)
        assert len(analyzer.results) == 0
    
    def test_analyze_model_performance(self, analyzer, sample_results):
        """Тест анализа производительности модели"""
        analyzer.analyze_model_performance(sample_results, "test_model")
        
        assert "test_model" in analyzer.results
        assert analyzer.results["test_model"] == sample_results
    
    def test_compare_models(self, analyzer, sample_results):
        """Тест сравнения моделей"""
        # Добавляем несколько моделей
        analyzer.analyze_model_performance(sample_results, "model1")
        
        model2_results = sample_results.copy()
        model2_results['total_profit'] = 2000.0
        model2_results['win_rate'] = 70.0
        analyzer.analyze_model_performance(model2_results, "model2")
        
        comparison_df = analyzer.compare_models()
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'Model' in comparison_df.columns
        assert 'Total Return (%)' in comparison_df.columns
        assert 'Sharpe Ratio' in comparison_df.columns
        assert 'Win Rate (%)' in comparison_df.columns
    
    def test_generate_report(self, analyzer, sample_results):
        """Тест генерации отчета"""
        analyzer.analyze_model_performance(sample_results, "test_model")
        
        report = analyzer.generate_report()
        
        assert isinstance(report, str)
        assert "ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ" in report
        assert "test_model" in report
        assert "60.00%" in report  # win rate

if __name__ == "__main__":
    pytest.main([__file__, "-v"])