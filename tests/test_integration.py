#!/usr/bin/env python3
"""
Integration Tests for ForexBot AI
Интеграционные тесты для полной системы
"""

import pytest
import asyncio
import json
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Добавление пути к модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrated_bot_advanced import IntegratedForexBotAdvanced
from advanced_ai_models import create_advanced_models
from advanced_backtesting import AdvancedBacktester
from monitoring_system import ForexBotMetrics
from security_integration import SecurityManager
from database_integration import DatabaseManager
from notifications_system import NotificationManager
from analytics_enhancement import AnalyticsManager

class TestSystemIntegration:
    """Интеграционные тесты системы"""
    
    @pytest.fixture
    def sample_config(self):
        """Создание тестовой конфигурации"""
        return {
            "mt5": {
                "server": "TestServer",
                "login": 12345,
                "password": "test",
                "symbols": ["EURUSD", "GBPUSD"]
            },
            "ai": {
                "models": ["lstm", "xgboost", "lightgbm"],
                "timeframes": ["M15", "H1"],
                "lookback_periods": [50, 100],
                "retrain_interval": 24,
                "min_accuracy_threshold": 0.65
            },
            "risk": {
                "max_risk_per_trade": 0.02,
                "max_daily_loss": 0.05,
                "max_concurrent_trades": 3,
                "stop_loss_pips": 50,
                "take_profit_pips": 100
            },
            "notifications": {
                "telegram": {"enabled": False},
                "email": {"enabled": False},
                "webhook": {"enabled": False}
            },
            "logging": {
                "level": "INFO",
                "file_path": "test_logs.log"
            },
            "database": {
                "url": "sqlite:///test.db"
            },
            "web_interface": {
                "host": "127.0.0.1",
                "port": 8000
            }
        }
    
    @pytest.fixture
    def bot_instance(self, sample_config):
        """Создание экземпляра бота"""
        return IntegratedForexBotAdvanced(sample_config)
    
    @pytest.mark.asyncio
    async def test_bot_initialization(self, bot_instance):
        """Тест инициализации бота"""
        # Проверка основных компонентов
        assert bot_instance.config is not None
        assert bot_instance.status == "stopped"
        assert bot_instance.capital == 10000
        assert len(bot_instance.trades) == 0
        assert len(bot_instance.positions) == 0
        
        # Проверка AI моделей
        assert bot_instance.ensemble_model is not None
        assert bot_instance.backtester is not None
        assert bot_instance.performance_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_ai_models_integration(self, bot_instance):
        """Тест интеграции AI моделей"""
        # Создание тестовых данных
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
        
        # Тест предсказаний
        predictions = bot_instance.ensemble_model.predict_ensemble(df)
        
        assert isinstance(predictions, dict)
        assert 'ensemble_prediction' in predictions
        assert 'confidence' in predictions
        assert 'model_predictions' in predictions
        assert len(predictions['model_predictions']) > 0
    
    @pytest.mark.asyncio
    async def test_backtesting_integration(self, bot_instance):
        """Тест интеграции backtesting"""
        # Создание тестовых данных
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
        
        # Получение предсказаний
        predictions = bot_instance.ensemble_model.predict_ensemble(df)
        pred_array = predictions['ensemble_prediction']
        
        # Запуск backtesting
        results = bot_instance.backtester.run_backtest(df, pred_array)
        
        assert isinstance(results, dict)
        assert 'total_trades' in results
        assert 'win_rate' in results
        assert 'total_profit' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, bot_instance):
        """Тест интеграции мониторинга"""
        # Создание метрик
        metrics = ForexBotMetrics()
        
        # Симуляция торговых операций
        metrics.record_trade('BUY', 'EURUSD', 3600, 50.0)
        metrics.record_signal('lstm', 'BUY', 0.1)
        metrics.record_api_request('/api/trades', 'GET', 0.05)
        
        # Обновление метрик производительности
        stats = {
            'total_profit': 1000.0,
            'current_drawdown': 5.0,
            'active_positions': 2,
            'capital_balance': 11000.0,
            'win_rate': 65.0,
            'sharpe_ratio': 1.2
        }
        metrics.update_performance_metrics(stats)
        
        # Проверка метрик
        assert metrics.trades_total._value.get() > 0
        assert metrics.total_profit._value.get() == 1000.0
        assert metrics.win_rate._value.get() == 65.0
    
    @pytest.mark.asyncio
    async def test_security_integration(self, bot_instance):
        """Тест интеграции безопасности"""
        # Создание менеджера безопасности
        security_manager = SecurityManager()
        
        # Создание пользователя
        user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'testpassword123',
            'role': 'user'
        }
        
        user = security_manager.create_user(user_data)
        assert user.username == 'testuser'
        assert user.role == 'user'
        
        # Аутентификация
        auth_user = security_manager.authenticate_user('testuser', 'testpassword123')
        assert auth_user is not None
        assert auth_user.username == 'testuser'
        
        # Создание токена
        token = security_manager.create_access_token(auth_user)
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Проверка токена
        token_data = security_manager.verify_token(token)
        assert token_data is not None
        assert token_data['sub'] == 'testuser'
    
    @pytest.mark.asyncio
    async def test_database_integration(self, bot_instance):
        """Тест интеграции базы данных"""
        # Создание менеджера БД
        db_manager = DatabaseManager("sqlite:///test.db")
        await db_manager.initialize()
        
        # Сохранение сделки
        trade_data = {
            'symbol': 'EURUSD',
            'direction': 'BUY',
            'volume': 1000.0,
            'open_price': 1.0850,
            'open_time': datetime.now(),
            'strategy': 'AI_Ensemble',
            'confidence': 0.75
        }
        
        trade_id = await db_manager.save_trade(trade_data)
        assert trade_id > 0
        
        # Получение сделок
        trades = await db_manager.get_trades(limit=10)
        assert len(trades) > 0
        assert trades[0]['symbol'] == 'EURUSD'
        
        # Получение статистики
        stats = await db_manager.get_trade_statistics(days=30)
        assert isinstance(stats, dict)
        assert 'total_trades' in stats
    
    @pytest.mark.asyncio
    async def test_notifications_integration(self, bot_instance):
        """Тест интеграции уведомлений"""
        # Создание менеджера уведомлений
        config = {
            'telegram': {'enabled': False},
            'email': {'enabled': False},
            'webhook': {'enabled': False}
        }
        
        notification_manager = NotificationManager(config)
        
        # Добавление правил алертов
        alert_rule = {
            'name': 'High Profit Alert',
            'condition': 'profit > 1000',
            'notification_type': 'email',
            'recipients': ['test@example.com']
        }
        notification_manager.add_alert_rule(alert_rule)
        
        # Проверка алертов
        current_data = {'profit': 1500}
        await notification_manager.check_alerts(current_data)
        
        # Получение истории уведомлений
        history = notification_manager.get_notification_history()
        assert isinstance(history, list)
    
    @pytest.mark.asyncio
    async def test_analytics_integration(self, bot_instance):
        """Тест интеграции аналитики"""
        # Создание менеджера аналитики
        analytics_manager = AnalyticsManager()
        
        # Добавление тестовых данных
        trade_data = {
            'symbol': 'EURUSD',
            'direction': 'BUY',
            'volume': 1000.0,
            'open_price': 1.0850,
            'close_price': 1.0900,
            'open_time': datetime.now() - timedelta(hours=1),
            'close_time': datetime.now(),
            'profit': 50.0,
            'status': 'CLOSED',
            'strategy': 'AI_Ensemble',
            'confidence': 0.75
        }
        analytics_manager.add_trade(trade_data)
        
        # Расчет метрик производительности
        metrics = analytics_manager.calculate_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'total_trades' in metrics
        assert 'win_rate' in metrics
        assert 'total_profit' in metrics
        
        # Генерация отчета
        report = analytics_manager.generate_report()
        assert isinstance(report, dict)
        assert 'performance_metrics' in report
        assert 'strategy_analysis' in report
    
    @pytest.mark.asyncio
    async def test_full_trading_cycle(self, bot_instance):
        """Тест полного торгового цикла"""
        # Симуляция запуска бота
        bot_instance.status = "running"
        bot_instance.start_time = datetime.now()
        
        # Создание тестовых рыночных данных
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
        
        # Получение сигналов
        predictions = bot_instance.ensemble_model.predict_ensemble(df)
        
        # Симуляция торговых операций
        for i in range(min(10, len(df))):
            current_price = df['close'].iloc[i]
            pred = predictions['ensemble_prediction'][i]
            
            # Определение сигнала
            max_prob = np.max(pred)
            max_class = np.argmax(pred)
            
            if max_prob > 0.6:  # Порог уверенности
                if max_class == 1:  # BUY
                    # Открытие позиции
                    position_size = bot_instance.capital * 0.1 / current_price
                    bot_instance.positions['EURUSD'] = {
                        'type': 'LONG',
                        'size': position_size,
                        'entry_price': current_price,
                        'entry_time': df.index[i]
                    }
                    bot_instance.capital -= position_size * current_price
                    
                elif max_class == 2 and 'EURUSD' in bot_instance.positions:  # SELL
                    # Закрытие позиции
                    position = bot_instance.positions['EURUSD']
                    profit = (current_price - position['entry_price']) * position['size']
                    bot_instance.capital += position['size'] * current_price + profit
                    
                    # Запись сделки
                    trade = {
                        'symbol': 'EURUSD',
                        'direction': 'BUY',
                        'volume': position['size'],
                        'open_price': position['entry_price'],
                        'close_price': current_price,
                        'open_time': position['entry_time'],
                        'close_time': df.index[i],
                        'profit': profit,
                        'status': 'CLOSED',
                        'strategy': 'AI_Ensemble',
                        'confidence': max_prob
                    }
                    bot_instance.trades.append(trade)
                    del bot_instance.positions['EURUSD']
        
        # Проверка результатов
        assert len(bot_instance.trades) > 0
        assert bot_instance.capital != 10000  # Капитал изменился
        
        # Обновление статистики
        bot_instance._update_statistics()
        
        assert bot_instance.stats['total_trades'] > 0
        assert 'win_rate' in bot_instance.stats
        assert 'total_profit' in bot_instance.stats
    
    @pytest.mark.asyncio
    async def test_error_handling(self, bot_instance):
        """Тест обработки ошибок"""
        # Тест с некорректными данными
        with pytest.raises(Exception):
            bot_instance.ensemble_model.predict_ensemble(None)
        
        # Тест с пустыми данными
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            bot_instance.ensemble_model.predict_ensemble(empty_df)
        
        # Тест с некорректной конфигурацией
        invalid_config = {'invalid': 'config'}
        with pytest.raises(Exception):
            invalid_bot = IntegratedForexBotAdvanced(invalid_config)
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, bot_instance):
        """Тест производительности под нагрузкой"""
        import time
        
        # Создание большого объема данных
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='1min')
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
        
        # Тест времени выполнения предсказаний
        start_time = time.time()
        predictions = bot_instance.ensemble_model.predict_ensemble(df.head(1000))
        end_time = time.time()
        
        prediction_time = end_time - start_time
        assert prediction_time < 10.0  # Предсказания должны выполняться менее 10 секунд
        
        # Тест памяти
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        assert memory_usage < 1000  # Использование памяти должно быть менее 1GB

if __name__ == "__main__":
    pytest.main([__file__, "-v"])