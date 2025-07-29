"""
ForexBot AI Trading System
Основной модуль торгового бота с интеграцией MetaTrader 5 и ИИ
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from loguru import logger
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

from mt5_connector.mt5_manager import MT5Manager
from ai_models.ensemble_predictor import EnsemblePredictor
from strategies.strategy_manager import StrategyManager
from risk_management.risk_manager import RiskManager
from solana_integration.token_manager import TokenManager


class ForexTradingBot:
    """Главный класс торгового бота"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.mt5_manager = MT5Manager(self.config["mt5"])
        self.ai_predictor = EnsemblePredictor(self.config["ai"])
        self.strategy_manager = StrategyManager(self.config["strategies"])
        self.risk_manager = RiskManager(self.config["risk"])
        self.token_manager = TokenManager(self.config["solana"])
        
        self.is_running = False
        self.positions = {}
        self.monthly_results = {}
        
        logger.info("ForexTradingBot инициализирован")
    
    def _load_config(self, config_path: str) -> dict:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Файл конфигурации {config_path} не найден, используем значения по умолчанию")
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Конфигурация по умолчанию"""
        return {
            "mt5": {
                "server": "",
                "login": 0,
                "password": "",
                "timeout": 5000,
                "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
            },
            "ai": {
                "models": ["lstm", "xgboost", "lightgbm"],
                "timeframes": ["M15", "H1", "H4"],
                "lookback_periods": [50, 100, 200],
                "retrain_interval": 24  # часы
            },
            "strategies": {
                "trend_following": {"enabled": True, "weight": 0.4},
                "mean_reversion": {"enabled": True, "weight": 0.3},
                "breakout": {"enabled": True, "weight": 0.3}
            },
            "risk": {
                "max_risk_per_trade": 0.02,  # 2% от депозита
                "max_daily_loss": 0.05,      # 5% от депозита
                "max_concurrent_trades": 5,
                "stop_loss_pips": 50,
                "take_profit_pips": 100
            },
            "solana": {
                "token_address": "",
                "burn_percentage": 0.1,  # 10% от прибыли
                "min_profit_for_burn": 100  # минимальная прибыль для сжигания
            }
        }
    
    async def initialize(self) -> bool:
        """Инициализация всех компонентов"""
        try:
            # Инициализация MT5
            if not await self.mt5_manager.initialize():
                raise Exception("Не удалось инициализировать MT5")
            
            # Загрузка AI моделей
            await self.ai_predictor.load_models()
            
            # Инициализация Solana
            await self.token_manager.initialize()
            
            logger.info("Все компоненты успешно инициализированы")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}")
            return False
    
    async def start_trading(self):
        """Запуск торговли"""
        if not await self.initialize():
            logger.error("Не удалось инициализировать бота")
            return
        
        self.is_running = True
        logger.info("Торговый бот запущен")
        
        # Основной торговый цикл
        while self.is_running:
            try:
                await self._trading_cycle()
                await asyncio.sleep(60)  # Проверка каждую минуту
                
            except Exception as e:
                logger.error(f"Ошибка в торговом цикле: {e}")
                await asyncio.sleep(300)  # Пауза 5 минут при ошибке
    
    async def _trading_cycle(self):
        """Один цикл торговли"""
        current_time = datetime.now()
        
        # Проверяем время торговли
        if not self._is_trading_time(current_time):
            return
        
        # Получаем рыночные данные
        market_data = await self.mt5_manager.get_market_data()
        if not market_data:
            return
        
        # Анализ ИИ
        predictions = await self.ai_predictor.predict(market_data)
        
        # Генерация сигналов
        signals = await self.strategy_manager.generate_signals(
            market_data, predictions
        )
        
        # Управление рисками
        filtered_signals = await self.risk_manager.filter_signals(
            signals, self.positions
        )
        
        # Исполнение сделок
        await self._execute_trades(filtered_signals)
        
        # Управление открытыми позициями
        await self._manage_positions()
        
        # Ежемесячная отчетность
        await self._check_monthly_results()
    
    def _is_trading_time(self, current_time: datetime) -> bool:
        """Проверка времени торговли"""
        # Избегаем торговли в выходные
        if current_time.weekday() >= 5:  # Суббота, воскресенье
            return False
        
        # Избегаем торговли во время важных новостей
        # Здесь можно добавить проверку экономического календаря
        
        return True
    
    async def _execute_trades(self, signals: List[Dict]):
        """Исполнение торговых сигналов"""
        for signal in signals:
            try:
                result = await self.mt5_manager.place_order(signal)
                if result["success"]:
                    self.positions[result["ticket"]] = {
                        "symbol": signal["symbol"],
                        "volume": signal["volume"],
                        "type": signal["type"],
                        "open_price": result["price"],
                        "open_time": datetime.now(),
                        "stop_loss": signal.get("stop_loss"),
                        "take_profit": signal.get("take_profit")
                    }
                    logger.info(f"Открыта позиция: {signal['symbol']} {signal['type']}")
                
            except Exception as e:
                logger.error(f"Ошибка исполнения сигнала: {e}")
    
    async def _manage_positions(self):
        """Управление открытыми позициями"""
        closed_positions = []
        
        for ticket, position in self.positions.items():
            try:
                # Проверяем статус позиции в MT5
                mt5_position = await self.mt5_manager.get_position(ticket)
                
                if not mt5_position:  # Позиция закрыта
                    # Записываем результат
                    profit = await self.mt5_manager.get_position_profit(ticket)
                    closed_positions.append(ticket)
                    
                    logger.info(f"Позиция {ticket} закрыта с прибылью: {profit}")
                    
                else:
                    # Обновляем стоп-лосс и тейк-профит
                    await self._update_position_levels(ticket, position)
                    
            except Exception as e:
                logger.error(f"Ошибка управления позицией {ticket}: {e}")
        
        # Удаляем закрытые позиции
        for ticket in closed_positions:
            del self.positions[ticket]
    
    async def _update_position_levels(self, ticket: int, position: dict):
        """Обновление уровней позиции"""
        # Реализация трейлинг стопа и других механизмов
        current_price = await self.mt5_manager.get_current_price(position["symbol"])
        
        # Логика обновления stop_loss и take_profit
        # Здесь можно добавить более сложные алгоритмы
        pass
    
    async def _check_monthly_results(self):
        """Проверка месячных результатов для сжигания токенов"""
        current_month = datetime.now().strftime("%Y-%m")
        
        if current_month not in self.monthly_results:
            # Первый день месяца - подводим итоги предыдущего
            prev_month = (datetime.now() - timedelta(days=30)).strftime("%Y-%m")
            
            if prev_month in self.monthly_results:
                await self._process_monthly_burn(prev_month)
            
            self.monthly_results[current_month] = {
                "trades": 0,
                "profit": 0.0,
                "start_balance": await self.mt5_manager.get_balance()
            }
    
    async def _process_monthly_burn(self, month: str):
        """Обработка месячного сжигания токенов"""
        results = self.monthly_results[month]
        profit_percentage = (results["profit"] / results["start_balance"]) * 100
        
        logger.info(f"Месячные результаты {month}: {profit_percentage:.2f}%")
        
        if profit_percentage > 0:  # Есть прибыль
            burn_amount = await self.token_manager.calculate_burn_amount(
                profit_percentage
            )
            
            if burn_amount > 0:
                await self.token_manager.burn_tokens(burn_amount)
                logger.info(f"Сожжено токенов: {burn_amount}")
    
    async def stop_trading(self):
        """Остановка торговли"""
        self.is_running = False
        
        # Закрываем все открытые позиции
        for ticket in list(self.positions.keys()):
            await self.mt5_manager.close_position(ticket)
        
        await self.mt5_manager.shutdown()
        logger.info("Торговый бот остановлен")
    
    async def get_status(self) -> dict:
        """Получение статуса бота"""
        return {
            "is_running": self.is_running,
            "open_positions": len(self.positions),
            "balance": await self.mt5_manager.get_balance(),
            "equity": await self.mt5_manager.get_equity(),
            "monthly_results": self.monthly_results,
            "ai_models_status": await self.ai_predictor.get_status(),
            "token_balance": await self.token_manager.get_balance()
        }


async def main():
    """Главная функция"""
    bot = ForexTradingBot()
    
    try:
        await bot.start_trading()
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки")
        await bot.stop_trading()


if __name__ == "__main__":
    asyncio.run(main())