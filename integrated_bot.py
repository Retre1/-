#!/usr/bin/env python3
"""
ForexBot AI Trading System - Integrated Bot
Единый интегрированный бот, объединяющий все компоненты системы
"""

import asyncio
import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import uvicorn
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Trading imports
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    print("MetaTrader5 не установлен, работаем в демо режиме")

import pandas as pd
import numpy as np
from loguru import logger

# Local imports
sys.path.append(str(Path(__file__).parent))

try:
    from trading_bot.mt5_connector.mt5_manager import MT5Manager
    from trading_bot.ai_models.ensemble_predictor import EnsemblePredictor
    from solana_integration.token_manager import TokenManager
except ImportError as e:
    logger.warning(f"Некоторые модули не найдены: {e}")
    # Создаем заглушки для отсутствующих модулей
    class MT5Manager:
        def __init__(self, config): pass
        async def initialize(self): return True
        async def get_account_info(self): return {"balance": 0, "equity": 0}
        async def get_positions(self): return []
        async def place_order(self, symbol, order_type, volume, price, sl, tp): return True
    
    class EnsemblePredictor:
        def __init__(self, config): pass
        async def load_models(self): return True
        async def predict(self, data): return {"signal": "HOLD", "confidence": 0.5}
    
    class TokenManager:
        def __init__(self, config): pass
        async def initialize(self): return True
        async def get_token_balance(self): return 0
        async def burn_tokens(self, amount): return True


# Pydantic models
class BotStatus(BaseModel):
    is_running: bool
    open_positions: int
    balance: float
    equity: float
    monthly_results: Dict
    ai_models_status: Dict
    token_balance: float


class TradingSignal(BaseModel):
    symbol: str
    direction: str
    confidence: float
    predicted_price: float
    current_price: float


class TokenBurnRequest(BaseModel):
    amount: int
    reason: str


class BotControlRequest(BaseModel):
    action: str  # start, stop, restart


class IntegratedForexBot:
    """Интегрированный торговый бот с веб-интерфейсом"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.mt5_manager = MT5Manager(self.config.get("mt5", {}))
        self.ai_predictor = EnsemblePredictor(self.config.get("ai", {}))
        self.token_manager = TokenManager(self.config.get("solana", {}))
        
        # Состояние бота
        self.is_running = False
        self.positions = {}
        self.monthly_results = {}
        self.connected_clients = []
        
        # Статистика
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "current_drawdown": 0.0
        }
        
        logger.info("IntegratedForexBot инициализирован")
    
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
                "retrain_interval": 24
            },
            "strategies": {
                "trend_following": {"enabled": True, "weight": 0.4},
                "mean_reversion": {"enabled": True, "weight": 0.3},
                "breakout": {"enabled": True, "weight": 0.3}
            },
            "risk": {
                "max_risk_per_trade": 0.02,
                "max_daily_loss": 0.05,
                "max_concurrent_trades": 5,
                "stop_loss_pips": 50,
                "take_profit_pips": 100
            },
            "solana": {
                "token_address": "",
                "burn_percentage": 0.1,
                "min_profit_for_burn": 100
            },
            "web_interface": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }
    
    async def initialize(self) -> bool:
        """Инициализация всех компонентов"""
        try:
            logger.info("Инициализация компонентов...")
            
            # Инициализация MT5
            if not await self.mt5_manager.initialize():
                logger.warning("Не удалось инициализировать MT5, работаем в демо режиме")
            
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
        if self.is_running:
            logger.warning("Бот уже запущен")
            return
        
        logger.info("Запуск торгового бота...")
        self.is_running = True
        
        # Запуск торгового цикла
        asyncio.create_task(self._trading_cycle())
        
        logger.info("Торговый бот запущен")
    
    async def stop_trading(self):
        """Остановка торговли"""
        logger.info("Остановка торгового бота...")
        self.is_running = False
        
        # Закрытие всех позиций (опционально)
        # await self._close_all_positions()
        
        logger.info("Торговый бот остановлен")
    
    async def _trading_cycle(self):
        """Основной торговый цикл"""
        while self.is_running:
            try:
                # Получение рыночных данных
                market_data = await self._get_market_data()
                
                # Анализ и генерация сигналов
                signals = await self._generate_signals(market_data)
                
                # Выполнение торговых операций
                if signals:
                    await self._execute_trades(signals)
                
                # Управление открытыми позициями
                await self._manage_positions()
                
                # Обновление статистики
                await self._update_statistics()
                
                # Пауза между циклами
                await asyncio.sleep(30)  # 30 секунд
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ошибка в торговом цикле: {e}")
                await asyncio.sleep(60)
    
    async def _get_market_data(self) -> Dict:
        """Получение рыночных данных"""
        try:
            # Здесь должна быть логика получения данных из MT5
            # Пока возвращаем заглушку
            return {
                "EURUSD": {"bid": 1.0850, "ask": 1.0852, "time": datetime.now()},
                "GBPUSD": {"bid": 1.2650, "ask": 1.2652, "time": datetime.now()},
                "USDJPY": {"bid": 148.50, "ask": 148.52, "time": datetime.now()}
            }
        except Exception as e:
            logger.error(f"Ошибка получения рыночных данных: {e}")
            return {}
    
    async def _generate_signals(self, market_data: Dict) -> List[Dict]:
        """Генерация торговых сигналов"""
        signals = []
        
        try:
            for symbol, data in market_data.items():
                # Получение предсказания от AI
                prediction = await self.ai_predictor.predict({
                    "symbol": symbol,
                    "data": data
                })
                
                if prediction["confidence"] > 0.7:  # Минимальный порог уверенности
                    signals.append({
                        "symbol": symbol,
                        "direction": prediction["signal"],
                        "confidence": prediction["confidence"],
                        "current_price": data["bid"],
                        "predicted_price": prediction.get("predicted_price", data["bid"])
                    })
        
        except Exception as e:
            logger.error(f"Ошибка генерации сигналов: {e}")
        
        return signals
    
    async def _execute_trades(self, signals: List[Dict]):
        """Выполнение торговых операций"""
        for signal in signals:
            try:
                if signal["direction"] == "BUY":
                    # Логика покупки
                    logger.info(f"Сигнал на покупку {signal['symbol']} с уверенностью {signal['confidence']}")
                    
                elif signal["direction"] == "SELL":
                    # Логика продажи
                    logger.info(f"Сигнал на продажу {signal['symbol']} с уверенностью {signal['confidence']}")
                
            except Exception as e:
                logger.error(f"Ошибка выполнения сделки: {e}")
    
    async def _manage_positions(self):
        """Управление открытыми позициями"""
        try:
            positions = await self.mt5_manager.get_positions()
            
            for position in positions:
                # Логика управления позициями
                # - Проверка стоп-лосса и тейк-профита
                # - Трейлинг-стоп
                # - Частичное закрытие
                pass
                
        except Exception as e:
            logger.error(f"Ошибка управления позициями: {e}")
    
    async def _update_statistics(self):
        """Обновление статистики"""
        try:
            account_info = await self.mt5_manager.get_account_info()
            self.stats["balance"] = account_info.get("balance", 0)
            self.stats["equity"] = account_info.get("equity", 0)
            
        except Exception as e:
            logger.error(f"Ошибка обновления статистики: {e}")
    
    async def get_status(self) -> Dict:
        """Получение статуса бота"""
        try:
            account_info = await self.mt5_manager.get_account_info()
            positions = await self.mt5_manager.get_positions()
            token_balance = await self.token_manager.get_token_balance()
            
            return {
                "is_running": self.is_running,
                "open_positions": len(positions),
                "balance": account_info.get("balance", 0),
                "equity": account_info.get("equity", 0),
                "monthly_results": self.monthly_results,
                "ai_models_status": {"status": "active"},
                "token_balance": token_balance,
                "statistics": self.stats
            }
        except Exception as e:
            logger.error(f"Ошибка получения статуса: {e}")
            return {
                "is_running": self.is_running,
                "open_positions": 0,
                "balance": 0,
                "equity": 0,
                "monthly_results": {},
                "ai_models_status": {"status": "error"},
                "token_balance": 0,
                "statistics": self.stats
            }
    
    async def burn_tokens(self, amount: int, reason: str = "") -> bool:
        """Сжигание токенов"""
        try:
            success = await self.token_manager.burn_tokens(amount)
            if success:
                logger.info(f"Сожжено {amount} токенов. Причина: {reason}")
            return success
        except Exception as e:
            logger.error(f"Ошибка сжигания токенов: {e}")
            return False


# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)


# Глобальные переменные
bot: Optional[IntegratedForexBot] = None
manager = ConnectionManager()


# FastAPI приложение
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global bot
    bot = IntegratedForexBot()
    await bot.initialize()
    yield
    # Shutdown
    if bot:
        await bot.stop_trading()


app = FastAPI(
    title="ForexBot AI Dashboard",
    description="Интегрированный торговый бот с веб-интерфейсом",
    version="1.0.0",
    lifespan=lifespan
)

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение статических файлов
app.mount("/static", StaticFiles(directory="web_interface/frontend"), name="static")


# Dependency
async def get_bot() -> IntegratedForexBot:
    if bot is None:
        raise HTTPException(status_code=503, detail="Бот не инициализирован")
    return bot


# API endpoints
@app.get("/")
async def read_root():
    """Главная страница с веб-интерфейсом"""
    try:
        with open("web_interface/frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {
            "message": "ForexBot AI Trading System",
            "version": "1.0.0",
            "status": "running",
            "note": "Веб-интерфейс не найден"
        }


@app.get("/api/status", response_model=BotStatus)
async def get_bot_status(bot_instance: IntegratedForexBot = Depends(get_bot)):
    """Получение статуса бота"""
    status = await bot_instance.get_status()
    return BotStatus(**status)


@app.post("/api/control")
async def control_bot(request: BotControlRequest, bot_instance: IntegratedForexBot = Depends(get_bot)):
    """Управление ботом"""
    try:
        if request.action == "start":
            await bot_instance.start_trading()
            return {"message": "Бот запущен"}
        elif request.action == "stop":
            await bot_instance.stop_trading()
            return {"message": "Бот остановлен"}
        elif request.action == "restart":
            await bot_instance.stop_trading()
            await asyncio.sleep(2)
            await bot_instance.start_trading()
            return {"message": "Бот перезапущен"}
        else:
            raise HTTPException(status_code=400, detail="Неизвестное действие")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions")
async def get_predictions(bot_instance: IntegratedForexBot = Depends(get_bot)):
    """Получение предсказаний"""
    try:
        market_data = await bot_instance._get_market_data()
        signals = await bot_instance._generate_signals(market_data)
        
        return {
            "signals": signals,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions")
async def get_positions(bot_instance: IntegratedForexBot = Depends(get_bot)):
    """Получение открытых позиций"""
    try:
        positions = await bot_instance.mt5_manager.get_positions()
        return {
            "positions": positions,
            "count": len(positions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/account")
async def get_account_info(bot_instance: IntegratedForexBot = Depends(get_bot)):
    """Получение информации о счете"""
    try:
        account_info = await bot_instance.mt5_manager.get_account_info()
        return account_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/token/info")
async def get_token_info(bot_instance: IntegratedForexBot = Depends(get_bot)):
    """Получение информации о токенах"""
    try:
        token_balance = await bot_instance.token_manager.get_token_balance()
        return {
            "token_balance": token_balance,
            "token_name": bot_instance.config.get("solana", {}).get("token_name", "ForexBot Token"),
            "token_symbol": bot_instance.config.get("solana", {}).get("token_symbol", "FBT")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/token/burn")
async def burn_tokens(request: TokenBurnRequest, bot_instance: IntegratedForexBot = Depends(get_bot)):
    """Сжигание токенов"""
    try:
        success = await bot_instance.burn_tokens(request.amount, request.reason)
        if success:
            return {"message": f"Сожжено {request.amount} токенов"}
        else:
            raise HTTPException(status_code=500, detail="Ошибка сжигания токенов")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/statistics")
async def get_statistics(days: int = 30, bot_instance: IntegratedForexBot = Depends(get_bot)):
    """Получение статистики"""
    try:
        return {
            "statistics": bot_instance.stats,
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket для real-time обновлений"""
    await manager.connect(websocket)
    try:
        while True:
            # Отправка обновлений каждые 5 секунд
            status = await bot.get_status() if bot else {}
            await websocket.send_text(json.dumps({
                "type": "status_update",
                "data": status,
                "timestamp": datetime.now().isoformat()
            }))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="ForexBot AI Trading System - Integrated Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python integrated_bot.py --mode demo          # Демо режим
  python integrated_bot.py --mode trade         # Торговый режим
  python integrated_bot.py --web-only           # Только веб-интерфейс
  python integrated_bot.py --port 8080          # Кастомный порт
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["demo", "trade"],
        default="demo",
        help="Режим работы (demo/trade)"
    )
    
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Запуск только веб-интерфейса"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Порт для веб-интерфейса"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Хост для веб-интерфейса"
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger.add(
        "data/logs/integrated_bot.log",
        rotation="10 MB",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )
    
    logger.info("=" * 50)
    logger.info("ForexBot AI Trading System - Integrated Bot")
    logger.info("=" * 50)
    
    # Запуск веб-сервера
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Остановка системы...")
    except Exception as e:
        logger.error(f"Ошибка запуска: {e}")


if __name__ == "__main__":
    main()