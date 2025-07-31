#!/usr/bin/env python3
"""
Advanced Integrated ForexBot with Professional AI Models
Интегрированный ForexBot с профессиональными AI моделями
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from loguru import logger
import pandas as pd
import numpy as np

# Импорт продвинутых AI моделей
from advanced_ai_models import AdvancedEnsembleModel, create_advanced_models
from advanced_backtesting import AdvancedBacktester, ModelOptimizer, PerformanceAnalyzer

# Импорт менеджера моделей
from model_manager import ModelManager, create_model_endpoints

# Pydantic модели для API
class TradeSignal(BaseModel):
    symbol: str
    direction: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    timestamp: datetime
    model: str
    strategy: str

class TradingConfig(BaseModel):
    initial_capital: float = 10000
    max_risk_per_trade: float = 0.02  # 2% риска на сделку
    max_positions: int = 5
    confidence_threshold: float = 0.6
    enable_ai_trading: bool = True
    enable_risk_management: bool = True

class BotStatus(BaseModel):
    status: str
    uptime: str
    total_trades: int
    profitable_trades: int
    total_profit: float
    current_drawdown: float
    ai_models_loaded: bool
    last_signal: Optional[TradeSignal] = None

class IntegratedForexBotAdvanced:
    """Продвинутый интегрированный торговый бот с AI"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.app = FastAPI(title="ForexBot AI Advanced", version="2.0.0")
        
        # Инициализация компонентов
        self.ensemble_model = create_advanced_models()
        self.backtester = AdvancedBacktester()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Инициализация менеджера моделей
        self.model_manager = ModelManager(config)
        
        # Состояние бота
        self.status = "stopped"
        self.start_time = None
        self.capital = 10000
        self.trades = []
        self.positions = {}
        self.stats = {}
        
        # Настройка API endpoints
        self._setup_endpoints()
        
        # Инициализация моделей при запуске
        self._initialize_models()
    
    def _initialize_models(self):
        """Инициализация AI моделей"""
        try:
            symbols = self.config.get('mt5', {}).get('symbols', ['EURUSD'])
            timeframes = self.config.get('ai', {}).get('timeframes', ['H1'])
            
            logger.info("🚀 Инициализация AI моделей...")
            success = self.model_manager.initialize_models(symbols, timeframes)
            
            if success:
                logger.info("✅ AI модели инициализированы успешно!")
            else:
                logger.warning("⚠️ Проблемы с инициализацией некоторых моделей")
                
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации моделей: {e}")
    
    def _setup_endpoints(self):
        """Настройка API endpoints"""
        
        # Существующие endpoints...
        
        # Добавление endpoints для управления моделями
        create_model_endpoints(self.app, self.model_manager)
        
        # Новые endpoints для AI моделей
        @app.get("/api/ai/models")
        async def get_ai_models():
            """Получение информации о AI моделях"""
            return {
                "available_models": list(self.ensemble_model.models.keys()),
                "model_status": self.model_manager.get_models_status(),
                "feature_engineer": {
                    "feature_count": len(self.ensemble_model.feature_engineer.feature_names),
                    "feature_names": self.ensemble_model.feature_engineer.feature_names[:10]  # Первые 10
                }
            }
        
        @app.post("/api/ai/predict")
        async def get_ai_prediction(request: Dict):
            """Получение предсказания от AI моделей"""
            symbol = request.get('symbol', 'EURUSD')
            timeframe = request.get('timeframe', 'H1')
            market_data = request.get('market_data', {})
            
            # Преобразование данных в DataFrame
            df = pd.DataFrame(market_data)
            
            if df.empty:
                return {"error": "No market data provided"}
            
            # Получение предсказания
            prediction = self.model_manager.get_prediction(symbol, timeframe, df)
            return prediction
        
        @app.post("/api/ai/retrain")
        async def retrain_ai_models(request: Dict):
            """Переобучение AI моделей"""
            symbol = request.get('symbol', 'EURUSD')
            timeframe = request.get('timeframe', 'H1')
            force = request.get('force', False)
            
            results = self.model_manager.retrain_models(symbol, timeframe, force)
            return results
        
        @app.get("/api/ai/performance/{symbol}/{timeframe}")
        async def get_ai_performance(symbol: str, timeframe: str):
            """Получение производительности AI моделей"""
            return self.model_manager.get_model_performance(symbol, timeframe)
        
        @app.post("/api/ai/backtest")
        async def run_ai_backtest(request: Dict):
            """Запуск backtesting для AI моделей"""
            symbol = request.get('symbol', 'EURUSD')
            timeframe = request.get('timeframe', 'H1')
            start_date = request.get('start_date', '2023-01-01')
            end_date = request.get('end_date', '2023-12-31')
            
            try:
                # Подготовка данных
                df, features, target, feature_names = self.model_manager.training_pipeline.prepare_training_data(
                    symbol, timeframe, start_date, end_date
                )
                
                # Получение предсказаний
                predictions = self.ensemble_model.predict_ensemble(df)
                pred_array = predictions['ensemble_prediction']
                
                # Запуск backtesting
                results = self.backtester.run_backtest(df, pred_array)
                
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "period": f"{start_date} - {end_date}",
                    "backtest_results": results,
                    "data_info": {
                        "total_records": len(df),
                        "features_count": len(feature_names),
                        "predictions_shape": pred_array.shape
                    }
                }
                
            except Exception as e:
                return {"error": str(e)}
        
        @app.get("/api/ai/features")
        async def get_ai_features():
            """Получение информации о признаках"""
            return {
                "feature_names": self.ensemble_model.feature_engineer.feature_names,
                "feature_count": len(self.ensemble_model.feature_engineer.feature_names),
                "technical_indicators": [
                    "sma_20", "sma_50", "sma_200", "ema_12", "ema_26",
                    "macd", "rsi", "bb_upper", "bb_lower", "stoch_k",
                    "atr", "volume_sma", "price_momentum", "volatility"
                ],
                "advanced_features": [
                    "fractal_high", "fractal_low", "higher_high", "lower_low",
                    "price_rsi_divergence", "mtf_trend", "market_regime",
                    "volatility_regime", "support", "resistance"
                ]
            }

    def setup_logging(self):
        """Настройка логирования"""
        log_dir = "data/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logger.add(
            f"{log_dir}/bot_{datetime.now().strftime('%Y%m%d')}.log",
            rotation="1 day",
            retention="30 days",
            level="INFO"
        )
        
    async def initialize_ai_models(self):
        """Инициализация AI моделей"""
        try:
            logger.info("Инициализация продвинутых AI моделей...")
            
            # Создание ансамбля моделей
            self.ensemble_model = create_advanced_models()
            
            # Загрузка предобученных моделей (если есть)
            models_dir = "models"
            if os.path.exists(models_dir):
                logger.info("Загрузка предобученных моделей...")
                self.ensemble_model.load_models(models_dir)
                self.ai_models_loaded = True
            else:
                logger.warning("Предобученные модели не найдены. Бот будет работать в демо режиме.")
                
            logger.info("AI модели инициализированы успешно")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации AI моделей: {e}")
            self.ai_models_loaded = False
            
    async def load_market_data(self, symbol: str = "EURUSD", timeframe: str = "1H", 
                             days: int = 365) -> pd.DataFrame:
        """Загрузка рыночных данных"""
        try:
            # В реальном приложении здесь будет подключение к MT5 или другому источнику данных
            # Для демо создаем синтетические данные
            
            logger.info(f"Загрузка данных для {symbol} за последние {days} дней...")
            
            # Создание синтетических данных
            dates = pd.date_range(
                end=datetime.now(),
                periods=days * 24,  # Часовые данные
                freq='1H'
            )
            
            np.random.seed(42)
            base_price = 1.0850  # EURUSD
            
            # Генерация реалистичных цен
            returns = np.random.normal(0, 0.0005, len(dates))  # 0.05% волатильность
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Добавление трендов и сезонности
            trend = np.linspace(0, 0.01, len(dates))  # Восходящий тренд
            seasonal = 0.001 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)  # Дневная сезонность
            
            prices += trend + seasonal
            
            # Создание OHLCV данных
            df = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.0001, len(dates))),
                'high': prices * (1 + abs(np.random.normal(0, 0.0002, len(dates)))),
                'low': prices * (1 - abs(np.random.normal(0, 0.0002, len(dates)))),
                'close': prices,
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
            
            logger.info(f"Загружено {len(df)} записей данных")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return pd.DataFrame()
            
    async def generate_ai_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        """Генерация AI сигналов"""
        if not self.ai_models_loaded or self.ensemble_model is None:
            logger.warning("AI модели не загружены, возвращаем пустые сигналы")
            return []
            
        try:
            # Получение предсказаний от ансамбля
            predictions = self.ensemble_model.predict_ensemble(df)
            ensemble_pred = predictions['ensemble']
            
            signals = []
            
            # Анализ последних предсказаний
            for i in range(min(10, len(ensemble_pred))):  # Последние 10 предсказаний
                pred = ensemble_pred[-(i+1)]
                max_prob = np.max(pred)
                max_class = np.argmax(pred)
                
                if max_prob >= self.trading_config.confidence_threshold:
                    direction = "HOLD"
                    if max_class == 1:
                        direction = "BUY"
                    elif max_class == 2:
                        direction = "SELL"
                        
                    if direction != "HOLD":
                        signal = TradeSignal(
                            symbol="EURUSD",
                            direction=direction,
                            confidence=max_prob,
                            price=df['close'].iloc[-(i+1)],
                            timestamp=df.index[-(i+1)],
                            model="ensemble",
                            strategy="ai_ensemble"
                        )
                        signals.append(signal)
                        
            logger.info(f"Сгенерировано {len(signals)} AI сигналов")
            return signals
            
        except Exception as e:
            logger.error(f"Ошибка генерации AI сигналов: {e}")
            return []
            
    async def execute_trade(self, signal: TradeSignal) -> bool:
        """Выполнение торговой операции"""
        try:
            if not self.trading_config.enable_ai_trading:
                logger.info("AI торговля отключена")
                return False
                
            # Проверка риска
            if self.trading_config.enable_risk_management:
                if not self._check_risk_limits(signal):
                    logger.warning("Операция отклонена из-за превышения лимитов риска")
                    return False
                    
            # Выполнение сделки
            if signal.direction == "BUY":
                success = await self._open_long_position(signal)
            elif signal.direction == "SELL":
                success = await self._close_long_position(signal)
            else:
                return False
                
            if success:
                self._update_statistics(signal)
                logger.info(f"Сделка выполнена: {signal.direction} {signal.symbol} по цене {signal.price}")
                
            return success
            
        except Exception as e:
            logger.error(f"Ошибка выполнения сделки: {e}")
            return False
            
    def _check_risk_limits(self, signal: TradeSignal) -> bool:
        """Проверка лимитов риска"""
        # Проверка максимального количества позиций
        if len(self.positions) >= self.trading_config.max_positions:
            return False
            
        # Проверка риска на сделку
        position_value = self.capital * self.trading_config.max_risk_per_trade
        if position_value > self.capital * 0.1:  # Максимум 10% капитала на позицию
            return False
            
        return True
        
    async def _open_long_position(self, signal: TradeSignal) -> bool:
        """Открытие длинной позиции"""
        try:
            position_size = self.capital * self.trading_config.max_risk_per_trade / signal.price
            cost = position_size * signal.price
            
            if cost <= self.capital:
                self.positions[signal.symbol] = {
                    'type': 'LONG',
                    'size': position_size,
                    'entry_price': signal.price,
                    'entry_time': signal.timestamp,
                    'signal': signal
                }
                self.capital -= cost
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Ошибка открытия позиции: {e}")
            return False
            
    async def _close_long_position(self, signal: TradeSignal) -> bool:
        """Закрытие длинной позиции"""
        try:
            if signal.symbol in self.positions:
                position = self.positions[signal.symbol]
                if position['type'] == 'LONG':
                    revenue = position['size'] * signal.price
                    profit = revenue - (position['size'] * position['entry_price'])
                    
                    self.trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': signal.timestamp,
                        'entry_price': position['entry_price'],
                        'exit_price': signal.price,
                        'size': position['size'],
                        'profit': profit,
                        'signal': signal
                    })
                    
                    self.capital += revenue
                    del self.positions[signal.symbol]
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции: {e}")
            return False
            
    def _update_statistics(self, signal: TradeSignal):
        """Обновление статистики"""
        self.stats["total_trades"] += 1
        
        if signal.direction == "SELL" and self.trades:
            last_trade = self.trades[-1]
            if last_trade['profit'] > 0:
                self.stats["profitable_trades"] += 1
                
        self.stats["total_profit"] = sum(t['profit'] for t in self.trades)
        self.stats["win_rate"] = self.stats["profitable_trades"] / self.stats["total_trades"] if self.stats["total_trades"] > 0 else 0
        
    async def start(self):
        """Запуск бота"""
        try:
            logger.info("Запуск продвинутого торгового бота...")
            
            self.status = "starting"
            self.start_time = datetime.now()
            
            # Инициализация AI моделей
            await self.initialize_ai_models()
            
            # Загрузка рыночных данных
            market_data = await self.load_market_data()
            
            if market_data.empty:
                logger.error("Не удалось загрузить рыночные данные")
                return False
                
            # Генерация AI сигналов
            signals = await self.generate_ai_signals(market_data)
            
            # Выполнение торговых операций
            for signal in signals:
                await self.execute_trade(signal)
                
            self.status = "running"
            logger.info("Бот успешно запущен")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка запуска бота: {e}")
            self.status = "error"
            return False
            
    async def stop(self):
        """Остановка бота"""
        logger.info("Остановка торгового бота...")
        self.status = "stopped"
        
    def get_status(self) -> BotStatus:
        """Получение статуса бота"""
        uptime = "N/A"
        if self.start_time:
            uptime = str(datetime.now() - self.start_time)
            
        last_signal = None
        if self.trades:
            last_signal = self.trades[-1].get('signal')
            
        return BotStatus(
            status=self.status,
            uptime=uptime,
            total_trades=self.stats["total_trades"],
            profitable_trades=self.stats["profitable_trades"],
            total_profit=self.stats["total_profit"],
            current_drawdown=self.stats["max_drawdown"],
            ai_models_loaded=self.ai_models_loaded,
            last_signal=last_signal
        )
        
    def get_trades(self) -> List[Dict]:
        """Получение истории сделок"""
        return self.trades
        
    def get_positions(self) -> Dict:
        """Получение открытых позиций"""
        return self.positions
        
    def get_statistics(self) -> Dict:
        """Получение статистики"""
        return self.stats

# FastAPI приложение
app = FastAPI(title="Advanced ForexBot AI", version="2.0.0")

# Монтирование статических файлов
app.mount("/static", StaticFiles(directory="web_interface/frontend"), name="static")

# Инициализация бота
bot = None

@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения"""
    global bot
    
    # Загрузка конфигурации
    config = {}
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            config = json.load(f)
    else:
        config = {
            "initial_capital": 10000,
            "max_risk_per_trade": 0.02,
            "max_positions": 5,
            "confidence_threshold": 0.6,
            "enable_ai_trading": True,
            "enable_risk_management": True
        }
        
    bot = IntegratedForexBotAdvanced(config)

@app.get("/")
async def read_root():
    """Главная страница"""
    try:
        with open("web_interface/frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {
            "message": "Advanced ForexBot AI Trading System",
            "version": "2.0.0",
            "status": "running",
            "note": "Веб-интерфейс не найден"
        }

@app.get("/api/status")
async def get_status():
    """Получение статуса бота"""
    if bot:
        return bot.get_status()
    return {"status": "not_initialized"}

@app.post("/api/start")
async def start_bot():
    """Запуск бота"""
    if bot:
        success = await bot.start()
        return {"success": success, "message": "Бот запущен" if success else "Ошибка запуска"}
    return {"success": False, "message": "Бот не инициализирован"}

@app.post("/api/stop")
async def stop_bot():
    """Остановка бота"""
    if bot:
        await bot.stop()
        return {"success": True, "message": "Бот остановлен"}
    return {"success": False, "message": "Бот не инициализирован"}

@app.get("/api/trades")
async def get_trades():
    """Получение истории сделок"""
    if bot:
        return {"trades": bot.get_trades()}
    return {"trades": []}

@app.get("/api/positions")
async def get_positions():
    """Получение открытых позиций"""
    if bot:
        return {"positions": bot.get_positions()}
    return {"positions": {}}

@app.get("/api/statistics")
async def get_statistics():
    """Получение статистики"""
    if bot:
        return {"statistics": bot.get_statistics()}
    return {"statistics": {}}

# WebSocket для real-time обновлений
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Обработка сообщений от клиента
            await manager.send_personal_message(f"Message text was: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)