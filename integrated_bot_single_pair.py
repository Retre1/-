#!/usr/bin/env python3
"""
Integrated ForexBot with Single Pair Models
Интегрированный бот с системой "одна пара - одна модель"
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from single_pair_model import SinglePairModelManager, SinglePairModel

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic модели
class TradeSignal(BaseModel):
    symbol: str
    timeframe: str
    direction: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    timestamp: str

class ModelInfo(BaseModel):
    symbol: str
    timeframe: str
    trained: bool
    accuracy: float
    last_trained: Optional[str]
    backtest_results: Dict

class IntegratedForexBotSinglePair:
    """Интегрированный торговый бот с отдельными моделями для каждой пары"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.app = FastAPI(title="ForexBot Single Pair", version="1.0.0")
        
        # Менеджер моделей
        self.model_manager = SinglePairModelManager(config)
        
        # Состояние бота
        self.status = "stopped"
        self.start_time = None
        self.capital = 10000
        self.trades = []
        self.positions = {}
        self.stats = {}
        
        # Настройка API endpoints
        self._setup_endpoints()
        
        # Инициализация моделей
        self._initialize_models()
    
    def _initialize_models(self):
        """Инициализация моделей для всех пар"""
        try:
            # Получение списка пар из конфигурации
            symbols = self.config.get('mt5', {}).get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY'])
            timeframes = self.config.get('ai', {}).get('timeframes', ['H1', 'H4'])
            
            logger.info(f"🚀 Инициализация моделей для {len(symbols)} пар и {len(timeframes)} таймфреймов")
            
            # Создание моделей
            for symbol in symbols:
                for timeframe in timeframes:
                    self.model_manager.create_model(symbol, timeframe)
            
            logger.info("✅ Модели инициализированы")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации моделей: {e}")
    
    def _setup_endpoints(self):
        """Настройка API endpoints"""
        
        @self.app.get("/")
        async def root():
            """Главная страница"""
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ForexBot Single Pair</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
                    .running { background-color: #d4edda; color: #155724; }
                    .stopped { background-color: #f8d7da; color: #721c24; }
                    .model-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
                    .model-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                    .model-card.trained { border-color: #28a745; }
                    .model-card.not-trained { border-color: #dc3545; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🤖 ForexBot Single Pair</h1>
                    <div id="status" class="status stopped">Статус: Остановлен</div>
                    
                    <h2>📊 Модели валютных пар</h2>
                    <div id="models" class="model-grid"></div>
                    
                    <h2>📈 Последние сигналы</h2>
                    <div id="signals"></div>
                    
                    <h2>💰 Статистика</h2>
                    <div id="stats"></div>
                </div>
                
                <script>
                    async function updateStatus() {
                        try {
                            const response = await fetch('/api/status');
                            const data = await response.json();
                            
                            const statusDiv = document.getElementById('status');
                            statusDiv.textContent = `Статус: ${data.status}`;
                            statusDiv.className = `status ${data.status}`;
                        } catch (error) {
                            console.error('Ошибка обновления статуса:', error);
                        }
                    }
                    
                    async function updateModels() {
                        try {
                            const response = await fetch('/api/models');
                            const data = await response.json();
                            
                            const modelsDiv = document.getElementById('models');
                            modelsDiv.innerHTML = '';
                            
                            for (const [key, model] of Object.entries(data.models)) {
                                const card = document.createElement('div');
                                card.className = `model-card ${model.trained ? 'trained' : 'not-trained'}`;
                                
                                card.innerHTML = `
                                    <h3>${key}</h3>
                                    <p><strong>Статус:</strong> ${model.trained ? 'Обучена' : 'Не обучена'}</p>
                                    <p><strong>Точность:</strong> ${(model.accuracy * 100).toFixed(2)}%</p>
                                    <p><strong>Последнее обучение:</strong> ${model.last_trained || 'Нет'}</p>
                                `;
                                
                                modelsDiv.appendChild(card);
                            }
                        } catch (error) {
                            console.error('Ошибка обновления моделей:', error);
                        }
                    }
                    
                    // Обновление каждые 5 секунд
                    setInterval(() => {
                        updateStatus();
                        updateModels();
                    }, 5000);
                    
                    // Начальное обновление
                    updateStatus();
                    updateModels();
                </script>
            </body>
            </html>
            """)
        
        @self.app.get("/api/status")
        async def get_status():
            """Получение статуса бота"""
            return {
                "status": self.status,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "capital": self.capital,
                "total_trades": len(self.trades),
                "active_positions": len(self.positions)
            }
        
        @self.app.get("/api/models")
        async def get_models():
            """Получение информации о моделях"""
            return {
                "models": self.model_manager.get_model_status(),
                "total_models": len(self.model_manager.models),
                "trained_models": len([m for m in self.model_manager.models.values() if m.model_status['trained']])
            }
        
        @self.app.post("/api/models/train")
        async def train_models(request: Dict):
            """Обучение моделей"""
            symbol = request.get('symbol')
            timeframe = request.get('timeframe')
            start_date = request.get('start_date', '2023-01-01')
            end_date = request.get('end_date', '2023-12-31')
            
            if symbol and timeframe:
                # Обучение конкретной модели
                model = self.model_manager.create_model(symbol, timeframe)
                result = model.train_model(start_date, end_date)
                return result
            else:
                # Обучение всех моделей
                pairs = [(s, tf) for s in ['EURUSD', 'GBPUSD', 'USDJPY'] 
                        for tf in ['H1', 'H4']]
                results = self.model_manager.train_all_models(pairs, start_date, end_date)
                return {"results": results}
        
        @self.app.post("/api/predict")
        async def get_prediction(request: Dict):
            """Получение предсказания"""
            symbol = request.get('symbol', 'EURUSD')
            timeframe = request.get('timeframe', 'H1')
            market_data = request.get('market_data', {})
            
            # Преобразование данных в DataFrame
            df = pd.DataFrame(market_data)
            
            if df.empty:
                return {"error": "No market data provided"}
            
            # Получение предсказания
            model = self.model_manager.create_model(symbol, timeframe)
            prediction = model.predict(df)
            
            return prediction
        
        @self.app.get("/api/models/{symbol}/{timeframe}")
        async def get_model_info(symbol: str, timeframe: str):
            """Получение информации о конкретной модели"""
            model = self.model_manager.create_model(symbol, timeframe)
            return model.get_model_info()
        
        @self.app.post("/api/models/{symbol}/{timeframe}/retrain")
        async def retrain_model(symbol: str, timeframe: str, request: Dict):
            """Переобучение конкретной модели"""
            model = self.model_manager.create_model(symbol, timeframe)
            days_back = request.get('days_back', 365)
            result = model.retrain_model(days_back)
            return result
        
        @self.app.get("/api/signals")
        async def get_signals():
            """Получение последних сигналов"""
            signals = []
            
            for model_key, model in self.model_manager.models.items():
                if model.model_status['trained']:
                    # Здесь можно добавить логику получения последних сигналов
                    signal = {
                        "symbol": model.symbol,
                        "timeframe": model.timeframe,
                        "direction": "HOLD",  # Заглушка
                        "confidence": model.model_status['accuracy'],
                        "timestamp": datetime.now().isoformat()
                    }
                    signals.append(signal)
            
            return {"signals": signals}
        
        @self.app.get("/api/statistics")
        async def get_statistics():
            """Получение статистики"""
            return {
                "total_models": len(self.model_manager.models),
                "trained_models": len([m for m in self.model_manager.models.values() 
                                     if m.model_status['trained']]),
                "average_accuracy": np.mean([m.model_status['accuracy'] 
                                           for m in self.model_manager.models.values() 
                                           if m.model_status['trained']]),
                "capital": self.capital,
                "total_trades": len(self.trades),
                "active_positions": len(self.positions)
            }
    
    def start_bot(self):
        """Запуск бота"""
        self.status = "running"
        self.start_time = datetime.now()
        logger.info("🚀 Бот запущен")
    
    def stop_bot(self):
        """Остановка бота"""
        self.status = "stopped"
        logger.info("🛑 Бот остановлен")
    
    def get_statistics(self) -> Dict:
        """Получение статистики"""
        return {
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "capital": self.capital,
            "total_trades": len(self.trades),
            "active_positions": len(self.positions),
            "models_info": self.model_manager.get_model_status()
        }

# Создание экземпляра бота
config = {
    "mt5": {
        "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    },
    "ai": {
        "models": ["lstm", "xgboost", "lightgbm"],
        "timeframes": ["H1", "H4"],
        "min_accuracy_threshold": 0.65
    }
}

bot = IntegratedForexBotSinglePair(config)

# Запуск бота
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Запуск ForexBot Single Pair...")
    bot.start_bot()
    
    uvicorn.run(bot.app, host="0.0.0.0", port=8000)