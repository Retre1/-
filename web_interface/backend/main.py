"""
ForexBot Web Interface Backend
FastAPI приложение для мониторинга торгового бота
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Добавляем путь к торговому боту
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from trading_bot.main import ForexTradingBot
from trading_bot.mt5_connector.mt5_manager import MT5Manager
from solana_integration.token_manager import SolanaTokenManager

# Инициализация FastAPI
app = FastAPI(
    title="ForexBot AI Dashboard",
    description="Мониторинг и управление торговым ботом на форекс с интеграцией Solana",
    version="1.0.0"
)

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
trading_bot: Optional[ForexTradingBot] = None
connected_clients: List[WebSocket] = []


# Pydantic модели
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
                # Удаляем отключенные соединения
                self.disconnect(connection)


manager = ConnectionManager()


# Dependency для получения торгового бота
async def get_trading_bot() -> ForexTradingBot:
    global trading_bot
    if trading_bot is None:
        trading_bot = ForexTradingBot()
    return trading_bot


# API Endpoints

@app.get("/")
async def read_root():
    """Главная страница"""
    return {"message": "ForexBot AI Dashboard API"}


@app.get("/api/status", response_model=BotStatus)
async def get_bot_status(bot: ForexTradingBot = Depends(get_trading_bot)):
    """Получение статуса торгового бота"""
    try:
        status = await bot.get_status()
        return BotStatus(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/control")
async def control_bot(request: BotControlRequest, bot: ForexTradingBot = Depends(get_trading_bot)):
    """Управление торговым ботом"""
    try:
        if request.action == "start":
            if not bot.is_running:
                asyncio.create_task(bot.start_trading())
                await manager.broadcast(json.dumps({
                    "type": "bot_status",
                    "data": {"status": "starting", "timestamp": datetime.now().isoformat()}
                }))
                return {"message": "Бот запускается", "success": True}
            else:
                return {"message": "Бот уже запущен", "success": False}
        
        elif request.action == "stop":
            if bot.is_running:
                await bot.stop_trading()
                await manager.broadcast(json.dumps({
                    "type": "bot_status", 
                    "data": {"status": "stopped", "timestamp": datetime.now().isoformat()}
                }))
                return {"message": "Бот остановлен", "success": True}
            else:
                return {"message": "Бот уже остановлен", "success": False}
        
        elif request.action == "restart":
            if bot.is_running:
                await bot.stop_trading()
            asyncio.create_task(bot.start_trading())
            await manager.broadcast(json.dumps({
                "type": "bot_status",
                "data": {"status": "restarting", "timestamp": datetime.now().isoformat()}
            }))
            return {"message": "Бот перезапускается", "success": True}
        
        else:
            raise HTTPException(status_code=400, detail="Неизвестное действие")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions")
async def get_predictions(bot: ForexTradingBot = Depends(get_trading_bot)):
    """Получение прогнозов ИИ"""
    try:
        if not bot.mt5_manager.connected:
            raise HTTPException(status_code=503, detail="MT5 не подключен")
        
        # Получаем рыночные данные
        market_data = await bot.mt5_manager.get_market_data()
        
        if not market_data:
            raise HTTPException(status_code=503, detail="Нет рыночных данных")
        
        # Получаем прогнозы
        predictions = await bot.ai_predictor.predict(market_data)
        
        # Преобразуем в формат API
        result = []
        for symbol, pred_data in predictions.items():
            result.append(TradingSignal(
                symbol=symbol,
                direction=pred_data["trend_direction"],
                confidence=pred_data["confidence"],
                predicted_price=pred_data["predicted_price"],
                current_price=pred_data["current_price"]
            ))
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions")
async def get_positions(bot: ForexTradingBot = Depends(get_trading_bot)):
    """Получение открытых позиций"""
    try:
        if not bot.mt5_manager.connected:
            raise HTTPException(status_code=503, detail="MT5 не подключен")
        
        positions = await bot.mt5_manager.get_open_positions()
        return positions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/account")
async def get_account_info(bot: ForexTradingBot = Depends(get_trading_bot)):
    """Получение информации о торговом счете"""
    try:
        if not bot.mt5_manager.connected:
            raise HTTPException(status_code=503, detail="MT5 не подключен")
        
        balance = await bot.mt5_manager.get_balance()
        equity = await bot.mt5_manager.get_equity()
        margin_level = await bot.mt5_manager.get_margin_level()
        
        return {
            "balance": balance,
            "equity": equity,
            "margin_level": margin_level,
            "profit_loss": equity - balance,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/token/info")
async def get_token_info(bot: ForexTradingBot = Depends(get_trading_bot)):
    """Получение информации о токене"""
    try:
        token_info = await bot.token_manager.get_token_info()
        burn_stats = await bot.token_manager.get_burn_statistics()
        
        return {
            "token_info": token_info,
            "burn_statistics": burn_stats,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/token/burn")
async def burn_tokens(request: TokenBurnRequest, bot: ForexTradingBot = Depends(get_trading_bot)):
    """Ручное сжигание токенов"""
    try:
        success = await bot.token_manager.burn_tokens(request.amount)
        
        if success:
            # Уведомляем клиентов
            await manager.broadcast(json.dumps({
                "type": "token_burn",
                "data": {
                    "amount": request.amount,
                    "reason": request.reason,
                    "timestamp": datetime.now().isoformat()
                }
            }))
            return {"message": f"Сожжено {request.amount} токенов", "success": True}
        else:
            return {"message": "Ошибка сжигания токенов", "success": False}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/statistics")
async def get_statistics(
    days: int = 30,
    bot: ForexTradingBot = Depends(get_trading_bot)
):
    """Получение торговой статистики"""
    try:
        # Здесь должна быть логика получения статистики из БД
        # Пока возвращаем заглушку
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        stats = {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_trades": 150,
            "winning_trades": 95,
            "losing_trades": 55,
            "win_rate": 63.33,
            "total_profit": 2500.75,
            "total_loss": -1200.50,
            "net_profit": 1300.25,
            "max_drawdown": -5.2,
            "sharpe_ratio": 1.85,
            "profit_factor": 2.08,
            "avg_win": 26.32,
            "avg_loss": -21.83,
            "largest_win": 125.50,
            "largest_loss": -85.25,
            "consecutive_wins": 8,
            "consecutive_losses": 3,
            "monthly_returns": [
                {"month": "2024-01", "return": 8.5},
                {"month": "2024-02", "return": 12.3},
                {"month": "2024-03", "return": -2.1},
                {"month": "2024-04", "return": 15.7}
            ]
        }
        
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str, bot: ForexTradingBot = Depends(get_trading_bot)):
    """Получение рыночных данных для символа"""
    try:
        if not bot.mt5_manager.connected:
            raise HTTPException(status_code=503, detail="MT5 не подключен")
        
        market_data = await bot.mt5_manager.get_market_data()
        
        if symbol not in market_data:
            raise HTTPException(status_code=404, detail=f"Символ {symbol} не найден")
        
        df = market_data[symbol]
        
        # Преобразуем последние 100 записей в JSON
        data = df.tail(100).reset_index()
        data['time'] = data['time'].astype(str)
        
        return {
            "symbol": symbol,
            "data": data.to_dict(orient="records"),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket для real-time обновлений"""
    await manager.connect(websocket)
    try:
        while True:
            # Отправляем обновления каждые 5 секунд
            await asyncio.sleep(5)
            
            if trading_bot and trading_bot.is_running:
                try:
                    status = await trading_bot.get_status()
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "status_update",
                            "data": status,
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
                except Exception:
                    pass
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Статические файлы для фронтенда
app.mount("/static", StaticFiles(directory="web_interface/frontend/build"), name="static")


# Обработчик для SPA
@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_spa(full_path: str):
    """Обслуживание React SPA"""
    try:
        with open("web_interface/frontend/build/index.html", 'r') as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend не найден</h1><p>Запустите сборку React приложения</p>")


# Background tasks
async def background_tasks():
    """Фоновые задачи"""
    while True:
        try:
            # Отправляем периодические обновления всем подключенным клиентам
            if trading_bot and len(manager.active_connections) > 0:
                status = await trading_bot.get_status()
                await manager.broadcast(json.dumps({
                    "type": "periodic_update",
                    "data": status,
                    "timestamp": datetime.now().isoformat()
                }))
        except Exception as e:
            print(f"Ошибка в фоновых задачах: {e}")
        
        await asyncio.sleep(30)  # Обновления каждые 30 секунд


@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global trading_bot
    
    # Создаем торгового бота
    trading_bot = ForexTradingBot()
    
    # Запускаем фоновые задачи
    asyncio.create_task(background_tasks())
    
    print("ForexBot Dashboard запущен")


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке"""
    global trading_bot
    
    if trading_bot and trading_bot.is_running:
        await trading_bot.stop_trading()
    
    print("ForexBot Dashboard остановлен")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )