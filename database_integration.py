#!/usr/bin/env python3
"""
Database Integration for ForexBot
Интеграция с базой данных для улучшения проекта
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select

Base = declarative_base()

# Модели данных
class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)  # BUY/SELL
    volume = Column(Float, nullable=False)
    open_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=True)
    open_time = Column(DateTime, nullable=False)
    close_time = Column(DateTime, nullable=True)
    profit = Column(Float, nullable=True)
    status = Column(String(20), default="OPEN")  # OPEN/CLOSED/CANCELLED
    strategy = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    
class Signal(Base):
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    model = Column(String(50), nullable=True)
    executed = Column(Boolean, default=False)
    
class TokenBurn(Base):
    __tablename__ = "token_burns"
    
    id = Column(Integer, primary_key=True)
    amount = Column(Integer, nullable=False)
    reason = Column(String(200), nullable=True)
    timestamp = Column(DateTime, nullable=False)
    tx_hash = Column(String(100), nullable=True)
    
class BotLog(Base):
    __tablename__ = "bot_logs"
    
    id = Column(Integer, primary_key=True)
    level = Column(String(20), nullable=False)  # INFO/WARNING/ERROR
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    module = Column(String(50), nullable=True)
    data = Column(Text, nullable=True)  # JSON дополнительных данных

class DatabaseManager:
    """Менеджер базы данных"""
    
    def __init__(self, db_url: str = "sqlite:///forexbot.db"):
        self.db_url = db_url
        self.engine = None
        self.SessionLocal = None
        
    async def initialize(self):
        """Инициализация базы данных"""
        # Создание таблиц
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def get_session(self):
        """Получение сессии БД"""
        return self.SessionLocal()
        
    async def save_trade(self, trade_data: Dict) -> int:
        """Сохранение сделки"""
        with self.get_session() as session:
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
            return trade.id
            
    async def update_trade(self, trade_id: int, update_data: Dict):
        """Обновление сделки"""
        with self.get_session() as session:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                for key, value in update_data.items():
                    setattr(trade, key, value)
                session.commit()
                
    async def get_trades(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Получение сделок"""
        with self.get_session() as session:
            trades = session.query(Trade).order_by(Trade.open_time.desc()).limit(limit).offset(offset).all()
            return [self._trade_to_dict(trade) for trade in trades]
            
    async def get_trade_statistics(self, days: int = 30) -> Dict:
        """Получение статистики сделок"""
        with self.get_session() as session:
            start_date = datetime.now() - timedelta(days=days)
            
            # Общая статистика
            total_trades = session.query(Trade).filter(Trade.open_time >= start_date).count()
            closed_trades = session.query(Trade).filter(
                Trade.open_time >= start_date,
                Trade.status == "CLOSED"
            ).count()
            
            # Прибыльные сделки
            profitable_trades = session.query(Trade).filter(
                Trade.open_time >= start_date,
                Trade.status == "CLOSED",
                Trade.profit > 0
            ).count()
            
            # Общая прибыль
            total_profit = session.query(Trade).filter(
                Trade.open_time >= start_date,
                Trade.status == "CLOSED"
            ).with_entities(func.sum(Trade.profit)).scalar() or 0
            
            return {
                "total_trades": total_trades,
                "closed_trades": closed_trades,
                "profitable_trades": profitable_trades,
                "total_profit": total_profit,
                "win_rate": (profitable_trades / closed_trades * 100) if closed_trades > 0 else 0
            }
            
    async def save_signal(self, signal_data: Dict) -> int:
        """Сохранение сигнала"""
        with self.get_session() as session:
            signal = Signal(**signal_data)
            session.add(signal)
            session.commit()
            return signal.id
            
    async def get_signals(self, limit: int = 50) -> List[Dict]:
        """Получение сигналов"""
        with self.get_session() as session:
            signals = session.query(Signal).order_by(Signal.timestamp.desc()).limit(limit).all()
            return [self._signal_to_dict(signal) for signal in signals]
            
    async def save_token_burn(self, burn_data: Dict) -> int:
        """Сохранение сжигания токенов"""
        with self.get_session() as session:
            burn = TokenBurn(**burn_data)
            session.add(burn)
            session.commit()
            return burn.id
            
    async def save_log(self, log_data: Dict) -> int:
        """Сохранение лога"""
        with self.get_session() as session:
            log = BotLog(**log_data)
            session.add(log)
            session.commit()
            return log.id
            
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Преобразование сделки в словарь"""
        return {
            "id": trade.id,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "volume": trade.volume,
            "open_price": trade.open_price,
            "close_price": trade.close_price,
            "open_time": trade.open_time.isoformat(),
            "close_time": trade.close_time.isoformat() if trade.close_time else None,
            "profit": trade.profit,
            "status": trade.status,
            "strategy": trade.strategy,
            "confidence": trade.confidence
        }
        
    def _signal_to_dict(self, signal: Signal) -> Dict:
        """Преобразование сигнала в словарь"""
        return {
            "id": signal.id,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "confidence": signal.confidence,
            "price": signal.price,
            "timestamp": signal.timestamp.isoformat(),
            "model": signal.model,
            "executed": signal.executed
        }

# Пример использования
async def main():
    """Пример использования базы данных"""
    db = DatabaseManager()
    await db.initialize()
    
    # Сохранение сделки
    trade_data = {
        "symbol": "EURUSD",
        "direction": "BUY",
        "volume": 0.1,
        "open_price": 1.0850,
        "open_time": datetime.now(),
        "strategy": "trend_following",
        "confidence": 0.85
    }
    
    trade_id = await db.save_trade(trade_data)
    print(f"Сделка сохранена с ID: {trade_id}")
    
    # Получение статистики
    stats = await db.get_trade_statistics(days=30)
    print(f"Статистика: {stats}")

if __name__ == "__main__":
    asyncio.run(main())