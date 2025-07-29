"""
MT5 Manager - управление подключением и торговлей через MetaTrader 5
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

import MetaTrader5 as mt5
from loguru import logger


class MT5Manager:
    """Менеджер для работы с MetaTrader 5"""
    
    def __init__(self, config: dict):
        self.config = config
        self.connected = False
        self.symbols = config.get("symbols", ["EURUSD", "GBPUSD", "USDJPY"])
        self.timeout = config.get("timeout", 5000)
        
    async def initialize(self) -> bool:
        """Инициализация подключения к MT5"""
        try:
            # Запуск MT5
            if not mt5.initialize():
                logger.error("Не удалось запустить MT5")
                return False
            
            # Авторизация (если указаны данные)
            if self.config.get("login") and self.config.get("password"):
                if not mt5.login(
                    login=self.config["login"],
                    password=self.config["password"],
                    server=self.config.get("server", "")
                ):
                    logger.error("Не удалось авторизоваться в MT5")
                    return False
            
            # Проверка подключения
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Не удалось получить информацию о счете")
                return False
            
            self.connected = True
            logger.info(f"MT5 подключен. Счет: {account_info.login}, Баланс: {account_info.balance}")
            
            # Проверяем доступность символов
            await self._check_symbols()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации MT5: {e}")
            return False
    
    async def _check_symbols(self):
        """Проверка доступности торговых символов"""
        available_symbols = []
        
        for symbol in self.symbols:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"Символ {symbol} недоступен")
                continue
            
            if not symbol_info.visible:
                # Пытаемся добавить символ в Market Watch
                if mt5.symbol_select(symbol, True):
                    logger.info(f"Символ {symbol} добавлен в Market Watch")
                else:
                    logger.warning(f"Не удалось добавить символ {symbol}")
                    continue
            
            available_symbols.append(symbol)
        
        self.symbols = available_symbols
        logger.info(f"Доступные символы: {self.symbols}")
    
    async def get_market_data(self) -> Dict[str, pd.DataFrame]:
        """Получение рыночных данных для всех символов"""
        if not self.connected:
            logger.error("MT5 не подключен")
            return {}
        
        market_data = {}
        
        for symbol in self.symbols:
            try:
                # Получаем данные за последние 1000 баров H1
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1000)
                
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    
                    # Добавляем технические индикаторы
                    df = await self._add_technical_indicators(df)
                    
                    market_data[symbol] = df
                    
                else:
                    logger.warning(f"Не удалось получить данные для {symbol}")
                    
            except Exception as e:
                logger.error(f"Ошибка получения данных для {symbol}: {e}")
        
        return market_data
    
    async def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление технических индикаторов"""
        try:
            # Скользящие средние
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # EMA
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
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
            
            # ATR
            high_low = df['high'] - df['low']
            high_close_prev = (df['high'] - df['close'].shift()).abs()
            low_close_prev = (df['low'] - df['close'].shift()).abs()
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
            # Объемы (если доступны)
            if 'tick_volume' in df.columns:
                df['volume_sma'] = df['tick_volume'].rolling(window=20).mean()
            
        except Exception as e:
            logger.error(f"Ошибка расчета технических индикаторов: {e}")
        
        return df
    
    async def place_order(self, signal: Dict) -> Dict:
        """Размещение ордера"""
        if not self.connected:
            return {"success": False, "error": "MT5 не подключен"}
        
        try:
            symbol = signal["symbol"]
            volume = signal["volume"]
            order_type = signal["type"]  # "buy" или "sell"
            
            # Получаем информацию о символе
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"success": False, "error": f"Символ {symbol} недоступен"}
            
            # Получаем текущие цены
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"success": False, "error": f"Не удалось получить цены для {symbol}"}
            
            # Определяем цену и тип ордера MT5
            if order_type.lower() == "buy":
                price = tick.ask
                mt5_order_type = mt5.ORDER_TYPE_BUY
            else:
                price = tick.bid
                mt5_order_type = mt5.ORDER_TYPE_SELL
            
            # Рассчитываем уровни SL и TP
            point = symbol_info.point
            sl_pips = signal.get("stop_loss_pips", 50)
            tp_pips = signal.get("take_profit_pips", 100)
            
            if order_type.lower() == "buy":
                sl_price = price - sl_pips * point * 10
                tp_price = price + tp_pips * point * 10
            else:
                sl_price = price + sl_pips * point * 10
                tp_price = price - tp_pips * point * 10
            
            # Формируем запрос
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_order_type,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 10,
                "magic": 12345,
                "comment": "ForexBot AI",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Отправляем ордер
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "success": False,
                    "error": f"Ошибка размещения ордера: {result.retcode}",
                    "comment": result.comment
                }
            
            return {
                "success": True,
                "ticket": result.order,
                "price": result.price,
                "volume": result.volume
            }
            
        except Exception as e:
            logger.error(f"Ошибка размещения ордера: {e}")
            return {"success": False, "error": str(e)}
    
    async def close_position(self, ticket: int) -> Dict:
        """Закрытие позиции"""
        try:
            # Получаем информацию о позиции
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                return {"success": False, "error": "Позиция не найдена"}
            
            position = position[0]
            
            # Определяем противоположный тип ордера
            if position.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(position.symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(position.symbol).ask
            
            # Формируем запрос на закрытие
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 10,
                "magic": 12345,
                "comment": "ForexBot AI Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "success": False,
                    "error": f"Ошибка закрытия позиции: {result.retcode}"
                }
            
            return {"success": True, "profit": result.profit}
            
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции {ticket}: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_position(self, ticket: int) -> Optional[Dict]:
        """Получение информации о позиции"""
        try:
            positions = mt5.positions_get(ticket=ticket)
            if positions is None or len(positions) == 0:
                return None
            
            position = positions[0]
            return {
                "ticket": position.ticket,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": "buy" if position.type == mt5.POSITION_TYPE_BUY else "sell",
                "open_price": position.price_open,
                "current_price": position.price_current,
                "profit": position.profit,
                "swap": position.swap,
                "commission": position.commission
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения позиции {ticket}: {e}")
            return None
    
    async def get_position_profit(self, ticket: int) -> float:
        """Получение прибыли по позиции"""
        position_info = await self.get_position(ticket)
        if position_info:
            return position_info["profit"]
        return 0.0
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Получение текущей цены символа"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            return (tick.ask + tick.bid) / 2
        except Exception as e:
            logger.error(f"Ошибка получения цены {symbol}: {e}")
            return None
    
    async def get_balance(self) -> float:
        """Получение баланса счета"""
        try:
            account_info = mt5.account_info()
            return account_info.balance if account_info else 0.0
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return 0.0
    
    async def get_equity(self) -> float:
        """Получение эквити счета"""
        try:
            account_info = mt5.account_info()
            return account_info.equity if account_info else 0.0
        except Exception as e:
            logger.error(f"Ошибка получения эквити: {e}")
            return 0.0
    
    async def get_margin_level(self) -> float:
        """Получение уровня маржи"""
        try:
            account_info = mt5.account_info()
            if account_info and account_info.margin > 0:
                return (account_info.equity / account_info.margin) * 100
            return 100.0
        except Exception as e:
            logger.error(f"Ошибка получения уровня маржи: {e}")
            return 100.0
    
    async def get_open_positions(self) -> List[Dict]:
        """Получение всех открытых позиций"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                result.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": "buy" if pos.type == mt5.POSITION_TYPE_BUY else "sell",
                    "open_price": pos.price_open,
                    "current_price": pos.price_current,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "commission": pos.commission,
                    "open_time": datetime.fromtimestamp(pos.time)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка получения открытых позиций: {e}")
            return []
    
    async def shutdown(self):
        """Завершение работы с MT5"""
        try:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 отключен")
        except Exception as e:
            logger.error(f"Ошибка отключения MT5: {e}")
    
    def __del__(self):
        """Деструктор"""
        if self.connected:
            mt5.shutdown()