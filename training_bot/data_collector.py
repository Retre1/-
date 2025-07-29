"""
Data Collection Utilities for Forex Training
Утилиты для сбора данных для обучения форекс моделей
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from loguru import logger

# Попытка импорта MT5 (может не быть доступен на всех платформах)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 недоступен, будет использоваться только yfinance")

# Technical Analysis
try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib недоступен, будут использоваться простые индикаторы")


class ForexDataCollector:
    """Сборщик данных форекс из различных источников"""
    
    def __init__(self, data_source: str = "auto"):
        """
        Args:
            data_source: "mt5", "yahoo", "auto"
        """
        self.data_source = data_source
        self.mt5_initialized = False
        
        if data_source == "auto":
            self.data_source = "mt5" if MT5_AVAILABLE else "yahoo"
        
        logger.info(f"Инициализирован сборщик данных: {self.data_source}")
    
    def _initialize_mt5(self) -> bool:
        """Инициализация MT5"""
        if not MT5_AVAILABLE:
            return False
            
        if not self.mt5_initialized:
            if mt5.initialize():
                self.mt5_initialized = True
                logger.info("MT5 успешно инициализирован")
                
                # Вывод информации о терминале
                terminal_info = mt5.terminal_info()._asdict()
                logger.info(f"Terminal: {terminal_info['name']} {terminal_info['build']}")
                
                account_info = mt5.account_info()._asdict()
                logger.info(f"Account: {account_info['login']} ({account_info['server']})")
                
                return True
            else:
                logger.error("Не удалось инициализировать MT5")
                return False
        return True
    
    def get_mt5_data(self, symbol: str, timeframe: int, count: int = 10000) -> pd.DataFrame:
        """Получение данных из MT5"""
        if not self._initialize_mt5():
            raise Exception("MT5 недоступен")
        
        # Проверка доступности символа
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise Exception(f"Символ {symbol} недоступен в MT5")
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise Exception(f"Не удалось выбрать символ {symbol}")
        
        # Получение данных
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        
        if rates is None or len(rates) == 0:
            raise Exception(f"Не удалось получить данные для {symbol}")
        
        # Конвертация в DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Переименование колонок
        df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        
        logger.info(f"Получено {len(df)} записей для {symbol} из MT5")
        return df[['open', 'high', 'low', 'close', 'tick_volume']]
    
    def get_yahoo_data(self, symbol: str, period: str = "2y", interval: str = "1h") -> pd.DataFrame:
        """Получение данных из Yahoo Finance"""
        
        # Конвертация символов форекс для Yahoo
        yahoo_symbol = self._convert_to_yahoo_symbol(symbol)
        
        try:
            # Получение данных
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise Exception(f"Нет данных для {yahoo_symbol}")
            
            # Переименование колонок
            df.columns = df.columns.str.lower()
            df.index.name = 'time'
            
            # Добавление объема (если нет, создаем синтетический)
            if 'volume' not in df.columns:
                df['volume'] = np.random.randint(1000, 10000, len(df))
            
            logger.info(f"Получено {len(df)} записей для {symbol} из Yahoo Finance")
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Ошибка получения данных от Yahoo Finance: {e}")
            raise
    
    def _convert_to_yahoo_symbol(self, symbol: str) -> str:
        """Конвертация символа в формат Yahoo Finance"""
        
        # Основные форекс пары
        forex_pairs = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X', 
            'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'NZDUSD': 'NZDUSD=X',
            'EURJPY': 'EURJPY=X',
            'GBPJPY': 'GBPJPY=X',
            'EURGBP': 'EURGBP=X',
            'EURCHF': 'EURCHF=X',
            'EURAUD': 'EURAUD=X',
            'EURCAD': 'EURCAD=X',
            'GBPCHF': 'GBPCHF=X',
            'GBPAUD': 'GBPAUD=X',
            'GBPCAD': 'GBPCAD=X',
            'AUDCHF': 'AUDCHF=X',
            'AUDJPY': 'AUDJPY=X',
            'AUDCAD': 'AUDCAD=X',
            'CHFJPY': 'CHFJPY=X',
            'CADJPY': 'CADJPY=X',
            'CADCHF': 'CADCHF=X',
            'NZDJPY': 'NZDJPY=X',
            'NZDCHF': 'NZDCHF=X',
            'NZDCAD': 'NZDCAD=X'
        }
        
        return forex_pairs.get(symbol.upper(), f"{symbol}=X")
    
    def get_data(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Универсальный метод получения данных"""
        
        if self.data_source == "mt5":
            # MT5 параметры
            timeframe = kwargs.get('timeframe', mt5.TIMEFRAME_H1)
            count = kwargs.get('count', 10000)
            return self.get_mt5_data(symbol, timeframe, count)
            
        elif self.data_source == "yahoo":
            # Yahoo Finance параметры
            period = kwargs.get('period', '2y')
            interval = kwargs.get('interval', '1h')
            return self.get_yahoo_data(symbol, period, interval)
            
        else:
            raise ValueError(f"Неизвестный источник данных: {self.data_source}")
    
    def create_synthetic_data(self, symbol: str, days: int = 365, freq: str = 'H') -> pd.DataFrame:
        """Создание синтетических данных для тестирования"""
        
        # Создание временного ряда
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Параметры для разных валютных пар
        pair_params = {
            'EURUSD': {'base': 1.1000, 'volatility': 0.001, 'trend': 0.0001},
            'GBPUSD': {'base': 1.2500, 'volatility': 0.0012, 'trend': -0.0001},
            'USDJPY': {'base': 110.00, 'volatility': 0.008, 'trend': 0.001},
            'AUDUSD': {'base': 0.7200, 'volatility': 0.0011, 'trend': 0.0001},
            'USDCHF': {'base': 0.9200, 'volatility': 0.0009, 'trend': -0.0001}
        }
        
        params = pair_params.get(symbol, pair_params['EURUSD'])
        
        # Генерация цен с трендом и волатильностью
        np.random.seed(42)
        
        # Базовый тренд
        trend = np.linspace(0, params['trend'] * len(dates), len(dates))
        
        # Случайные изменения
        returns = np.random.normal(0, params['volatility'], len(dates))
        
        # Автокорреляция (имитация momentum)
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]
        
        # Расчет цен
        prices = params['base'] + trend + np.cumsum(returns)
        
        # Создание OHLC данных
        df = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.0003, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.0003, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000, 50000, len(dates))
        }, index=dates)
        
        # Корректировка high/low
        df['high'] = np.maximum(df[['open', 'close']].max(axis=1), df['high'])
        df['low'] = np.minimum(df[['open', 'close']].min(axis=1), df['low'])
        
        logger.info(f"Создано {len(df)} синтетических записей для {symbol}")
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление технических индикаторов"""
        
        df = df.copy()
        
        if TALIB_AVAILABLE:
            # Используем TA-Lib если доступен
            df = self._add_talib_indicators(df)
        else:
            # Простые индикаторы
            df = self._add_simple_indicators(df)
        
        return df
    
    def _add_talib_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление индикаторов через TA-Lib"""
        
        # Скользящие средние
        df['sma_5'] = ta.SMA(df['close'], timeperiod=5)
        df['sma_10'] = ta.SMA(df['close'], timeperiod=10)
        df['sma_20'] = ta.SMA(df['close'], timeperiod=20)
        df['sma_50'] = ta.SMA(df['close'], timeperiod=50)
        df['sma_100'] = ta.SMA(df['close'], timeperiod=100)
        df['sma_200'] = ta.SMA(df['close'], timeperiod=200)
        
        # Экспоненциальные скользящие
        df['ema_12'] = ta.EMA(df['close'], timeperiod=12)
        df['ema_26'] = ta.EMA(df['close'], timeperiod=26)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(df['close'])
        
        # RSI
        df['rsi'] = ta.RSI(df['close'], timeperiod=14)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(df['close'])
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'])
        
        # ADX
        df['adx'] = ta.ADX(df['high'], df['low'], df['close'])
        
        # CCI
        df['cci'] = ta.CCI(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = ta.WILLR(df['high'], df['low'], df['close'])
        
        # ATR
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'])
        
        # OBV (если есть объем)
        if 'volume' in df.columns:
            df['obv'] = ta.OBV(df['close'], df['volume'])
        
        # Candlestick patterns
        df['doji'] = ta.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = ta.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['hanging_man'] = ta.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
        df['shooting_star'] = ta.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        logger.info("Добавлены индикаторы TA-Lib")
        return df
    
    def _add_simple_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Простые индикаторы без TA-Lib"""
        
        # Скользящие средние
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # EMA
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
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
        hl = df['high'] - df['low']
        hc = np.abs(df['high'] - df['close'].shift())
        lc = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(hl, np.maximum(hc, lc))
        df['atr'] = tr.rolling(window=14).mean()
        
        # Stochastic
        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        
        logger.info("Добавлены простые индикаторы")
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление производных признаков"""
        
        df = df.copy()
        
        # Ценовые признаки
        df['hl_ratio'] = df['high'] / df['low']
        df['oc_ratio'] = df['open'] / df['close']
        df['body_size'] = np.abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        
        # Доходности
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Волатильность
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
            df[f'volatility_{period}_annualized'] = df[f'volatility_{period}'] * np.sqrt(252 * 24)  # Для часовых данных
        
        # Лаговые признаки
        for lag in [1, 2, 3, 5, 10]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
        
        # Скользящие статистики
        for window in [5, 10, 20, 50]:
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'close_min_{window}'] = df['close'].rolling(window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window).max()
            
            if 'volume' in df.columns:
                df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
                df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
        
        # Разности со скользящими средними
        for period in [10, 20, 50]:
            df[f'close_sma_diff_{period}'] = df['close'] - df[f'sma_{period}']
            df[f'close_sma_ratio_{period}'] = df['close'] / df[f'sma_{period}']
        
        # Позиция в диапазоне
        for period in [10, 20, 50]:
            period_high = df['high'].rolling(window=period).max()
            period_low = df['low'].rolling(window=period).min()
            df[f'position_in_range_{period}'] = (df['close'] - period_low) / (period_high - period_low)
        
        # Bollinger Band позиция
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Количество стандартных отклонений от среднего
        for period in [20, 50]:
            mean = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            df[f'z_score_{period}'] = (df['close'] - mean) / std
        
        # Паттерны объема (если доступен)
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['price_volume_trend'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
        
        logger.info("Добавлены производные признаки")
        return df
    
    def create_target_variables(self, df: pd.DataFrame, horizons: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
        """Создание целевых переменных для прогнозирования"""
        
        df = df.copy()
        
        for horizon in horizons:
            # Будущая цена
            df[f'target_price_{horizon}'] = df['close'].shift(-horizon)
            
            # Будущая доходность
            df[f'target_return_{horizon}'] = (df[f'target_price_{horizon}'] - df['close']) / df['close']
            
            # Направление движения
            df[f'target_direction_{horizon}'] = (df[f'target_return_{horizon}'] > 0).astype(int)
            
            # Категории движения
            returns = df[f'target_return_{horizon}']
            df[f'target_category_{horizon}'] = pd.cut(
                returns, 
                bins=[-np.inf, -0.002, -0.0005, 0.0005, 0.002, np.inf],
                labels=['strong_down', 'weak_down', 'sideways', 'weak_up', 'strong_up']
            )
            
            # Максимальная цена в период
            df[f'target_max_{horizon}'] = df['high'].rolling(window=horizon).max().shift(-horizon)
            df[f'target_min_{horizon}'] = df['low'].rolling(window=horizon).min().shift(-horizon)
            
            # Максимальная доходность в период
            max_return = (df[f'target_max_{horizon}'] - df['close']) / df['close']
            min_return = (df[f'target_min_{horizon}'] - df['close']) / df['close']
            df[f'target_max_return_{horizon}'] = max_return
            df[f'target_min_return_{horizon}'] = min_return
        
        logger.info(f"Созданы целевые переменные для горизонтов: {horizons}")
        return df
    
    async def collect_multiple_symbols(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """Сбор данных для нескольких символов"""
        
        results = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Сбор данных для {symbol}...")
                
                # Получение сырых данных
                raw_data = self.get_data(symbol, **kwargs)
                
                # Добавление индикаторов
                data_with_indicators = self.add_technical_indicators(raw_data)
                
                # Добавление производных признаков
                data_with_features = self.add_derived_features(data_with_indicators)
                
                # Создание целевых переменных
                final_data = self.create_target_variables(data_with_features)
                
                results[symbol] = final_data
                
                logger.info(f"✓ Данные для {symbol} собраны: {len(final_data)} записей, {len(final_data.columns)} признаков")
                
            except Exception as e:
                logger.error(f"✗ Ошибка при сборе данных для {symbol}: {e}")
                
                # Если не удалось получить реальные данные, создаем синтетические
                try:
                    logger.warning(f"Создание синтетических данных для {symbol}")
                    synthetic_data = self.create_synthetic_data(symbol)
                    data_with_indicators = self.add_technical_indicators(synthetic_data)
                    data_with_features = self.add_derived_features(data_with_indicators)
                    final_data = self.create_target_variables(data_with_features)
                    results[symbol] = final_data
                    logger.info(f"✓ Синтетические данные для {symbol} созданы")
                except Exception as e2:
                    logger.error(f"✗ Не удалось создать синтетические данные для {symbol}: {e2}")
        
        return results
    
    def save_data(self, data: Dict[str, pd.DataFrame], base_path: str = "data/collected"):
        """Сохранение собранных данных"""
        
        os.makedirs(base_path, exist_ok=True)
        
        for symbol, df in data.items():
            file_path = os.path.join(base_path, f"{symbol}.csv")
            df.to_csv(file_path)
            logger.info(f"Данные для {symbol} сохранены в {file_path}")
        
        # Сохранение метаданных
        metadata = {
            'collection_time': datetime.now().isoformat(),
            'data_source': self.data_source,
            'symbols': list(data.keys()),
            'total_records': sum(len(df) for df in data.values()),
            'features_count': len(data[list(data.keys())[0]].columns) if data else 0
        }
        
        import json
        with open(os.path.join(base_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Метаданные сохранены в {base_path}/metadata.json")
    
    def load_data(self, base_path: str = "data/collected") -> Dict[str, pd.DataFrame]:
        """Загрузка сохраненных данных"""
        
        if not os.path.exists(base_path):
            logger.error(f"Папка {base_path} не существует")
            return {}
        
        data = {}
        
        for file_name in os.listdir(base_path):
            if file_name.endswith('.csv'):
                symbol = file_name.replace('.csv', '')
                file_path = os.path.join(base_path, file_name)
                
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                data[symbol] = df
                logger.info(f"Загружены данные для {symbol}: {len(df)} записей")
        
        return data


# Пример использования
async def main():
    """Демонстрация работы сборщика данных"""
    
    # Список символов для сбора
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    
    # Создание сборщика
    collector = ForexDataCollector(data_source="auto")
    
    # Сбор данных
    logger.info("Начинаем сбор данных...")
    data = await collector.collect_multiple_symbols(
        symbols,
        period="1y",  # Для Yahoo Finance
        interval="1h",
        count=8760  # Для MT5 (1 год часовых данных)
    )
    
    # Сохранение данных
    collector.save_data(data)
    
    # Статистика
    logger.info("Сбор данных завершен!")
    for symbol, df in data.items():
        logger.info(f"{symbol}: {len(df)} записей, {len(df.columns)} признаков")


if __name__ == "__main__":
    asyncio.run(main())