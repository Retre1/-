#!/usr/bin/env python3
"""
📊 Сбор данных из MetaTrader5 для обучения ForexBot AI
Подробное руководство по получению исторических данных
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MT5DataCollector:
    """Класс для сбора данных из MetaTrader5"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.mt5_config = config.get('mt5', {})
        self.symbols = self.mt5_config.get('symbols', ['EURUSD'])
        self.timeframes = self.mt5_config.get('timeframes', ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'])
        self.data_dir = Path('data/market_data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Словарь соответствия таймфреймов
        self.timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
    def connect_mt5(self) -> bool:
        """Подключение к MetaTrader5"""
        try:
            # Инициализация MT5
            if not mt5.initialize():
                logger.error(f"❌ Ошибка инициализации MT5: {mt5.last_error()}")
                return False
            
            # Настройка подключения
            mt5_config = self.mt5_config
            if mt5_config.get('server') and mt5_config.get('login'):
                # Подключение к конкретному серверу
                if not mt5.login(
                    login=mt5_config.get('login', 0),
                    password=mt5_config.get('password', ''),
                    server=mt5_config.get('server', '')
                ):
                    logger.error(f"❌ Ошибка входа в MT5: {mt5.last_error()}")
                    return False
            
            logger.info("✅ Успешное подключение к MetaTrader5")
            
            # Получение информации об аккаунте
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"💰 Аккаунт: {account_info.login}, Баланс: {account_info.balance}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к MT5: {e}")
            return False
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Получение информации о символе"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"⚠️ Символ {symbol} не найден")
                return None
            
            # Активация символа для торговли
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"⚠️ Не удалось активировать символ {symbol}")
                return None
            
            return {
                'name': symbol_info.name,
                'point': symbol_info.point,
                'digits': symbol_info.digits,
                'spread': symbol_info.spread,
                'trade_contract_size': symbol_info.trade_contract_size,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения информации о символе {symbol}: {e}")
            return None
    
    def collect_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: datetime,
        max_bars: int = 100000
    ) -> Optional[pd.DataFrame]:
        """Сбор исторических данных"""
        try:
            # Проверка подключения
            if not mt5.terminal_info():
                logger.error("❌ MT5 не подключен")
                return None
            
            # Получение MT5 таймфрейма
            mt5_timeframe = self.timeframe_map.get(timeframe)
            if mt5_timeframe is None:
                logger.error(f"❌ Неподдерживаемый таймфрейм: {timeframe}")
                return None
            
            logger.info(f"📊 Сбор данных для {symbol} {timeframe} с {start_date} по {end_date}")
            
            # Сбор данных
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"⚠️ Нет данных для {symbol} {timeframe}")
                return None
            
            # Преобразование в DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Переименование колонок
            df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            
            # Добавление дополнительных колонок
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            logger.info(f"✅ Собрано {len(df)} записей для {symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка сбора данных для {symbol} {timeframe}: {e}")
            return None
    
    def collect_multiple_timeframes(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Сбор данных для нескольких таймфреймов"""
        data = {}
        
        for timeframe in self.timeframes:
            logger.info(f"🔄 Сбор данных для {symbol} {timeframe}")
            
            df = self.collect_historical_data(symbol, timeframe, start_date, end_date)
            if df is not None:
                data[timeframe] = df
                
                # Сохранение данных
                self.save_data(symbol, timeframe, df)
            
            # Пауза между запросами
            time.sleep(0.1)
        
        return data
    
    def save_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Сохранение данных в файл"""
        try:
            # Создание директории
            symbol_dir = self.data_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Имя файла
            filename = f"{symbol}_{timeframe}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}.csv"
            filepath = symbol_dir / filename
            
            # Сохранение
            df.to_csv(filepath)
            logger.info(f"💾 Данные сохранены: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения данных: {e}")
    
    def get_recent_data(self, symbol: str, timeframe: str, bars: int = 1000) -> Optional[pd.DataFrame]:
        """Получение последних данных"""
        try:
            mt5_timeframe = self.timeframe_map.get(timeframe)
            if mt5_timeframe is None:
                return None
            
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
            
            if rates is None or len(rates) == 0:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения последних данных: {e}")
            return None
    
    def collect_all_symbols_data(
        self, 
        start_date: datetime, 
        end_date: datetime,
        timeframes: List[str] = None
    ):
        """Сбор данных для всех символов"""
        if timeframes is None:
            timeframes = self.timeframes
        
        for symbol in self.symbols:
            logger.info(f"🔄 Сбор данных для символа: {symbol}")
            
            # Получение информации о символе
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                continue
            
            # Сбор данных для всех таймфреймов
            data = self.collect_multiple_timeframes(symbol, start_date, end_date)
            
            # Сохранение метаданных
            self.save_symbol_metadata(symbol, symbol_info, data)
    
    def save_symbol_metadata(self, symbol: str, symbol_info: Dict, data: Dict):
        """Сохранение метаданных символа"""
        try:
            metadata = {
                'symbol': symbol,
                'symbol_info': symbol_info,
                'data_info': {
                    timeframe: {
                        'records': len(df),
                        'start_date': df.index[0].isoformat(),
                        'end_date': df.index[-1].isoformat(),
                        'file_size': os.path.getsize(f"data/market_data/{symbol}/{symbol}_{timeframe}_*.csv")
                    }
                    for timeframe, df in data.items()
                },
                'collected_at': datetime.now().isoformat()
            }
            
            # Сохранение метаданных
            metadata_file = self.data_dir / symbol / f"{symbol}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"💾 Метаданные сохранены: {metadata_file}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения метаданных: {e}")
    
    def validate_data_quality(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Проверка качества данных"""
        validation = {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_records': len(df),
            'date_range': {
                'start': df.index[0].isoformat(),
                'end': df.index[-1].isoformat()
            },
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.index.duplicated().sum(),
            'price_range': {
                'min': df[['open', 'high', 'low', 'close']].min().to_dict(),
                'max': df[['open', 'high', 'low', 'close']].max().to_dict()
            },
            'volume_stats': {
                'mean': df['tick_volume'].mean(),
                'std': df['tick_volume'].std(),
                'max': df['tick_volume'].max()
            }
        }
        
        # Проверка на аномалии
        validation['anomalies'] = {
            'zero_volume': (df['tick_volume'] == 0).sum(),
            'negative_prices': ((df[['open', 'high', 'low', 'close']] < 0).any(axis=1)).sum(),
            'high_low_inversion': (df['high'] < df['low']).sum()
        }
        
        return validation
    
    def generate_data_report(self):
        """Генерация отчета о собранных данных"""
        report = {
            'collection_date': datetime.now().isoformat(),
            'symbols': {},
            'total_files': 0,
            'total_size_mb': 0
        }
        
        for symbol in self.symbols:
            symbol_dir = self.data_dir / symbol
            if not symbol_dir.exists():
                continue
            
            symbol_data = {
                'timeframes': {},
                'files': [],
                'total_size_mb': 0
            }
            
            for file in symbol_dir.glob("*.csv"):
                timeframe = file.stem.split('_')[1]
                size_mb = file.stat().st_size / (1024 * 1024)
                
                symbol_data['files'].append({
                    'filename': file.name,
                    'size_mb': size_mb,
                    'timeframe': timeframe
                })
                symbol_data['total_size_mb'] += size_mb
                report['total_size_mb'] += size_mb
                report['total_files'] += 1
            
            report['symbols'][symbol] = symbol_data
        
        # Сохранение отчета
        report_file = self.data_dir / 'data_collection_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Отчет о данных сохранен: {report_file}")
        return report

def main():
    """Основная функция сбора данных"""
    
    # Загрузка конфигурации
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Создание коллектора
    collector = MT5DataCollector(config)
    
    # Подключение к MT5
    if not collector.connect_mt5():
        logger.error("❌ Не удалось подключиться к MT5")
        return
    
    # Определение периода сбора данных
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 год данных
    
    logger.info(f"📅 Период сбора данных: {start_date} - {end_date}")
    
    # Сбор данных для всех символов
    collector.collect_all_symbols_data(start_date, end_date)
    
    # Генерация отчета
    report = collector.generate_data_report()
    
    logger.info("✅ Сбор данных завершен!")
    logger.info(f"📊 Собрано файлов: {report['total_files']}")
    logger.info(f"💾 Общий размер: {report['total_size_mb']:.2f} MB")
    
    # Отключение от MT5
    mt5.shutdown()

if __name__ == "__main__":
    main()