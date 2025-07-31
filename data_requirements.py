#!/usr/bin/env python3
"""
Data Requirements for Professional AI Training
Требования к данным для профессионального обучения AI
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import yfinance as yf
import requests
import json

class DataRequirements:
    """Требования к данным для обучения AI моделей"""
    
    def __init__(self):
        # Рекомендуемые валютные пары
        self.major_pairs = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", 
            "AUDUSD", "USDCAD", "NZDUSD"
        ]
        
        self.cross_pairs = [
            "EURGBP", "EURJPY", "GBPJPY", "AUDJPY",
            "EURAUD", "GBPAUD", "AUDCAD", "CADJPY"
        ]
        
        self.exotic_pairs = [
            "USDTRY", "USDZAR", "EURTRY", "GBPZAR",
            "USDSGD", "USDHKD", "USDMXN", "USDBRL"
        ]
        
        # Таймфреймы для обучения
        self.timeframes = {
            "M1": "1m",      # 1 минута
            "M5": "5m",      # 5 минут
            "M15": "15m",    # 15 минут
            "M30": "30m",    # 30 минут
            "H1": "1h",      # 1 час
            "H4": "4h",      # 4 часа
            "D1": "1d",      # 1 день
            "W1": "1wk",     # 1 неделя
            "MN1": "1mo"     # 1 месяц
        }
        
        # Минимальные требования к данным
        self.min_data_requirements = {
            "training_samples": 10000,    # Минимум 10,000 записей для обучения
            "validation_samples": 2000,   # Минимум 2,000 записей для валидации
            "test_samples": 2000,         # Минимум 2,000 записей для тестирования
            "min_years": 3,               # Минимум 3 года исторических данных
            "min_quality_score": 0.95     # Минимум 95% качество данных
        }
    
    def get_recommended_pairs(self, category: str = "all") -> List[str]:
        """Получение рекомендуемых валютных пар"""
        if category == "major":
            return self.major_pairs
        elif category == "cross":
            return self.cross_pairs
        elif category == "exotic":
            return self.exotic_pairs
        else:
            return self.major_pairs + self.cross_pairs + self.exotic_pairs
    
    def get_timeframe_recommendations(self, strategy_type: str = "general") -> Dict:
        """Рекомендации по таймфреймам для разных стратегий"""
        
        recommendations = {
            "scalping": {
                "primary": ["M1", "M5", "M15"],
                "secondary": ["M30", "H1"],
                "description": "Скальпинг требует высокочастотных данных",
                "min_samples": 50000,
                "min_years": 1
            },
            "day_trading": {
                "primary": ["M15", "M30", "H1"],
                "secondary": ["H4", "D1"],
                "description": "Дневная торговля использует средние таймфреймы",
                "min_samples": 20000,
                "min_years": 2
            },
            "swing_trading": {
                "primary": ["H1", "H4", "D1"],
                "secondary": ["W1", "MN1"],
                "description": "Свинг-трейдинг использует старшие таймфреймы",
                "min_samples": 15000,
                "min_years": 3
            },
            "position_trading": {
                "primary": ["H4", "D1", "W1"],
                "secondary": ["MN1"],
                "description": "Позиционная торговля требует долгосрочных данных",
                "min_samples": 10000,
                "min_years": 5
            },
            "general": {
                "primary": ["M15", "H1", "H4"],
                "secondary": ["D1"],
                "description": "Универсальная стратегия",
                "min_samples": 15000,
                "min_years": 3
            }
        }
        
        return recommendations.get(strategy_type, recommendations["general"])
    
    def calculate_data_requirements(self, 
                                 pairs: List[str], 
                                 timeframes: List[str],
                                 strategy_type: str = "general") -> Dict:
        """Расчет требований к данным"""
        
        timeframe_rec = self.get_timeframe_recommendations(strategy_type)
        
        # Расчет общего количества данных
        total_pairs = len(pairs)
        total_timeframes = len(timeframes)
        
        # Оценка размера данных
        samples_per_pair_timeframe = timeframe_rec["min_samples"]
        total_samples = total_pairs * total_timeframes * samples_per_pair_timeframe
        
        # Оценка времени загрузки
        estimated_download_time = total_samples / 1000  # секунды (примерно)
        
        # Оценка размера файлов
        bytes_per_record = 100  # примерный размер одной записи
        total_size_mb = (total_samples * bytes_per_record) / (1024 * 1024)
        
        return {
            "total_pairs": total_pairs,
            "total_timeframes": total_timeframes,
            "samples_per_pair_timeframe": samples_per_pair_timeframe,
            "total_samples": total_samples,
            "estimated_download_time_minutes": estimated_download_time / 60,
            "estimated_size_mb": total_size_mb,
            "min_years_required": timeframe_rec["min_years"],
            "strategy_type": strategy_type,
            "recommended_timeframes": timeframe_rec
        }
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Проверка качества данных"""
        
        quality_report = {
            "total_records": len(df),
            "date_range": {
                "start": df.index.min() if not df.empty else None,
                "end": df.index.max() if not df.empty else None
            },
            "missing_data": {},
            "duplicates": 0,
            "outliers": {},
            "quality_score": 0.0,
            "issues": []
        }
        
        if df.empty:
            quality_report["issues"].append("DataFrame пустой")
            return quality_report
        
        # Проверка пропущенных данных
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_percent = (missing_count / len(df)) * 100
                quality_report["missing_data"][col] = {
                    "count": missing_count,
                    "percent": missing_percent
                }
                
                if missing_percent > 5:
                    quality_report["issues"].append(f"Много пропущенных данных в {col}: {missing_percent:.2f}%")
        
        # Проверка дубликатов
        duplicates = df.duplicated().sum()
        quality_report["duplicates"] = duplicates
        
        if duplicates > 0:
            quality_report["issues"].append(f"Обнаружены дубликаты: {duplicates}")
        
        # Проверка выбросов
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_percent = (outliers / len(df)) * 100
                
                quality_report["outliers"][col] = {
                    "count": outliers,
                    "percent": outlier_percent
                }
                
                if outlier_percent > 10:
                    quality_report["issues"].append(f"Много выбросов в {col}: {outlier_percent:.2f}%")
        
        # Расчет общего качества
        total_issues = len(quality_report["issues"])
        max_issues = 10  # максимальное количество проблем
        quality_report["quality_score"] = max(0, 1 - (total_issues / max_issues))
        
        return quality_report
    
    def get_data_sources(self) -> Dict:
        """Рекомендуемые источники данных"""
        
        return {
            "free_sources": {
                "yfinance": {
                    "description": "Бесплатные данные Yahoo Finance",
                    "coverage": "Основные валютные пары",
                    "quality": "Средняя",
                    "update_frequency": "Real-time",
                    "limitations": "Ограниченная история, возможны пропуски"
                },
                "alpha_vantage": {
                    "description": "Бесплатный API с лимитами",
                    "coverage": "Широкий спектр инструментов",
                    "quality": "Хорошая",
                    "update_frequency": "Real-time",
                    "limitations": "Лимит запросов, платная подписка для больших объемов"
                },
                "quandl": {
                    "description": "Бесплатные экономические данные",
                    "coverage": "Макроэкономические данные",
                    "quality": "Высокая",
                    "update_frequency": "Daily",
                    "limitations": "Ограниченные валютные данные"
                }
            },
            "paid_sources": {
                "fxcm": {
                    "description": "Профессиональные данные FXCM",
                    "coverage": "Полный спектр валютных пар",
                    "quality": "Очень высокая",
                    "update_frequency": "Tick data",
                    "cost": "От $50/месяц"
                },
                "oanda": {
                    "description": "Данные OANDA",
                    "coverage": "Широкий спектр инструментов",
                    "quality": "Высокая",
                    "update_frequency": "Real-time",
                    "cost": "От $100/месяц"
                },
                "dukascopy": {
                    "description": "Исторические данные Dukascopy",
                    "coverage": "Основные валютные пары",
                    "quality": "Высокая",
                    "update_frequency": "Historical",
                    "cost": "Бесплатно для исторических данных"
                },
                "bloomberg": {
                    "description": "Профессиональная платформа Bloomberg",
                    "coverage": "Все инструменты",
                    "quality": "Максимальная",
                    "update_frequency": "Real-time",
                    "cost": "От $2000/месяц"
                }
            }
        }
    
    def generate_data_plan(self, 
                          strategy_type: str = "general",
                          budget: str = "free") -> Dict:
        """Генерация плана сбора данных"""
        
        # Рекомендации по парам
        if strategy_type == "scalping":
            recommended_pairs = self.major_pairs[:3]  # EURUSD, GBPUSD, USDJPY
        elif strategy_type == "day_trading":
            recommended_pairs = self.major_pairs + self.cross_pairs[:2]
        else:
            recommended_pairs = self.major_pairs + self.cross_pairs[:4]
        
        # Рекомендации по таймфреймам
        timeframe_rec = self.get_timeframe_recommendations(strategy_type)
        recommended_timeframes = timeframe_rec["primary"]
        
        # Расчет требований
        requirements = self.calculate_data_requirements(
            recommended_pairs, recommended_timeframes, strategy_type
        )
        
        # Выбор источников данных
        data_sources = self.get_data_sources()
        if budget == "free":
            sources = list(data_sources["free_sources"].keys())
        else:
            sources = list(data_sources["paid_sources"].keys())
        
        return {
            "strategy_type": strategy_type,
            "recommended_pairs": recommended_pairs,
            "recommended_timeframes": recommended_timeframes,
            "data_requirements": requirements,
            "data_sources": sources,
            "budget": budget,
            "estimated_cost": "Free" if budget == "free" else "$50-200/month",
            "implementation_steps": [
                "1. Выберите источник данных из списка",
                "2. Зарегистрируйтесь и получите API ключи",
                "3. Скачайте исторические данные",
                "4. Проверьте качество данных",
                "5. Подготовьте данные для обучения"
            ]
        }

# Пример использования
if __name__ == "__main__":
    dr = DataRequirements()
    
    print("📊 Анализ требований к данным для обучения AI моделей")
    print("=" * 60)
    
    # Анализ для разных стратегий
    strategies = ["scalping", "day_trading", "swing_trading", "position_trading"]
    
    for strategy in strategies:
        print(f"\n🎯 Стратегия: {strategy.upper()}")
        plan = dr.generate_data_plan(strategy, "free")
        
        print(f"Рекомендуемые пары: {', '.join(plan['recommended_pairs'])}")
        print(f"Рекомендуемые таймфреймы: {', '.join(plan['recommended_timeframes'])}")
        print(f"Общее количество записей: {plan['data_requirements']['total_samples']:,}")
        print(f"Минимальный период: {plan['data_requirements']['min_years_required']} лет")
        print(f"Примерный размер данных: {plan['data_requirements']['estimated_size_mb']:.1f} MB")
    
    print("\n" + "=" * 60)
    print("📋 Рекомендации по источникам данных:")
    
    sources = dr.get_data_sources()
    for category, source_list in sources.items():
        print(f"\n{category.upper()}:")
        for name, info in source_list.items():
            print(f"  {name}: {info['description']} - {info.get('cost', 'Free')}")
    
    print("\n🎯 Для профессионального обучения рекомендуется:")
    print("1. Использовать платные источники данных (FXCM, OANDA)")
    print("2. Собрать минимум 3-5 лет исторических данных")
    print("3. Использовать несколько таймфреймов (H1, H4, D1)")
    print("4. Обучаться на основных валютных парах")
    print("5. Регулярно обновлять данные и переобучать модели")