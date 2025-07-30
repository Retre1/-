#!/usr/bin/env python3
"""
Analytics Enhancement for ForexBot
Расширенная аналитика и отчетность
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

class AnalyticsManager:
    """Менеджер аналитики"""
    
    def __init__(self):
        self.trades_data = []
        self.signals_data = []
        self.market_data = []
        
    def add_trade(self, trade_data: Dict):
        """Добавление сделки для анализа"""
        self.trades_data.append(trade_data)
        
    def add_signal(self, signal_data: Dict):
        """Добавление сигнала для анализа"""
        self.signals_data.append(signal_data)
        
    def add_market_data(self, market_data: Dict):
        """Добавление рыночных данных"""
        self.market_data.append(market_data)
        
    def calculate_performance_metrics(self) -> Dict:
        """Расчет метрик производительности"""
        if not self.trades_data:
            return {}
            
        df = pd.DataFrame(self.trades_data)
        closed_trades = df[df['status'] == 'CLOSED']
        
        if len(closed_trades) == 0:
            return {}
            
        # Базовые метрики
        total_trades = len(closed_trades)
        profitable_trades = len(closed_trades[closed_trades['profit'] > 0])
        total_profit = closed_trades['profit'].sum()
        win_rate = (profitable_trades / total_trades) * 100
        
        # Продвинутые метрики
        avg_win = closed_trades[closed_trades['profit'] > 0]['profit'].mean()
        avg_loss = closed_trades[closed_trades['profit'] < 0]['profit'].mean()
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Максимальная просадка
        cumulative_profit = closed_trades['profit'].cumsum()
        running_max = cumulative_profit.expanding().max()
        drawdown = (cumulative_profit - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (упрощенный)
        returns = closed_trades['profit'] / closed_trades['volume']
        sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0
        
        return {
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "total_profit": total_profit,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio
        }
        
    def generate_strategy_analysis(self) -> Dict:
        """Анализ стратегий"""
        if not self.trades_data:
            return {}
            
        df = pd.DataFrame(self.trades_data)
        closed_trades = df[df['status'] == 'CLOSED']
        
        if len(closed_trades) == 0:
            return {}
            
        # Анализ по стратегиям
        strategy_analysis = {}
        for strategy in closed_trades['strategy'].unique():
            strategy_trades = closed_trades[closed_trades['strategy'] == strategy]
            
            if len(strategy_trades) > 0:
                strategy_analysis[strategy] = {
                    "total_trades": len(strategy_trades),
                    "profitable_trades": len(strategy_trades[strategy_trades['profit'] > 0]),
                    "total_profit": strategy_trades['profit'].sum(),
                    "win_rate": (len(strategy_trades[strategy_trades['profit'] > 0]) / len(strategy_trades)) * 100,
                    "avg_profit": strategy_trades['profit'].mean(),
                    "avg_confidence": strategy_trades['confidence'].mean()
                }
                
        return strategy_analysis
        
    def generate_symbol_analysis(self) -> Dict:
        """Анализ по символам"""
        if not self.trades_data:
            return {}
            
        df = pd.DataFrame(self.trades_data)
        closed_trades = df[df['status'] == 'CLOSED']
        
        if len(closed_trades) == 0:
            return {}
            
        # Анализ по символам
        symbol_analysis = {}
        for symbol in closed_trades['symbol'].unique():
            symbol_trades = closed_trades[closed_trades['symbol'] == symbol]
            
            if len(symbol_trades) > 0:
                symbol_analysis[symbol] = {
                    "total_trades": len(symbol_trades),
                    "profitable_trades": len(symbol_trades[symbol_trades['profit'] > 0]),
                    "total_profit": symbol_trades['profit'].sum(),
                    "win_rate": (len(symbol_trades[symbol_trades['profit'] > 0]) / len(symbol_trades)) * 100,
                    "avg_profit": symbol_trades['profit'].mean(),
                    "volume_traded": symbol_trades['volume'].sum()
                }
                
        return symbol_analysis
        
    def generate_signal_analysis(self) -> Dict:
        """Анализ сигналов"""
        if not self.signals_data:
            return {}
            
        df = pd.DataFrame(self.signals_data)
        
        # Анализ точности сигналов
        signal_analysis = {
            "total_signals": len(df),
            "executed_signals": len(df[df['executed'] == True]),
            "avg_confidence": df['confidence'].mean(),
            "high_confidence_signals": len(df[df['confidence'] > 0.8]),
            "signals_by_direction": df['direction'].value_counts().to_dict(),
            "signals_by_model": df['model'].value_counts().to_dict() if 'model' in df.columns else {}
        }
        
        return signal_analysis
        
    def create_performance_chart(self) -> str:
        """Создание графика производительности"""
        if not self.trades_data:
            return ""
            
        df = pd.DataFrame(self.trades_data)
        closed_trades = df[df['status'] == 'CLOSED']
        
        if len(closed_trades) == 0:
            return ""
            
        # Создание графика
        fig = go.Figure()
        
        # Кумулятивная прибыль
        cumulative_profit = closed_trades['profit'].cumsum()
        fig.add_trace(go.Scatter(
            x=closed_trades['close_time'],
            y=cumulative_profit,
            mode='lines+markers',
            name='Кумулятивная прибыль',
            line=dict(color='green', width=2)
        ))
        
        # Настройка графика
        fig.update_layout(
            title="Кумулятивная прибыль",
            xaxis_title="Дата",
            yaxis_title="Прибыль ($)",
            template="plotly_white"
        )
        
        return fig.to_html(include_plotlyjs='cdn')
        
    def create_drawdown_chart(self) -> str:
        """Создание графика просадки"""
        if not self.trades_data:
            return ""
            
        df = pd.DataFrame(self.trades_data)
        closed_trades = df[df['status'] == 'CLOSED']
        
        if len(closed_trades) == 0:
            return ""
            
        # Расчет просадки
        cumulative_profit = closed_trades['profit'].cumsum()
        running_max = cumulative_profit.expanding().max()
        drawdown = (cumulative_profit - running_max) / running_max * 100
        
        # Создание графика
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=closed_trades['close_time'],
            y=drawdown,
            mode='lines',
            name='Просадка',
            line=dict(color='red', width=2),
            fill='tonexty'
        ))
        
        # Настройка графика
        fig.update_layout(
            title="Просадка (%)",
            xaxis_title="Дата",
            yaxis_title="Просадка (%)",
            template="plotly_white"
        )
        
        return fig.to_html(include_plotlyjs='cdn')
        
    def create_win_rate_chart(self) -> str:
        """Создание графика винрейта по стратегиям"""
        strategy_analysis = self.generate_strategy_analysis()
        
        if not strategy_analysis:
            return ""
            
        strategies = list(strategy_analysis.keys())
        win_rates = [strategy_analysis[s]['win_rate'] for s in strategies]
        
        # Создание графика
        fig = go.Figure(data=[
            go.Bar(x=strategies, y=win_rates, marker_color='blue')
        ])
        
        fig.update_layout(
            title="Винрейт по стратегиям",
            xaxis_title="Стратегия",
            yaxis_title="Винрейт (%)",
            template="plotly_white"
        )
        
        return fig.to_html(include_plotlyjs='cdn')
        
    def generate_report(self) -> Dict:
        """Генерация полного отчета"""
        performance_metrics = self.calculate_performance_metrics()
        strategy_analysis = self.generate_strategy_analysis()
        symbol_analysis = self.generate_symbol_analysis()
        signal_analysis = self.generate_signal_analysis()
        
        # Создание графиков
        performance_chart = self.create_performance_chart()
        drawdown_chart = self.create_drawdown_chart()
        win_rate_chart = self.create_win_rate_chart()
        
        return {
            "performance_metrics": performance_metrics,
            "strategy_analysis": strategy_analysis,
            "symbol_analysis": symbol_analysis,
            "signal_analysis": signal_analysis,
            "charts": {
                "performance": performance_chart,
                "drawdown": drawdown_chart,
                "win_rate": win_rate_chart
            },
            "generated_at": datetime.now().isoformat()
        }

# Пример использования
def create_sample_data():
    """Создание тестовых данных"""
    analytics = AnalyticsManager()
    
    # Добавление тестовых сделок
    sample_trades = [
        {
            "symbol": "EURUSD",
            "direction": "BUY",
            "volume": 0.1,
            "open_price": 1.0850,
            "close_price": 1.0870,
            "open_time": datetime.now() - timedelta(days=30),
            "close_time": datetime.now() - timedelta(days=29),
            "profit": 20.0,
            "status": "CLOSED",
            "strategy": "trend_following",
            "confidence": 0.85
        },
        {
            "symbol": "GBPUSD",
            "direction": "SELL",
            "volume": 0.1,
            "open_price": 1.2650,
            "close_price": 1.2630,
            "open_time": datetime.now() - timedelta(days=25),
            "close_time": datetime.now() - timedelta(days=24),
            "profit": 20.0,
            "status": "CLOSED",
            "strategy": "mean_reversion",
            "confidence": 0.75
        },
        {
            "symbol": "EURUSD",
            "direction": "BUY",
            "volume": 0.1,
            "open_price": 1.0860,
            "close_price": 1.0840,
            "open_time": datetime.now() - timedelta(days=20),
            "close_time": datetime.now() - timedelta(days=19),
            "profit": -20.0,
            "status": "CLOSED",
            "strategy": "trend_following",
            "confidence": 0.70
        }
    ]
    
    for trade in sample_trades:
        analytics.add_trade(trade)
        
    # Добавление тестовых сигналов
    sample_signals = [
        {
            "symbol": "EURUSD",
            "direction": "BUY",
            "confidence": 0.85,
            "price": 1.0850,
            "timestamp": datetime.now() - timedelta(hours=2),
            "model": "lstm",
            "executed": True
        },
        {
            "symbol": "GBPUSD",
            "direction": "SELL",
            "confidence": 0.75,
            "price": 1.2650,
            "timestamp": datetime.now() - timedelta(hours=1),
            "model": "xgboost",
            "executed": False
        }
    ]
    
    for signal in sample_signals:
        analytics.add_signal(signal)
        
    return analytics

if __name__ == "__main__":
    # Создание тестовых данных
    analytics = create_sample_data()
    
    # Генерация отчета
    report = analytics.generate_report()
    
    print("=== АНАЛИТИЧЕСКИЙ ОТЧЕТ ===")
    print(f"Метрики производительности: {report['performance_metrics']}")
    print(f"Анализ стратегий: {report['strategy_analysis']}")
    print(f"Анализ символов: {report['symbol_analysis']}")
    print(f"Анализ сигналов: {report['signal_analysis']}")
    print(f"Отчет сгенерирован: {report['generated_at']}")