#!/usr/bin/env python3
"""
Advanced Backtesting System for AI Models
Продвинутая система backtesting для AI моделей
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedBacktester:
    """Продвинутый backtester для AI моделей"""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
        
    def reset(self):
        """Сброс состояния backtester"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.max_drawdown = 0
        self.peak_capital = self.initial_capital
        
    def run_backtest(self, df: pd.DataFrame, predictions: np.ndarray, 
                    confidence_threshold: float = 0.6) -> Dict:
        """Запуск backtest"""
        self.reset()
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_time = df.index[i]
            
            # Получение предсказания
            if i < len(predictions):
                pred = predictions[i]
                signal = self._get_signal(pred, confidence_threshold)
            else:
                signal = 'HOLD'
                
            # Выполнение торговых операций
            self._execute_trade(signal, current_price, current_time)
            
            # Обновление кривой доходности
            self._update_equity_curve(current_price)
            
        return self._generate_results()
        
    def _get_signal(self, prediction: np.ndarray, confidence_threshold: float) -> str:
        """Получение торгового сигнала"""
        max_prob = np.max(prediction)
        max_class = np.argmax(prediction)
        
        if max_prob < confidence_threshold:
            return 'HOLD'
            
        if max_class == 0:
            return 'HOLD'
        elif max_class == 1:
            return 'BUY'
        else:
            return 'SELL'
            
    def _execute_trade(self, signal: str, price: float, timestamp: datetime):
        """Выполнение торговой операции"""
        if signal == 'BUY' and 'EURUSD' not in self.positions:
            # Открытие длинной позиции
            position_size = self.capital * 0.95 / price  # 95% капитала
            cost = position_size * price * (1 + self.commission)
            
            if cost <= self.capital:
                self.positions['EURUSD'] = {
                    'type': 'LONG',
                    'size': position_size,
                    'entry_price': price,
                    'entry_time': timestamp
                }
                self.capital -= cost
                
        elif signal == 'SELL' and 'EURUSD' in self.positions:
            # Закрытие длинной позиции
            position = self.positions['EURUSD']
            if position['type'] == 'LONG':
                revenue = position['size'] * price * (1 - self.commission)
                profit = revenue - (position['size'] * position['entry_price'] * (1 + self.commission))
                
                self.trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'entry_price': position['entry_price'],
                    'exit_price': price,
                    'size': position['size'],
                    'profit': profit,
                    'return': profit / (position['size'] * position['entry_price'])
                })
                
                self.capital += revenue
                del self.positions['EURUSD']
                
    def _update_equity_curve(self, current_price: float):
        """Обновление кривой доходности"""
        # Расчет текущего капитала
        current_equity = self.capital
        
        for symbol, position in self.positions.items():
            if position['type'] == 'LONG':
                current_equity += position['size'] * current_price
                
        self.equity_curve.append(current_equity)
        
        # Обновление максимального капитала и просадки
        if current_equity > self.peak_capital:
            self.peak_capital = current_equity
            
        current_drawdown = (self.peak_capital - current_equity) / self.peak_capital
        self.drawdown_curve.append(current_drawdown)
        
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
            
    def _generate_results(self) -> Dict:
        """Генерация результатов backtest"""
        if not self.trades:
            return self._empty_results()
            
        # Базовые метрики
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['profit'] > 0])
        losing_trades = total_trades - winning_trades
        
        total_profit = sum(t['profit'] for t in self.trades)
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Продвинутые метрики
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t['profit'] for t in self.trades if t['profit'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['profit'] for t in self.trades if t['profit'] < 0]) if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Sharpe Ratio
        returns = [t['return'] for t in self.trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Maximum Drawdown
        max_dd = self.max_drawdown
        
        # Calmar Ratio
        calmar_ratio = total_return / max_dd if max_dd > 0 else 0
        
        # Sortino Ratio
        negative_returns = [r for r in returns if r < 0]
        downside_deviation = np.std(negative_returns) if negative_returns else 0
        sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'equity_curve': self.equity_curve,
            'drawdown_curve': self.drawdown_curve,
            'trades': self.trades
        }
        
    def _empty_results(self) -> Dict:
        """Результаты для пустого backtest"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'total_return': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0,
            'sortino_ratio': 0,
            'equity_curve': [self.initial_capital],
            'drawdown_curve': [0],
            'trades': []
        }

class ModelOptimizer:
    """Оптимизатор моделей"""
    
    def __init__(self, ensemble_model):
        self.ensemble_model = ensemble_model
        self.best_params = {}
        self.optimization_history = []
        
    def optimize_hyperparameters(self, df: pd.DataFrame, target: pd.Series) -> Dict:
        """Оптимизация гиперпараметров"""
        print("Начинаем оптимизацию гиперпараметров...")
        
        # Оптимизация для каждой модели
        for model_name in self.ensemble_model.models.keys():
            print(f"Оптимизация {model_name}...")
            
            if model_name == 'xgboost':
                self._optimize_xgboost(df, target)
            elif model_name == 'lightgbm':
                self._optimize_lightgbm(df, target)
            elif model_name == 'random_forest':
                self._optimize_random_forest(df, target)
                
        return self.best_params
        
    def _optimize_xgboost(self, df: pd.DataFrame, target: pd.Series):
        """Оптимизация XGBoost"""
        features, _ = self.ensemble_model.feature_engineer.prepare_features(df)
        y = self.ensemble_model._create_target_variable(target)
        
        # Параметры для оптимизации
        param_grid = {
            'n_estimators': [500, 1000, 1500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Grid search
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import make_scorer, accuracy_score
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(features, y)
        
        self.best_params['xgboost'] = grid_search.best_params_
        self.ensemble_model.models['xgboost'] = grid_search.best_estimator_
        
        print(f"Лучшие параметры XGBoost: {grid_search.best_params_}")
        
    def _optimize_lightgbm(self, df: pd.DataFrame, target: pd.Series):
        """Оптимизация LightGBM"""
        features, _ = self.ensemble_model.feature_engineer.prepare_features(df)
        y = self.ensemble_model._create_target_variable(target)
        
        # Параметры для оптимизации
        param_grid = {
            'n_estimators': [500, 1000, 1500],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Grid search
        from sklearn.model_selection import GridSearchCV
        
        lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        grid_search = GridSearchCV(
            lgb_model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(features, y)
        
        self.best_params['lightgbm'] = grid_search.best_params_
        self.ensemble_model.models['lightgbm'] = grid_search.best_estimator_
        
        print(f"Лучшие параметры LightGBM: {grid_search.best_params_}")
        
    def _optimize_random_forest(self, df: pd.DataFrame, target: pd.Series):
        """Оптимизация Random Forest"""
        features, _ = self.ensemble_model.feature_engineer.prepare_features(df)
        y = self.ensemble_model._create_target_variable(target)
        
        # Параметры для оптимизации
        param_grid = {
            'n_estimators': [300, 500, 700],
            'max_depth': [8, 10, 12],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        
        rf_model = RandomForestClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            rf_model,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(features, y)
        
        self.best_params['random_forest'] = grid_search.best_params_
        self.ensemble_model.models['random_forest'] = grid_search.best_estimator_
        
        print(f"Лучшие параметры Random Forest: {grid_search.best_params_}")

class PerformanceAnalyzer:
    """Анализатор производительности"""
    
    def __init__(self):
        self.results = {}
        
    def analyze_model_performance(self, backtest_results: Dict, model_name: str):
        """Анализ производительности модели"""
        self.results[model_name] = backtest_results
        
    def compare_models(self) -> pd.DataFrame:
        """Сравнение моделей"""
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Total Return (%)': results['total_return'] * 100,
                'Sharpe Ratio': results['sharpe_ratio'],
                'Max Drawdown (%)': results['max_drawdown'] * 100,
                'Win Rate (%)': results['win_rate'] * 100,
                'Profit Factor': results['profit_factor'],
                'Total Trades': results['total_trades'],
                'Calmar Ratio': results['calmar_ratio'],
                'Sortino Ratio': results['sortino_ratio']
            })
            
        return pd.DataFrame(comparison_data)
        
    def plot_equity_curves(self):
        """Построение графиков кривых доходности"""
        plt.figure(figsize=(15, 10))
        
        # Кривые доходности
        plt.subplot(2, 2, 1)
        for model_name, results in self.results.items():
            plt.plot(results['equity_curve'], label=model_name)
        plt.title('Equity Curves')
        plt.xlabel('Time')
        plt.ylabel('Capital')
        plt.legend()
        plt.grid(True)
        
        # Кривые просадки
        plt.subplot(2, 2, 2)
        for model_name, results in self.results.items():
            plt.plot(results['drawdown_curve'], label=model_name)
        plt.title('Drawdown Curves')
        plt.xlabel('Time')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)
        
        # Распределение прибыли
        plt.subplot(2, 2, 3)
        for model_name, results in self.results.items():
            profits = [t['profit'] for t in results['trades']]
            plt.hist(profits, alpha=0.5, label=model_name, bins=20)
        plt.title('Profit Distribution')
        plt.xlabel('Profit')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Сравнение метрик
        plt.subplot(2, 2, 4)
        comparison_df = self.compare_models()
        metrics = ['Total Return (%)', 'Sharpe Ratio', 'Win Rate (%)']
        
        x = np.arange(len(comparison_df))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, comparison_df[metric], width, label=metric)
            
        plt.xlabel('Models')
        plt.ylabel('Value')
        plt.title('Model Comparison')
        plt.xticks(x + width, comparison_df['Model'])
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def generate_report(self) -> str:
        """Генерация отчета"""
        comparison_df = self.compare_models()
        
        report = "=== ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ ===\n\n"
        
        # Лучшая модель по каждому критерию
        best_return = comparison_df.loc[comparison_df['Total Return (%)'].idxmax()]
        best_sharpe = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax()]
        best_drawdown = comparison_df.loc[comparison_df['Max Drawdown (%)'].idxmin()]
        
        report += f"Лучшая модель по доходности: {best_return['Model']} ({best_return['Total Return (%)']:.2f}%)\n"
        report += f"Лучшая модель по Sharpe Ratio: {best_sharpe['Model']} ({best_sharpe['Sharpe Ratio']:.2f})\n"
        report += f"Лучшая модель по просадке: {best_drawdown['Model']} ({best_drawdown['Max Drawdown (%)']:.2f}%)\n\n"
        
        # Детальная таблица
        report += "Детальное сравнение:\n"
        report += comparison_df.to_string(index=False)
        
        return report

# Интеграция с основными моделями
def integrate_advanced_models_with_backtesting():
    """Интеграция продвинутых моделей с backtesting"""
    from advanced_ai_models import create_advanced_models
    
    # Создание тестовых данных
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='1H')
    np.random.seed(42)
    
    # Симуляция рыночных данных
    price = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)
    volume = np.random.randint(1000, 10000, len(dates))
    
    df = pd.DataFrame({
        'open': price * (1 + np.random.randn(len(dates)) * 0.001),
        'high': price * (1 + abs(np.random.randn(len(dates)) * 0.002)),
        'low': price * (1 - abs(np.random.randn(len(dates)) * 0.002)),
        'close': price,
        'volume': volume
    }, index=dates)
    
    # Создание моделей
    ensemble = create_advanced_models()
    
    # Оптимизация
    optimizer = ModelOptimizer(ensemble)
    optimizer.optimize_hyperparameters(df, df['close'])
    
    # Обучение моделей
    ensemble.train_models(df, df['close'])
    
    # Backtesting
    backtester = AdvancedBacktester(initial_capital=10000)
    analyzer = PerformanceAnalyzer()
    
    # Тестирование каждой модели отдельно
    models_to_test = ['xgboost', 'lightgbm', 'random_forest']
    
    for model_name in models_to_test:
        print(f"Тестирование {model_name}...")
        
        # Получение предсказаний от одной модели
        features, _ = ensemble.feature_engineer.prepare_features(df)
        predictions = ensemble.models[model_name].predict_proba(features)
        
        # Backtesting
        results = backtester.run_backtest(df, predictions)
        analyzer.analyze_model_performance(results, model_name)
        
    # Тестирование ансамбля
    print("Тестирование ансамбля...")
    ensemble_predictions = ensemble.predict_ensemble(df)
    ensemble_results = backtester.run_backtest(df, ensemble_predictions['ensemble'])
    analyzer.analyze_model_performance(ensemble_results, 'Ensemble')
    
    # Генерация отчета
    report = analyzer.generate_report()
    print(report)
    
    # Построение графиков
    analyzer.plot_equity_curves()
    
    return analyzer

if __name__ == "__main__":
    # Запуск полного тестирования
    analyzer = integrate_advanced_models_with_backtesting()
    
    print("=== ПРОДВИНУТОЕ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО ===")
    print("Результаты сохранены в анализаторе производительности")