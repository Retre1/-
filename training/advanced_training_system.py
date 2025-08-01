#!/usr/bin/env python3
"""
🎓 Продвинутая система обучения ForexBot AI
Подробное руководство по обучению моделей
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Импорт наших модулей
from advanced_ai_models import AdvancedFeatureEngineer, AdvancedEnsembleModel
from advanced_backtesting import AdvancedBacktester, PerformanceAnalyzer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Конфигурация обучения"""
    symbol: str
    timeframe: str
    lookback_periods: List[int] = None
    test_size: float = 0.2
    validation_size: float = 0.1
    sequence_length: int = 50
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    min_samples: int = 10000
    feature_importance_threshold: float = 0.01
    class_balance_method: str = 'smote'  # 'smote', 'undersample', 'oversample'
    
    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [20, 50, 100, 200]

class AdvancedTrainingSystem:
    """Продвинутая система обучения AI моделей"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.training_config = TrainingConfig(
            symbol=config.get('symbol', 'EURUSD'),
            timeframe=config.get('timeframe', 'H1')
        )
        
        # Директории
        self.data_dir = Path('data/market_data')
        self.models_dir = Path('data/models')
        self.reports_dir = Path('data/reports')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Компоненты
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ensemble_model = AdvancedEnsembleModel()
        self.backtester = AdvancedBacktester()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Результаты обучения
        self.training_results = {}
        self.model_performance = {}
        
    def load_market_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Загрузка рыночных данных"""
        try:
            # Поиск файла данных
            symbol_dir = self.data_dir / symbol
            pattern = f"{symbol}_{timeframe}_*.csv"
            
            files = list(symbol_dir.glob(pattern))
            if not files:
                logger.error(f"❌ Файлы данных не найдены для {symbol} {timeframe}")
                return None
            
            # Загрузка самого свежего файла
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            logger.info(f"📊 Загрузка данных из: {latest_file}")
            
            df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            logger.info(f"✅ Загружено {len(df)} записей для {symbol} {timeframe}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных: {e}")
            return None
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Подготовка данных для обучения"""
        try:
            logger.info("🔧 Подготовка данных для обучения...")
            
            # Создание признаков
            df_features = self.feature_engineer.create_technical_indicators(df.copy())
            df_features = self.feature_engineer.create_advanced_features(df_features)
            
            # Удаление NaN значений
            df_features = df_features.dropna()
            
            if len(df_features) < self.training_config.min_samples:
                logger.warning(f"⚠️ Недостаточно данных: {len(df_features)} < {self.training_config.min_samples}")
                return None, None, None
            
            # Создание целевой переменной
            target = self._create_target_variable(df_features['close'])
            
            # Подготовка признаков
            feature_columns = [col for col in df_features.columns 
                             if col not in ['symbol', 'timeframe', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
            
            features = df_features[feature_columns].values
            target = target[~np.isnan(features).any(axis=1)]
            features = features[~np.isnan(features).any(axis=1)]
            
            logger.info(f"✅ Подготовлено {len(features)} образцов с {len(feature_columns)} признаками")
            
            return features, target, feature_columns
            
        except Exception as e:
            logger.error(f"❌ Ошибка подготовки данных: {e}")
            return None, None, None
    
    def _create_target_variable(self, prices: pd.Series) -> np.ndarray:
        """Создание целевой переменной для классификации"""
        try:
            # Расчет будущих доходностей
            future_returns = prices.shift(-1) / prices - 1
            
            # Определение порогов для классификации
            threshold_buy = 0.001   # 0.1% рост
            threshold_sell = -0.001  # 0.1% падение
            
            # Создание классов
            target = np.zeros(len(prices))
            target[future_returns > threshold_buy] = 1    # BUY
            target[future_returns < threshold_sell] = 2   # SELL
            # 0 = HOLD (по умолчанию)
            
            # Удаление последнего элемента (нет будущих данных)
            target = target[:-1]
            
            return target
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания целевой переменной: {e}")
            return np.zeros(len(prices) - 1)
    
    def balance_classes(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Балансировка классов"""
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import RandomUnderSampler
            from imblearn.over_sampling import RandomOverSampler
            
            method = self.training_config.class_balance_method
            
            if method == 'smote':
                smote = SMOTE(random_state=42)
                features_balanced, target_balanced = smote.fit_resample(features, target)
            elif method == 'undersample':
                rus = RandomUnderSampler(random_state=42)
                features_balanced, target_balanced = rus.fit_resample(features, target)
            elif method == 'oversample':
                ros = RandomOverSampler(random_state=42)
                features_balanced, target_balanced = ros.fit_resample(features, target)
            else:
                features_balanced, target_balanced = features, target
            
            logger.info(f"✅ Балансировка классов: {method}")
            logger.info(f"📊 Распределение классов: {np.bincount(target_balanced)}")
            
            return features_balanced, target_balanced
            
        except Exception as e:
            logger.error(f"❌ Ошибка балансировки классов: {e}")
            return features, target
    
    def split_data(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Разделение данных на обучающую и тестовую выборки"""
        try:
            # Временное разделение (TimeSeriesSplit)
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Используем последний сплит для финального разделения
            for train_idx, test_idx in tscv.split(features):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = target[train_idx], target[test_idx]
            
            # Дополнительное разделение обучающей выборки
            split_point = int(len(X_train) * (1 - self.training_config.validation_size))
            X_train_final = X_train[:split_point]
            y_train_final = y_train[:split_point]
            X_val = X_train[split_point:]
            y_val = y_train[split_point:]
            
            logger.info(f"📊 Разделение данных:")
            logger.info(f"   Обучающая выборка: {len(X_train_final)}")
            logger.info(f"   Валидационная выборка: {len(X_val)}")
            logger.info(f"   Тестовая выборка: {len(X_test)}")
            
            return X_train_final, X_val, X_test, y_test
            
        except Exception as e:
            logger.error(f"❌ Ошибка разделения данных: {e}")
            return None, None, None, None
    
    def train_models(self, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> Dict:
        """Обучение всех моделей"""
        try:
            logger.info("🎓 Начало обучения моделей...")
            
            training_results = {}
            
            # Обучение каждой модели
            for model_name, model in self.ensemble_model.models.items():
                logger.info(f"🔄 Обучение модели: {model_name}")
                
                try:
                    if hasattr(model, 'fit'):
                        # Стандартные модели (XGBoost, LightGBM, RandomForest)
                        model.fit(X_train, y_train)
                        
                        # Предсказания
                        y_pred_train = model.predict(X_train)
                        y_pred_val = model.predict(X_val)
                        
                        # Оценка качества
                        train_accuracy = np.mean(y_pred_train == y_train)
                        val_accuracy = np.mean(y_pred_val == y_val)
                        
                        training_results[model_name] = {
                            'model': model,
                            'train_accuracy': train_accuracy,
                            'val_accuracy': val_accuracy,
                            'predictions_train': y_pred_train,
                            'predictions_val': y_pred_val
                        }
                        
                        logger.info(f"✅ {model_name}: Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}")
                        
                    elif hasattr(model, 'create_model'):
                        # Нейронные сети (LSTM, Transformer)
                        model_obj = model.create_model()
                        
                        # Подготовка данных для нейронных сетей
                        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)
                        X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
                        
                        # Обучение
                        history = model_obj.fit(
                            X_train_seq, y_train_seq,
                            validation_data=(X_val_seq, y_val_seq),
                            epochs=self.training_config.epochs,
                            batch_size=self.training_config.batch_size,
                            verbose=0
                        )
                        
                        # Предсказания
                        y_pred_train = model_obj.predict(X_train_seq)
                        y_pred_val = model_obj.predict(X_val_seq)
                        
                        # Оценка качества
                        train_accuracy = np.mean(np.argmax(y_pred_train, axis=1) == y_train)
                        val_accuracy = np.mean(np.argmax(y_pred_val, axis=1) == y_val)
                        
                        training_results[model_name] = {
                            'model': model_obj,
                            'history': history,
                            'train_accuracy': train_accuracy,
                            'val_accuracy': val_accuracy,
                            'predictions_train': y_pred_train,
                            'predictions_val': y_pred_val
                        }
                        
                        logger.info(f"✅ {model_name}: Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}")
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка обучения {model_name}: {e}")
                    continue
            
            self.training_results = training_results
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения моделей: {e}")
            return {}
    
    def _prepare_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка последовательностей для нейронных сетей"""
        try:
            sequence_length = self.training_config.sequence_length
            
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, len(features)):
                X_sequences.append(features[i-sequence_length:i])
                y_sequences.append(target[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            # One-hot encoding для целевой переменной
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_sequences)
            
            from tensorflow.keras.utils import to_categorical
            y_categorical = to_categorical(y_encoded, num_classes=3)
            
            return X_sequences, y_categorical
            
        except Exception as e:
            logger.error(f"❌ Ошибка подготовки последовательностей: {e}")
            return np.array([]), np.array([])
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Оценка качества моделей"""
        try:
            logger.info("📊 Оценка качества моделей...")
            
            evaluation_results = {}
            
            for model_name, result in self.training_results.items():
                model = result['model']
                
                try:
                    # Предсказания на тестовой выборке
                    if hasattr(model, 'predict'):
                        y_pred_test = model.predict(X_test)
                    else:
                        # Для нейронных сетей
                        X_test_seq, _ = self._prepare_sequences(X_test, y_test)
                        y_pred_test = model.predict(X_test_seq)
                        y_pred_test = np.argmax(y_pred_test, axis=1)
                    
                    # Метрики качества
                    accuracy = np.mean(y_pred_test == y_test)
                    
                    # Детальная классификация
                    from sklearn.metrics import classification_report, confusion_matrix
                    class_report = classification_report(y_test, y_pred_test, output_dict=True)
                    conf_matrix = confusion_matrix(y_test, y_pred_test)
                    
                    evaluation_results[model_name] = {
                        'accuracy': accuracy,
                        'classification_report': class_report,
                        'confusion_matrix': conf_matrix,
                        'predictions': y_pred_test
                    }
                    
                    logger.info(f"📊 {model_name}: Test Accuracy = {accuracy:.4f}")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка оценки {model_name}: {e}")
                    continue
            
            self.model_performance = evaluation_results
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Ошибка оценки моделей: {e}")
            return {}
    
    def save_models(self, symbol: str, timeframe: str):
        """Сохранение обученных моделей"""
        try:
            logger.info("💾 Сохранение моделей...")
            
            model_dir = self.models_dir / symbol / timeframe
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Сохранение каждой модели
            for model_name, result in self.training_results.items():
                model = result['model']
                
                try:
                    if hasattr(model, 'save'):
                        # Нейронные сети
                        model_path = model_dir / f"{model_name}.h5"
                        model.save(str(model_path))
                    else:
                        # Стандартные модели
                        model_path = model_dir / f"{model_name}.pkl"
                        with open(model_path, 'wb') as f:
                            pickle.dump(model, f)
                    
                    logger.info(f"💾 Сохранена модель: {model_path}")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка сохранения {model_name}: {e}")
                    continue
            
            # Сохранение метаданных
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'training_date': datetime.now().isoformat(),
                'training_config': self.training_config.__dict__,
                'model_performance': self.model_performance,
                'feature_columns': getattr(self.feature_engineer, 'feature_names', [])
            }
            
            metadata_path = model_dir / 'training_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Сохранены метаданные: {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения моделей: {e}")
    
    def generate_training_report(self, symbol: str, timeframe: str):
        """Генерация отчета об обучении"""
        try:
            logger.info("📊 Генерация отчета об обучении...")
            
            report_dir = self.reports_dir / symbol / timeframe
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Создание отчета
            report = {
                'training_info': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'training_date': datetime.now().isoformat(),
                    'config': self.training_config.__dict__
                },
                'model_performance': self.model_performance,
                'training_results': {
                    name: {
                        'train_accuracy': result['train_accuracy'],
                        'val_accuracy': result['val_accuracy']
                    }
                    for name, result in self.training_results.items()
                }
            }
            
            # Сохранение отчета
            report_path = report_dir / 'training_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Создание визуализаций
            self._create_training_visualizations(report_dir)
            
            logger.info(f"📊 Отчет сохранен: {report_path}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации отчета: {e}")
    
    def _create_training_visualizations(self, report_dir: Path):
        """Создание визуализаций для отчета"""
        try:
            # График точности моделей
            model_names = list(self.model_performance.keys())
            accuracies = [self.model_performance[name]['accuracy'] for name in model_names]
            
            plt.figure(figsize=(12, 8))
            
            # График точности
            plt.subplot(2, 2, 1)
            plt.bar(model_names, accuracies)
            plt.title('Точность моделей на тестовой выборке')
            plt.ylabel('Точность')
            plt.xticks(rotation=45)
            
            # График обучения (для нейронных сетей)
            plt.subplot(2, 2, 2)
            for model_name, result in self.training_results.items():
                if 'history' in result:
                    history = result['history']
                    plt.plot(history.history['accuracy'], label=f'{model_name} (train)')
                    plt.plot(history.history['val_accuracy'], label=f'{model_name} (val)')
            
            plt.title('Кривые обучения')
            plt.xlabel('Эпоха')
            plt.ylabel('Точность')
            plt.legend()
            
            # Матрицы ошибок
            for i, model_name in enumerate(model_names[:2]):
                plt.subplot(2, 2, 3 + i)
                conf_matrix = self.model_performance[model_name]['confusion_matrix']
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Матрица ошибок: {model_name}')
                plt.ylabel('Истинные значения')
                plt.xlabel('Предсказанные значения')
            
            plt.tight_layout()
            plt.savefig(report_dir / 'training_visualizations.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Визуализации сохранены: {report_dir / 'training_visualizations.png'}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания визуализаций: {e}")
    
    def run_complete_training(self, symbol: str, timeframe: str) -> bool:
        """Полный цикл обучения"""
        try:
            logger.info(f"🚀 Начало полного обучения для {symbol} {timeframe}")
            
            # Загрузка данных
            df = self.load_market_data(symbol, timeframe)
            if df is None:
                return False
            
            # Подготовка данных
            features, target, feature_columns = self.prepare_training_data(df)
            if features is None:
                return False
            
            # Балансировка классов
            features_balanced, target_balanced = self.balance_classes(features, target)
            
            # Разделение данных
            X_train, X_val, X_test, y_test = self.split_data(features_balanced, target_balanced)
            if X_train is None:
                return False
            
            # Обучение моделей
            training_results = self.train_models(X_train, X_val, y_train, y_val)
            if not training_results:
                return False
            
            # Оценка моделей
            evaluation_results = self.evaluate_models(X_test, y_test)
            if not evaluation_results:
                return False
            
            # Сохранение моделей
            self.save_models(symbol, timeframe)
            
            # Генерация отчета
            self.generate_training_report(symbol, timeframe)
            
            logger.info("✅ Обучение завершено успешно!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка полного обучения: {e}")
            return False

def main():
    """Основная функция обучения"""
    
    # Загрузка конфигурации
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Создание системы обучения
    training_system = AdvancedTrainingSystem(config)
    
    # Обучение для всех символов и таймфреймов
    symbols = config.get('mt5', {}).get('symbols', ['EURUSD'])
    timeframes = config.get('ai', {}).get('timeframes', ['H1'])
    
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"🎓 Обучение для {symbol} {timeframe}")
            
            success = training_system.run_complete_training(symbol, timeframe)
            
            if success:
                logger.info(f"✅ Обучение {symbol} {timeframe} завершено успешно")
            else:
                logger.error(f"❌ Ошибка обучения {symbol} {timeframe}")

if __name__ == "__main__":
    main()