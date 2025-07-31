#!/usr/bin/env python3
"""
Monitoring System for ForexBot AI
Система мониторинга с Prometheus и Grafana
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.exposition import start_http_server
from fastapi import FastAPI, Request
from fastapi.responses import Response
import asyncio
import json

class ForexBotMetrics:
    """Метрики для ForexBot AI"""
    
    def __init__(self):
        # Счетчики
        self.trades_total = Counter('trades_total', 'Total number of trades', ['direction', 'symbol'])
        self.signals_generated = Counter('signals_generated', 'Total number of signals generated', ['model', 'direction'])
        self.errors_total = Counter('errors_total', 'Total number of errors', ['module', 'error_type'])
        self.api_requests_total = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
        
        # Гистограммы
        self.trade_duration = Histogram('trade_duration_seconds', 'Trade duration in seconds', ['symbol'])
        self.prediction_latency = Histogram('prediction_latency_seconds', 'AI model prediction latency', ['model'])
        self.api_response_time = Histogram('api_response_time_seconds', 'API response time', ['endpoint'])
        
        # Гейджи
        self.total_profit = Gauge('total_profit', 'Total profit/loss')
        self.current_drawdown = Gauge('current_drawdown_percent', 'Current drawdown percentage')
        self.active_positions = Gauge('active_positions', 'Number of active positions')
        self.capital_balance = Gauge('capital_balance', 'Current capital balance')
        self.win_rate = Gauge('win_rate_percent', 'Current win rate percentage')
        self.sharpe_ratio = Gauge('sharpe_ratio', 'Current Sharpe ratio')
        
        # Системные метрики
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage')
        self.disk_usage = Gauge('disk_usage_percent', 'Disk usage percentage')
        self.network_io = Gauge('network_io_bytes', 'Network I/O in bytes', ['direction'])
        
        # Сводки
        self.model_accuracy = Summary('model_accuracy', 'Model accuracy', ['model'])
        self.profit_per_trade = Summary('profit_per_trade', 'Average profit per trade')
        
        # Дополнительные метрики
        self.ai_models_loaded = Gauge('ai_models_loaded', 'Number of loaded AI models')
        self.backtest_runs = Counter('backtest_runs_total', 'Total backtest runs')
        self.notifications_sent = Counter('notifications_sent_total', 'Total notifications sent', ['type'])
        
    def record_trade(self, direction: str, symbol: str, duration: float, profit: float):
        """Запись метрик сделки"""
        self.trades_total.labels(direction=direction, symbol=symbol).inc()
        self.trade_duration.labels(symbol=symbol).observe(duration)
        self.profit_per_trade.observe(profit)
        
    def record_signal(self, model: str, direction: str, latency: float):
        """Запись метрик сигнала"""
        self.signals_generated.labels(model=model, direction=direction).inc()
        self.prediction_latency.labels(model=model).observe(latency)
        
    def record_error(self, module: str, error_type: str):
        """Запись метрик ошибок"""
        self.errors_total.labels(module=module, error_type=error_type).inc()
        
    def record_api_request(self, endpoint: str, method: str, response_time: float):
        """Запись метрик API запросов"""
        self.api_requests_total.labels(endpoint=endpoint, method=method).inc()
        self.api_response_time.labels(endpoint=endpoint).observe(response_time)
        
    def update_performance_metrics(self, stats: Dict):
        """Обновление метрик производительности"""
        self.total_profit.set(stats.get('total_profit', 0))
        self.current_drawdown.set(stats.get('current_drawdown', 0))
        self.active_positions.set(stats.get('active_positions', 0))
        self.capital_balance.set(stats.get('capital_balance', 0))
        self.win_rate.set(stats.get('win_rate', 0))
        self.sharpe_ratio.set(stats.get('sharpe_ratio', 0))
        
    def update_system_metrics(self):
        """Обновление системных метрик"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.set(cpu_percent)
        
        # Memory
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.percent)
        
        # Disk
        disk = psutil.disk_usage('/')
        self.disk_usage.set(disk.percent)
        
        # Network
        network = psutil.net_io_counters()
        self.network_io.labels(direction='bytes_sent').set(network.bytes_sent)
        self.network_io.labels(direction='bytes_recv').set(network.bytes_recv)
        
    def update_model_accuracy(self, model: str, accuracy: float):
        """Обновление точности модели"""
        self.model_accuracy.labels(model=model).observe(accuracy)
        
    def record_backtest_run(self, duration: float):
        """Запись метрик backtesting"""
        self.backtest_runs.inc()
        
    def record_notification(self, notification_type: str):
        """Запись метрик уведомлений"""
        self.notifications_sent.labels(type=notification_type).inc()

class SystemMonitor:
    """Монитор системы"""
    
    def __init__(self, metrics: ForexBotMetrics):
        self.metrics = metrics
        self.running = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Запуск мониторинга"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Цикл мониторинга"""
        while self.running:
            try:
                self.metrics.update_system_metrics()
                time.sleep(30)  # Обновление каждые 30 секунд
            except Exception as e:
                print(f"Ошибка мониторинга: {e}")
                time.sleep(60)

class PrometheusMiddleware:
    """Middleware для FastAPI для сбора метрик"""
    
    def __init__(self, metrics: ForexBotMetrics):
        self.metrics = metrics
        
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        # Обработка запроса
        response = await call_next(request)
        
        # Запись метрик
        duration = time.time() - start_time
        endpoint = request.url.path
        method = request.method
        
        self.metrics.record_api_request(endpoint, method, duration)
        
        return response

class AlertManager:
    """Менеджер алертов"""
    
    def __init__(self, metrics: ForexBotMetrics):
        self.metrics = metrics
        self.alert_rules = []
        self.alert_history = []
        
    def add_alert_rule(self, rule: Dict):
        """Добавление правила алерта"""
        self.alert_rules.append(rule)
        
    def check_alerts(self):
        """Проверка алертов"""
        current_time = datetime.now()
        
        for rule in self.alert_rules:
            if self._evaluate_rule(rule):
                alert = {
                    'rule': rule,
                    'timestamp': current_time,
                    'severity': rule.get('severity', 'warning')
                }
                self.alert_history.append(alert)
                self._trigger_alert(alert)
                
    def _evaluate_rule(self, rule: Dict) -> bool:
        """Оценка правила алерта"""
        rule_type = rule.get('type')
        
        if rule_type == 'cpu_high':
            return psutil.cpu_percent() > rule.get('threshold', 80)
        elif rule_type == 'memory_high':
            return psutil.virtual_memory().percent > rule.get('threshold', 80)
        elif rule_type == 'disk_high':
            return psutil.disk_usage('/').percent > rule.get('threshold', 90)
        elif rule_type == 'profit_low':
            return self.metrics.total_profit._value.get() < rule.get('threshold', -1000)
        elif rule_type == 'drawdown_high':
            return self.metrics.current_drawdown._value.get() > rule.get('threshold', 20)
            
        return False
        
    def _trigger_alert(self, alert: Dict):
        """Срабатывание алерта"""
        print(f"🚨 АЛЕРТ: {alert['rule'].get('name', 'Unknown')} - {alert['severity']}")
        # Здесь можно добавить отправку уведомлений
        
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Получение истории алертов"""
        return self.alert_history[-limit:]

# Интеграция с FastAPI
def setup_monitoring(app: FastAPI, metrics: ForexBotMetrics):
    """Настройка мониторинга для FastAPI"""
    
    # Middleware для сбора метрик
    app.add_middleware(PrometheusMiddleware, metrics=metrics)
    
    # Endpoint для метрик Prometheus
    @app.get("/metrics")
    async def get_metrics():
        """Получение метрик Prometheus"""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    # Endpoint для системной информации
    @app.get("/api/monitoring/system")
    async def get_system_info():
        """Получение системной информации"""
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_io": dict(psutil.net_io_counters()._asdict()),
            "uptime": time.time() - psutil.boot_time()
        }
    
    # Endpoint для метрик приложения
    @app.get("/api/monitoring/metrics")
    async def get_app_metrics():
        """Получение метрик приложения"""
        return {
            "trades_total": metrics.trades_total._value.get(),
            "total_profit": metrics.total_profit._value.get(),
            "current_drawdown": metrics.current_drawdown._value.get(),
            "active_positions": metrics.active_positions._value.get(),
            "win_rate": metrics.win_rate._value.get(),
            "sharpe_ratio": metrics.sharpe_ratio._value.get()
        }

# Пример использования
if __name__ == "__main__":
    # Создание метрик
    metrics = ForexBotMetrics()
    
    # Создание монитора системы
    monitor = SystemMonitor(metrics)
    monitor.start_monitoring()
    
    # Создание менеджера алертов
    alert_manager = AlertManager(metrics)
    
    # Добавление правил алертов
    alert_manager.add_alert_rule({
        'name': 'High CPU Usage',
        'type': 'cpu_high',
        'threshold': 80,
        'severity': 'warning'
    })
    
    alert_manager.add_alert_rule({
        'name': 'High Memory Usage',
        'type': 'memory_high',
        'threshold': 85,
        'severity': 'critical'
    })
    
    alert_manager.add_alert_rule({
        'name': 'High Drawdown',
        'type': 'drawdown_high',
        'threshold': 15,
        'severity': 'warning'
    })
    
    # Запуск HTTP сервера для метрик
    start_http_server(8001)
    
    print("🚀 Система мониторинга запущена на порту 8001")
    print("📊 Метрики доступны по адресу: http://localhost:8001/metrics")
    
    try:
        while True:
            alert_manager.check_alerts()
            time.sleep(60)  # Проверка алертов каждую минуту
    except KeyboardInterrupt:
        print("\n🛑 Остановка системы мониторинга...")
        monitor.stop_monitoring()