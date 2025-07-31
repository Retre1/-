#!/usr/bin/env python3
"""
Scaling System for ForexBot AI
Система горизонтального масштабирования для распределенной работы
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import logging
import psutil
import os

logger = logging.getLogger(__name__)

class NodeManager:
    """Менеджер узлов для распределенной системы"""
    
    def __init__(self, node_id: str = None, host: str = 'localhost', port: int = 8000):
        self.node_id = node_id or str(uuid.uuid4())
        self.host = host
        self.port = port
        self.status = "active"
        self.start_time = datetime.now()
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.active_tasks = 0
        self.max_tasks = 10
        
        # Redis для координации
        self.redis = None
        self.nodes = {}
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        
    async def initialize(self, redis_url: str = "redis://localhost"):
        """Инициализация узла"""
        self.redis = await aioredis.from_url(redis_url)
        
        # Регистрация узла
        await self._register_node()
        
        # Запуск мониторинга
        asyncio.create_task(self._monitor_node())
        
        # Запуск обработки задач
        asyncio.create_task(self._process_tasks())
        
        logger.info(f"Узел {self.node_id} инициализирован")
    
    async def _register_node(self):
        """Регистрация узла в системе"""
        node_info = {
            'node_id': self.node_id,
            'host': self.host,
            'port': self.port,
            'status': self.status,
            'start_time': self.start_time.isoformat(),
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'active_tasks': self.active_tasks,
            'max_tasks': self.max_tasks
        }
        
        await self.redis.hset(f"nodes:{self.node_id}", mapping=node_info)
        await self.redis.sadd("active_nodes", self.node_id)
        
    async def _monitor_node(self):
        """Мониторинг состояния узла"""
        while True:
            try:
                # Обновление метрик
                self.cpu_usage = psutil.cpu_percent()
                self.memory_usage = psutil.virtual_memory().percent
                
                # Обновление информации в Redis
                await self.redis.hset(f"nodes:{self.node_id}", 
                                    "cpu_usage", self.cpu_usage,
                                    "memory_usage", self.memory_usage,
                                    "active_tasks", self.active_tasks)
                
                await asyncio.sleep(30)  # Обновление каждые 30 секунд
                
            except Exception as e:
                logger.error(f"Ошибка мониторинга узла: {e}")
                await asyncio.sleep(60)
    
    async def _process_tasks(self):
        """Обработка задач из очереди"""
        while True:
            try:
                task = await self.task_queue.get()
                await self._execute_task(task)
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Ошибка обработки задачи: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: Dict):
        """Выполнение задачи"""
        task_id = task.get('task_id')
        task_type = task.get('type')
        
        try:
            self.active_tasks += 1
            
            # Выполнение задачи в зависимости от типа
            if task_type == 'prediction':
                result = await self._execute_prediction_task(task)
            elif task_type == 'backtest':
                result = await self._execute_backtest_task(task)
            elif task_type == 'analysis':
                result = await self._execute_analysis_task(task)
            else:
                result = {'error': f'Неизвестный тип задачи: {task_type}'}
            
            # Сохранение результата
            await self.redis.set(f"task_result:{task_id}", json.dumps(result), ex=3600)
            
        except Exception as e:
            logger.error(f"Ошибка выполнения задачи {task_id}: {e}")
            await self.redis.set(f"task_result:{task_id}", 
                               json.dumps({'error': str(e)}), ex=3600)
        finally:
            self.active_tasks -= 1
    
    async def _execute_prediction_task(self, task: Dict) -> Dict:
        """Выполнение задачи предсказания"""
        # Здесь будет логика выполнения предсказаний
        await asyncio.sleep(1)  # Симуляция работы
        
        return {
            'task_id': task.get('task_id'),
            'type': 'prediction',
            'result': {
                'prediction': [0.3, 0.4, 0.3],
                'confidence': 0.75,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    async def _execute_backtest_task(self, task: Dict) -> Dict:
        """Выполнение задачи backtesting"""
        # Здесь будет логика выполнения backtesting
        await asyncio.sleep(2)  # Симуляция работы
        
        return {
            'task_id': task.get('task_id'),
            'type': 'backtest',
            'result': {
                'total_trades': 100,
                'win_rate': 65.0,
                'total_profit': 1000.0,
                'sharpe_ratio': 1.2
            }
        }
    
    async def _execute_analysis_task(self, task: Dict) -> Dict:
        """Выполнение задачи анализа"""
        # Здесь будет логика выполнения анализа
        await asyncio.sleep(1)  # Симуляция работы
        
        return {
            'task_id': task.get('task_id'),
            'type': 'analysis',
            'result': {
                'metrics': {
                    'total_profit': 1000.0,
                    'win_rate': 65.0,
                    'sharpe_ratio': 1.2
                }
            }
        }
    
    async def shutdown(self):
        """Завершение работы узла"""
        self.status = "shutdown"
        await self.redis.srem("active_nodes", self.node_id)
        await self.redis.delete(f"nodes:{self.node_id}")
        logger.info(f"Узел {self.node_id} завершил работу")

class LoadBalancer:
    """Балансировщик нагрузки"""
    
    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis = None
        self.redis_url = redis_url
        self.strategy = "round_robin"  # round_robin, least_loaded, cpu_based
        
    async def initialize(self):
        """Инициализация балансировщика"""
        self.redis = await aioredis.from_url(self.redis_url)
        logger.info("Балансировщик нагрузки инициализирован")
    
    async def get_available_nodes(self) -> List[Dict]:
        """Получение доступных узлов"""
        try:
            active_nodes = await self.redis.smembers("active_nodes")
            nodes = []
            
            for node_id in active_nodes:
                node_info = await self.redis.hgetall(f"nodes:{node_id.decode()}")
                if node_info:
                    nodes.append({
                        'node_id': node_info[b'node_id'].decode(),
                        'host': node_info[b'host'].decode(),
                        'port': int(node_info[b'port']),
                        'cpu_usage': float(node_info[b'cpu_usage']),
                        'memory_usage': float(node_info[b'memory_usage']),
                        'active_tasks': int(node_info[b'active_tasks']),
                        'max_tasks': int(node_info[b'max_tasks'])
                    })
            
            return nodes
        except Exception as e:
            logger.error(f"Ошибка получения узлов: {e}")
            return []
    
    async def select_node(self, task_type: str = None) -> Optional[Dict]:
        """Выбор узла для выполнения задачи"""
        nodes = await self.get_available_nodes()
        
        if not nodes:
            return None
        
        if self.strategy == "round_robin":
            return await self._round_robin_selection(nodes)
        elif self.strategy == "least_loaded":
            return await self._least_loaded_selection(nodes)
        elif self.strategy == "cpu_based":
            return await self._cpu_based_selection(nodes)
        else:
            return nodes[0]  # По умолчанию первый узел
    
    async def _round_robin_selection(self, nodes: List[Dict]) -> Dict:
        """Выбор узла по принципу round robin"""
        # Простая реализация round robin
        return nodes[0]
    
    async def _least_loaded_selection(self, nodes: List[Dict]) -> Dict:
        """Выбор наименее загруженного узла"""
        return min(nodes, key=lambda x: x['active_tasks'])
    
    async def _cpu_based_selection(self, nodes: List[Dict]) -> Dict:
        """Выбор узла на основе загрузки CPU"""
        return min(nodes, key=lambda x: x['cpu_usage'])

class DistributedTaskManager:
    """Менеджер распределенных задач"""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.task_counter = 0
        
    async def submit_task(self, task_type: str, data: Dict) -> str:
        """Отправка задачи на выполнение"""
        # Выбор узла
        node = await self.load_balancer.select_node(task_type)
        if not node:
            raise Exception("Нет доступных узлов")
        
        # Создание задачи
        task_id = str(uuid.uuid4())
        task = {
            'task_id': task_id,
            'type': task_type,
            'data': data,
            'node_id': node['node_id'],
            'created_at': datetime.now().isoformat()
        }
        
        # Отправка задачи на узел
        await self._send_task_to_node(node, task)
        
        return task_id
    
    async def _send_task_to_node(self, node: Dict, task: Dict):
        """Отправка задачи на узел"""
        try:
            url = f"http://{node['host']}:{node['port']}/api/tasks"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=task) as response:
                    if response.status != 200:
                        raise Exception(f"Ошибка отправки задачи: {response.status}")
        except Exception as e:
            logger.error(f"Ошибка отправки задачи на узел {node['node_id']}: {e}")
            raise
    
    async def get_task_result(self, task_id: str) -> Optional[Dict]:
        """Получение результата задачи"""
        try:
            result = await self.load_balancer.redis.get(f"task_result:{task_id}")
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            logger.error(f"Ошибка получения результата задачи {task_id}: {e}")
            return None
    
    async def wait_for_task(self, task_id: str, timeout: int = 300) -> Optional[Dict]:
        """Ожидание завершения задачи"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = await self.get_task_result(task_id)
            if result:
                return result
            await asyncio.sleep(1)
        
        return None

class ClusterManager:
    """Менеджер кластера"""
    
    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis_url = redis_url
        self.load_balancer = LoadBalancer(redis_url)
        self.task_manager = DistributedTaskManager(self.load_balancer)
        
    async def initialize(self):
        """Инициализация менеджера кластера"""
        await self.load_balancer.initialize()
        logger.info("Менеджер кластера инициализирован")
    
    async def get_cluster_status(self) -> Dict:
        """Получение статуса кластера"""
        nodes = await self.load_balancer.get_available_nodes()
        
        total_cpu = sum(node['cpu_usage'] for node in nodes)
        total_memory = sum(node['memory_usage'] for node in nodes)
        total_tasks = sum(node['active_tasks'] for node in nodes)
        
        return {
            'total_nodes': len(nodes),
            'active_nodes': len([n for n in nodes if n['active_tasks'] < n['max_tasks']]),
            'average_cpu_usage': total_cpu / len(nodes) if nodes else 0,
            'average_memory_usage': total_memory / len(nodes) if nodes else 0,
            'total_active_tasks': total_tasks,
            'nodes': nodes
        }
    
    async def scale_up(self, target_nodes: int):
        """Масштабирование вверх"""
        current_nodes = len(await self.load_balancer.get_available_nodes())
        
        if current_nodes < target_nodes:
            nodes_to_add = target_nodes - current_nodes
            logger.info(f"Добавление {nodes_to_add} узлов")
            
            # Здесь будет логика добавления новых узлов
            # Например, запуск новых контейнеров или инстансов
    
    async def scale_down(self, target_nodes: int):
        """Масштабирование вниз"""
        current_nodes = len(await self.load_balancer.get_available_nodes())
        
        if current_nodes > target_nodes:
            nodes_to_remove = current_nodes - target_nodes
            logger.info(f"Удаление {nodes_to_remove} узлов")
            
            # Здесь будет логика удаления узлов
            # Например, остановка контейнеров или инстансов

# FastAPI приложение для узла
app = FastAPI(title="ForexBot AI Node", version="1.0.0")

# Глобальные переменные
node_manager = None
cluster_manager = None

@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения"""
    global node_manager, cluster_manager
    
    # Инициализация узла
    node_manager = NodeManager()
    await node_manager.initialize()
    
    # Инициализация менеджера кластера
    cluster_manager = ClusterManager()
    await cluster_manager.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Событие завершения приложения"""
    if node_manager:
        await node_manager.shutdown()

@app.post("/api/tasks")
async def submit_task(task: Dict):
    """Получение задачи от балансировщика"""
    try:
        await node_manager.task_queue.put(task)
        return {"status": "accepted", "task_id": task.get('task_id')}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/status")
async def get_node_status():
    """Получение статуса узла"""
    return {
        "node_id": node_manager.node_id,
        "status": node_manager.status,
        "cpu_usage": node_manager.cpu_usage,
        "memory_usage": node_manager.memory_usage,
        "active_tasks": node_manager.active_tasks,
        "max_tasks": node_manager.max_tasks,
        "uptime": (datetime.now() - node_manager.start_time).total_seconds()
    }

@app.get("/api/cluster/status")
async def get_cluster_status():
    """Получение статуса кластера"""
    return await cluster_manager.get_cluster_status()

@app.post("/api/cluster/scale")
async def scale_cluster(scale_request: Dict):
    """Масштабирование кластера"""
    action = scale_request.get('action')
    target_nodes = scale_request.get('target_nodes', 1)
    
    if action == 'up':
        await cluster_manager.scale_up(target_nodes)
    elif action == 'down':
        await cluster_manager.scale_down(target_nodes)
    
    return {"status": "scaling", "action": action, "target_nodes": target_nodes}

# Пример использования
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Запуск узла ForexBot AI...")
    print(f"📊 Узел ID: {NodeManager().node_id}")
    print("🌐 Доступен по адресу: http://localhost:8000")
    print("📈 Метрики: http://localhost:8000/api/status")
    print("🔗 Статус кластера: http://localhost:8000/api/cluster/status")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)