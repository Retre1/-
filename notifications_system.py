#!/usr/bin/env python3
"""
Notifications System for ForexBot
Система уведомлений и алертинга
"""

import asyncio
import smtplib
import requests
from datetime import datetime
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json

class NotificationManager:
    """Менеджер уведомлений"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.notification_history = []
        self.alert_rules = []
        
    async def send_email_notification(self, subject: str, message: str, recipients: List[str]):
        """Отправка email уведомления"""
        if not self.config.get("email", {}).get("enabled", False):
            return False
            
        email_config = self.config["email"]
        
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config["username"]
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
            server.starttls()
            server.login(email_config["username"], email_config["password"])
            
            text = msg.as_string()
            server.sendmail(email_config["username"], recipients, text)
            server.quit()
            
            self._log_notification("email", subject, message, recipients)
            return True
            
        except Exception as e:
            print(f"Ошибка отправки email: {e}")
            return False
            
    async def send_telegram_notification(self, message: str, chat_id: str = None):
        """Отправка Telegram уведомления"""
        if not self.config.get("telegram", {}).get("enabled", False):
            return False
            
        telegram_config = self.config["telegram"]
        chat_id = chat_id or telegram_config["chat_id"]
        
        try:
            url = f"https://api.telegram.org/bot{telegram_config['bot_token']}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                self._log_notification("telegram", "Telegram Notification", message, [chat_id])
                return True
            else:
                print(f"Ошибка отправки Telegram: {response.text}")
                return False
                
        except Exception as e:
            print(f"Ошибка отправки Telegram: {e}")
            return False
            
    async def send_webhook_notification(self, event_type: str, data: Dict):
        """Отправка webhook уведомления"""
        if not self.config.get("webhook", {}).get("enabled", False):
            return False
            
        webhook_config = self.config["webhook"]
        
        try:
            payload = {
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            response = requests.post(
                webhook_config["url"],
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                self._log_notification("webhook", event_type, str(data), [webhook_config["url"]])
                return True
            else:
                print(f"Ошибка отправки webhook: {response.text}")
                return False
                
        except Exception as e:
            print(f"Ошибка отправки webhook: {e}")
            return False
            
    async def send_discord_notification(self, message: str, webhook_url: str = None):
        """Отправка Discord уведомления"""
        if not webhook_url and not self.config.get("discord", {}).get("webhook_url"):
            return False
            
        webhook_url = webhook_url or self.config["discord"]["webhook_url"]
        
        try:
            payload = {
                "content": message,
                "username": "ForexBot AI"
            }
            
            response = requests.post(webhook_url, json=payload)
            
            if response.status_code == 204:
                self._log_notification("discord", "Discord Notification", message, [webhook_url])
                return True
            else:
                print(f"Ошибка отправки Discord: {response.text}")
                return False
                
        except Exception as e:
            print(f"Ошибка отправки Discord: {e}")
            return False
            
    def add_alert_rule(self, rule: Dict):
        """Добавление правила алерта"""
        self.alert_rules.append(rule)
        
    async def check_alerts(self, current_data: Dict):
        """Проверка алертов"""
        for rule in self.alert_rules:
            if await self._evaluate_rule(rule, current_data):
                await self._trigger_alert(rule, current_data)
                
    async def _evaluate_rule(self, rule: Dict, data: Dict) -> bool:
        """Оценка правила алерта"""
        rule_type = rule.get("type")
        
        if rule_type == "profit_threshold":
            current_profit = data.get("total_profit", 0)
            threshold = rule.get("threshold", 0)
            return current_profit >= threshold
            
        elif rule_type == "loss_threshold":
            current_loss = abs(data.get("total_profit", 0))
            threshold = rule.get("threshold", 0)
            return current_loss >= threshold
            
        elif rule_type == "drawdown_threshold":
            current_drawdown = data.get("current_drawdown", 0)
            threshold = rule.get("threshold", 0)
            return current_drawdown >= threshold
            
        elif rule_type == "trade_count":
            current_trades = data.get("total_trades", 0)
            threshold = rule.get("threshold", 0)
            return current_trades >= threshold
            
        elif rule_type == "signal_confidence":
            signal_confidence = data.get("signal_confidence", 0)
            threshold = rule.get("threshold", 0)
            return signal_confidence >= threshold
            
        return False
        
    async def _trigger_alert(self, rule: Dict, data: Dict):
        """Срабатывание алерта"""
        alert_message = self._format_alert_message(rule, data)
        
        # Отправка уведомлений
        await asyncio.gather(
            self.send_email_notification(
                f"Алерт: {rule.get('name', 'Unknown Alert')}",
                alert_message,
                rule.get("email_recipients", [])
            ),
            self.send_telegram_notification(alert_message),
            self.send_webhook_notification("alert", {
                "rule": rule,
                "data": data,
                "message": alert_message
            })
        )
        
    def _format_alert_message(self, rule: Dict, data: Dict) -> str:
        """Форматирование сообщения алерта"""
        rule_name = rule.get("name", "Unknown Alert")
        rule_type = rule.get("type", "unknown")
        
        message = f"🚨 АЛЕРТ: {rule_name}\n\n"
        message += f"Тип: {rule_type}\n"
        message += f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if rule_type == "profit_threshold":
            message += f"Прибыль достигла: ${data.get('total_profit', 0):.2f}\n"
            message += f"Порог: ${rule.get('threshold', 0):.2f}\n"
            
        elif rule_type == "loss_threshold":
            message += f"Убыток достиг: ${abs(data.get('total_profit', 0)):.2f}\n"
            message += f"Порог: ${rule.get('threshold', 0):.2f}\n"
            
        elif rule_type == "drawdown_threshold":
            message += f"Просадка достигла: {data.get('current_drawdown', 0):.2f}%\n"
            message += f"Порог: {rule.get('threshold', 0):.2f}%\n"
            
        elif rule_type == "trade_count":
            message += f"Количество сделок: {data.get('total_trades', 0)}\n"
            message += f"Порог: {rule.get('threshold', 0)}\n"
            
        elif rule_type == "signal_confidence":
            message += f"Уверенность сигнала: {data.get('signal_confidence', 0)*100:.1f}%\n"
            message += f"Порог: {rule.get('threshold', 0)*100:.1f}%\n"
            
        return message
        
    def _log_notification(self, method: str, subject: str, message: str, recipients: List[str]):
        """Логирование уведомления"""
        notification = {
            "method": method,
            "subject": subject,
            "message": message,
            "recipients": recipients,
            "timestamp": datetime.now().isoformat()
        }
        
        self.notification_history.append(notification)
        
        # Ограничение истории
        if len(self.notification_history) > 1000:
            self.notification_history = self.notification_history[-1000:]
            
    def get_notification_history(self, limit: int = 100) -> List[Dict]:
        """Получение истории уведомлений"""
        return self.notification_history[-limit:]
        
    async def send_trade_notification(self, trade_data: Dict):
        """Уведомление о сделке"""
        trade_type = "ПОКУПКА" if trade_data["direction"] == "BUY" else "ПРОДАЖА"
        
        message = f"📊 НОВАЯ СДЕЛКА\n\n"
        message += f"Символ: {trade_data['symbol']}\n"
        message += f"Тип: {trade_type}\n"
        message += f"Объем: {trade_data['volume']}\n"
        message += f"Цена: {trade_data['open_price']}\n"
        message += f"Стратегия: {trade_data.get('strategy', 'N/A')}\n"
        message += f"Уверенность: {trade_data.get('confidence', 0)*100:.1f}%\n"
        message += f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await asyncio.gather(
            self.send_telegram_notification(message),
            self.send_webhook_notification("trade_opened", trade_data)
        )
        
    async def send_profit_notification(self, profit_data: Dict):
        """Уведомление о прибыли"""
        message = f"💰 ПРИБЫЛЬ\n\n"
        message += f"Сделка: {profit_data['symbol']}\n"
        message += f"Прибыль: ${profit_data['profit']:.2f}\n"
        message += f"Общая прибыль: ${profit_data.get('total_profit', 0):.2f}\n"
        message += f"Винрейт: {profit_data.get('win_rate', 0):.1f}%"
        
        await asyncio.gather(
            self.send_telegram_notification(message),
            self.send_webhook_notification("profit", profit_data)
        )
        
    async def send_error_notification(self, error_data: Dict):
        """Уведомление об ошибке"""
        message = f"❌ ОШИБКА\n\n"
        message += f"Модуль: {error_data.get('module', 'Unknown')}\n"
        message += f"Ошибка: {error_data.get('error', 'Unknown error')}\n"
        message += f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await asyncio.gather(
            self.send_telegram_notification(message),
            self.send_email_notification(
                "ForexBot Error",
                message,
                self.config.get("email", {}).get("recipient", [])
            ),
            self.send_webhook_notification("error", error_data)
        )

# Пример конфигурации
SAMPLE_CONFIG = {
    "email": {
        "enabled": True,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your-email@gmail.com",
        "password": "your-app-password",
        "recipient": ["admin@forexbot.com"]
    },
    "telegram": {
        "enabled": True,
        "bot_token": "your-bot-token",
        "chat_id": "your-chat-id"
    },
    "webhook": {
        "enabled": True,
        "url": "https://your-webhook-url.com/webhook"
    },
    "discord": {
        "webhook_url": "https://discord.com/api/webhooks/your-webhook-url"
    }
}

# Пример использования
async def main():
    """Пример использования системы уведомлений"""
    notification_manager = NotificationManager(SAMPLE_CONFIG)
    
    # Добавление правил алертов
    notification_manager.add_alert_rule({
        "name": "Высокая прибыль",
        "type": "profit_threshold",
        "threshold": 1000,
        "email_recipients": ["admin@forexbot.com"]
    })
    
    notification_manager.add_alert_rule({
        "name": "Большая просадка",
        "type": "drawdown_threshold",
        "threshold": 10
    })
    
    # Тестирование уведомлений
    await notification_manager.send_telegram_notification("Тестовое уведомление от ForexBot!")
    
    # Проверка алертов
    current_data = {
        "total_profit": 1200,
        "current_drawdown": 5,
        "total_trades": 50
    }
    
    await notification_manager.check_alerts(current_data)
    
    # Уведомление о сделке
    trade_data = {
        "symbol": "EURUSD",
        "direction": "BUY",
        "volume": 0.1,
        "open_price": 1.0850,
        "strategy": "trend_following",
        "confidence": 0.85
    }
    
    await notification_manager.send_trade_notification(trade_data)

if __name__ == "__main__":
    asyncio.run(main())