"""
Solana Token Manager - управление токеном и механизмом сжигания
"""

import asyncio
import json
import os
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal

from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TokenAccountOpts
from solana.rpc.commitment import Commitment
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.transaction import Transaction
from solana.system_program import transfer, TransferParams
from solders.transaction import VersionedTransaction
from solders.message import to_bytes_versioned

# SPL Token imports
from spl.token.async_client import AsyncToken
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import (
    burn, BurnParams,
    mint_to, MintToParams,
    create_account, CreateAccountParams,
    initialize_mint, InitializeMintParams
)

from loguru import logger


class SolanaTokenManager:
    """Менеджер для работы с токеном на Solana"""
    
    def __init__(self, config: dict):
        self.config = config
        self.client = None
        self.keypair = None
        self.token_mint = None
        self.token_account = None
        
        # Параметры токена
        self.token_address = config.get("token_address", "")
        self.burn_percentage = config.get("burn_percentage", 0.1)  # 10%
        self.min_profit_for_burn = config.get("min_profit_for_burn", 100)
        
        # RPC endpoint
        self.rpc_endpoint = config.get("rpc_endpoint", "https://api.mainnet-beta.solana.com")
        
        # Статистика сжигания
        self.burn_history = []
        self.total_burned = 0
    
    async def initialize(self) -> bool:
        """Инициализация подключения к Solana"""
        try:
            # Создаем клиент
            self.client = AsyncClient(self.rpc_endpoint)
            
            # Загружаем или создаем кошелек
            await self._load_or_create_wallet()
            
            # Проверяем подключение
            response = await self.client.get_health()
            if response.value != "ok":
                raise Exception("Solana RPC не отвечает")
            
            # Инициализируем токен
            if self.token_address:
                await self._initialize_existing_token()
            else:
                await self._create_new_token()
            
            logger.info(f"Solana подключен. Кошелек: {self.keypair.public_key}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации Solana: {e}")
            return False
    
    async def _load_or_create_wallet(self):
        """Загрузка или создание кошелька"""
        wallet_path = "data/solana_wallet.json"
        
        if os.path.exists(wallet_path):
            try:
                with open(wallet_path, 'r') as f:
                    wallet_data = json.load(f)
                    self.keypair = Keypair.from_secret_key(bytes(wallet_data))
                logger.info("Кошелек загружен")
            except Exception as e:
                logger.error(f"Ошибка загрузки кошелька: {e}")
                self.keypair = Keypair()
                await self._save_wallet()
        else:
            # Создаем новый кошелек
            self.keypair = Keypair()
            await self._save_wallet()
            logger.info("Создан новый кошелек")
    
    async def _save_wallet(self):
        """Сохранение кошелька"""
        wallet_path = "data/solana_wallet.json"
        os.makedirs(os.path.dirname(wallet_path), exist_ok=True)
        
        try:
            with open(wallet_path, 'w') as f:
                json.dump(list(self.keypair.secret_key), f)
            logger.info("Кошелек сохранен")
        except Exception as e:
            logger.error(f"Ошибка сохранения кошелька: {e}")
    
    async def _initialize_existing_token(self):
        """Инициализация существующего токена"""
        try:
            self.token_mint = PublicKey(self.token_address)
            
            # Получаем информацию о токене
            token_info = await self.client.get_account_info(self.token_mint)
            if not token_info.value:
                raise Exception(f"Токен {self.token_address} не найден")
            
            # Создаем клиент токена
            self.token = AsyncToken(
                self.client,
                self.token_mint,
                TOKEN_PROGRAM_ID,
                self.keypair
            )
            
            # Получаем или создаем аккаунт токена
            await self._get_or_create_token_account()
            
            logger.info(f"Токен {self.token_address} инициализирован")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации токена: {e}")
            raise
    
    async def _create_new_token(self):
        """Создание нового токена"""
        try:
            # Генерируем адрес для mint
            mint_keypair = Keypair()
            self.token_mint = mint_keypair.public_key
            
            # Параметры токена
            decimals = 9  # Стандартные decimals для SPL токена
            mint_authority = self.keypair.public_key
            freeze_authority = self.keypair.public_key
            
            # Создаем токен
            self.token = AsyncToken(
                self.client,
                self.token_mint,
                TOKEN_PROGRAM_ID,
                self.keypair
            )
            
            # Создаем mint аккаунт
            await self.token.create_mint(
                mint_keypair,
                mint_authority,
                freeze_authority,
                decimals,
                self.keypair
            )
            
            # Создаем аккаунт токена
            await self._get_or_create_token_account()
            
            # Выпускаем начальное количество токенов (1 миллион)
            initial_supply = 1_000_000 * (10 ** decimals)
            await self.token.mint_to(
                self.token_account,
                mint_authority,
                initial_supply,
                signers=[self.keypair]
            )
            
            # Сохраняем адрес токена
            self.token_address = str(self.token_mint)
            
            logger.info(f"Создан новый токен: {self.token_address}")
            logger.info(f"Начальный выпуск: {initial_supply / (10 ** decimals)} токенов")
            
        except Exception as e:
            logger.error(f"Ошибка создания токена: {e}")
            raise
    
    async def _get_or_create_token_account(self):
        """Получение или создание аккаунта токена"""
        try:
            # Пытаемся найти существующий аккаунт
            response = await self.client.get_token_accounts_by_owner(
                self.keypair.public_key,
                TokenAccountOpts(mint=self.token_mint)
            )
            
            if response.value:
                # Используем существующий аккаунт
                self.token_account = response.value[0].pubkey
                logger.info("Использован существующий аккаунт токена")
            else:
                # Создаем новый аккаунт
                self.token_account = await self.token.create_account(
                    self.keypair.public_key,
                    self.keypair
                )
                logger.info("Создан новый аккаунт токена")
                
        except Exception as e:
            logger.error(f"Ошибка создания аккаунта токена: {e}")
            raise
    
    async def get_balance(self) -> float:
        """Получение баланса токенов"""
        try:
            if not self.token_account:
                return 0.0
            
            response = await self.client.get_token_account_balance(self.token_account)
            if response.value:
                return float(response.value.amount) / (10 ** response.value.decimals)
            return 0.0
            
        except Exception as e:
            logger.error(f"Ошибка получения баланса: {e}")
            return 0.0
    
    async def calculate_burn_amount(self, profit_percentage: float) -> int:
        """Расчет количества токенов для сжигания"""
        try:
            if profit_percentage <= 0:
                return 0
            
            # Получаем текущий баланс
            current_balance = await self.get_balance()
            
            if current_balance == 0:
                return 0
            
            # Рассчитываем количество для сжигания
            # Формула: burn_amount = (profit_percentage / 100) * burn_percentage * current_balance
            burn_ratio = (profit_percentage / 100) * self.burn_percentage
            burn_amount = int(current_balance * burn_ratio)
            
            # Минимальное количество для сжигания
            min_burn = 1000  # 1000 токенов
            
            if burn_amount < min_burn:
                return 0
            
            # Максимальное количество - не больше 10% от баланса
            max_burn = int(current_balance * 0.1)
            burn_amount = min(burn_amount, max_burn)
            
            logger.info(f"Рассчитано для сжигания: {burn_amount} токенов")
            return burn_amount
            
        except Exception as e:
            logger.error(f"Ошибка расчета сжигания: {e}")
            return 0
    
    async def burn_tokens(self, amount: int) -> bool:
        """Сжигание токенов"""
        try:
            if amount <= 0:
                logger.warning("Количество для сжигания должно быть больше 0")
                return False
            
            if not self.token or not self.token_account:
                logger.error("Токен не инициализирован")
                return False
            
            # Проверяем баланс
            current_balance = await self.get_balance()
            decimals = 9  # Стандартные decimals
            amount_with_decimals = amount * (10 ** decimals)
            
            if current_balance * (10 ** decimals) < amount_with_decimals:
                logger.error(f"Недостаточно токенов для сжигания: {current_balance} < {amount}")
                return False
            
            # Создаем транзакцию сжигания
            burn_instruction = burn(
                BurnParams(
                    program_id=TOKEN_PROGRAM_ID,
                    account=self.token_account,
                    mint=self.token_mint,
                    owner=self.keypair.public_key,
                    amount=amount_with_decimals,
                    signers=[self.keypair.public_key]
                )
            )
            
            # Отправляем транзакцию
            transaction = Transaction().add(burn_instruction)
            response = await self.client.send_transaction(
                transaction,
                self.keypair,
                opts={"skip_confirmation": False}
            )
            
            if response.value:
                # Записываем в историю
                burn_record = {
                    "timestamp": datetime.now().isoformat(),
                    "amount": amount,
                    "transaction": response.value,
                    "balance_before": current_balance,
                    "balance_after": await self.get_balance()
                }
                
                self.burn_history.append(burn_record)
                self.total_burned += amount
                
                logger.info(f"Сожжено {amount} токенов. Транзакция: {response.value}")
                return True
            else:
                logger.error("Транзакция сжигания не подтверждена")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка сжигания токенов: {e}")
            return False
    
    async def get_burn_statistics(self) -> Dict:
        """Получение статистики сжигания"""
        current_balance = await self.get_balance()
        
        # Статистика за последний месяц
        month_ago = datetime.now() - timedelta(days=30)
        monthly_burns = [
            record for record in self.burn_history
            if datetime.fromisoformat(record["timestamp"]) >= month_ago
        ]
        
        monthly_burned = sum(record["amount"] for record in monthly_burns)
        
        return {
            "total_burned": self.total_burned,
            "monthly_burned": monthly_burned,
            "current_balance": current_balance,
            "burn_events_total": len(self.burn_history),
            "burn_events_monthly": len(monthly_burns),
            "last_burn": self.burn_history[-1] if self.burn_history else None,
            "token_address": self.token_address
        }
    
    async def transfer_tokens(self, recipient: str, amount: int) -> bool:
        """Перевод токенов на другой адрес"""
        try:
            recipient_pubkey = PublicKey(recipient)
            decimals = 9
            amount_with_decimals = amount * (10 ** decimals)
            
            # Получаем или создаем аккаунт получателя
            recipient_token_account = await self.token.create_associated_token_account(
                recipient_pubkey
            )
            
            # Создаем транзакцию перевода
            transfer_instruction = await self.token.transfer(
                self.token_account,
                recipient_token_account,
                self.keypair.public_key,
                amount_with_decimals,
                signers=[self.keypair]
            )
            
            logger.info(f"Переведено {amount} токенов на {recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка перевода токенов: {e}")
            return False
    
    async def get_token_info(self) -> Dict:
        """Получение информации о токене"""
        try:
            if not self.token_mint:
                return {}
            
            # Получаем информацию о mint
            mint_info = await self.client.get_account_info(self.token_mint)
            
            # Получаем supply
            supply_info = await self.client.get_token_supply(self.token_mint)
            
            current_balance = await self.get_balance()
            
            return {
                "address": self.token_address,
                "mint": str(self.token_mint),
                "total_supply": float(supply_info.value.amount) / (10 ** supply_info.value.decimals) if supply_info.value else 0,
                "decimals": supply_info.value.decimals if supply_info.value else 9,
                "our_balance": current_balance,
                "our_percentage": (current_balance / (float(supply_info.value.amount) / (10 ** supply_info.value.decimals))) * 100 if supply_info.value and supply_info.value.amount != "0" else 0
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о токене: {e}")
            return {}
    
    async def close(self):
        """Закрытие подключения"""
        if self.client:
            await self.client.close()
            logger.info("Solana клиент закрыт")


# Алиас для совместимости
TokenManager = SolanaTokenManager