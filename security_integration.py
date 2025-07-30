#!/usr/bin/env python3
"""
Security Integration for ForexBot
Интеграция аутентификации и безопасности
"""

import jwt
import bcrypt
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Модели безопасности
class User(BaseModel):
    username: str
    email: str
    role: str = "user"  # user/admin
    is_active: bool = True

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: str = "user"

class LoginRequest(BaseModel):
    username: str
    password: str

class SecurityManager:
    """Менеджер безопасности"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.users = {}  # В продакшене использовать БД
        self.blacklisted_tokens = set()
        
    def hash_password(self, password: str) -> str:
        """Хеширование пароля"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        
    def verify_password(self, password: str, hashed: str) -> bool:
        """Проверка пароля"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        
    def create_user(self, user_data: UserCreate) -> User:
        """Создание пользователя"""
        if user_data.username in self.users:
            raise ValueError("Пользователь уже существует")
            
        hashed_password = self.hash_password(user_data.password)
        
        user = User(
            username=user_data.username,
            email=user_data.email,
            role=user_data.role
        )
        
        self.users[user_data.username] = {
            "user": user,
            "hashed_password": hashed_password
        }
        
        return user
        
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Аутентификация пользователя"""
        if username not in self.users:
            return None
            
        user_data = self.users[username]
        if not self.verify_password(password, user_data["hashed_password"]):
            return None
            
        return user_data["user"]
        
    def create_access_token(self, user: User, expires_delta: timedelta = None) -> str:
        """Создание JWT токена"""
        if expires_delta is None:
            expires_delta = timedelta(hours=24)
            
        expire = datetime.utcnow() + expires_delta
        
        to_encode = {
            "sub": user.username,
            "email": user.email,
            "role": user.role,
            "exp": expire
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm="HS256")
        
    def verify_token(self, token: str) -> Optional[Dict]:
        """Проверка JWT токена"""
        try:
            if token in self.blacklisted_tokens:
                return None
                
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None
            
    def blacklist_token(self, token: str):
        """Добавление токена в черный список"""
        self.blacklisted_tokens.add(token)
        
    def is_admin(self, user: User) -> bool:
        """Проверка прав администратора"""
        return user.role == "admin"

# FastAPI интеграция
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Получение текущего пользователя"""
    token = credentials.credentials
    payload = security_manager.verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Недействительный токен",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    username = payload.get("sub")
    if username not in security_manager.users:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден"
        )
        
    return security_manager.users[username]["user"]

def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Получение текущего администратора"""
    if not security_manager.is_admin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав"
        )
    return current_user

# Инициализация
security_manager = SecurityManager()

# Создание тестовых пользователей
def create_test_users():
    """Создание тестовых пользователей"""
    admin_user = UserCreate(
        username="admin",
        email="admin@forexbot.com",
        password="admin123",
        role="admin"
    )
    
    user = UserCreate(
        username="user",
        email="user@forexbot.com",
        password="user123",
        role="user"
    )
    
    security_manager.create_user(admin_user)
    security_manager.create_user(user)
    
    print("Тестовые пользователи созданы:")
    print("admin/admin123 (администратор)")
    print("user/user123 (пользователь)")

# Пример защищенных endpoints
def create_protected_endpoints(app):
    """Создание защищенных endpoints"""
    
    @app.post("/api/auth/register")
    async def register(user_data: UserCreate):
        """Регистрация пользователя"""
        try:
            user = security_manager.create_user(user_data)
            return {"message": "Пользователь создан", "username": user.username}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/api/auth/login")
    async def login(login_data: LoginRequest):
        """Вход в систему"""
        user = security_manager.authenticate_user(login_data.username, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверные учетные данные"
            )
            
        access_token = security_manager.create_access_token(user)
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "username": user.username,
                "email": user.email,
                "role": user.role
            }
        }
    
    @app.post("/api/auth/logout")
    async def logout(current_user: User = Depends(get_current_user)):
        """Выход из системы"""
        # В реальном приложении здесь нужно добавить токен в черный список
        return {"message": "Успешный выход"}
    
    @app.get("/api/auth/me")
    async def get_current_user_info(current_user: User = Depends(get_current_user)):
        """Получение информации о текущем пользователе"""
        return {
            "username": current_user.username,
            "email": current_user.email,
            "role": current_user.role
        }
    
    # Защищенные endpoints для администраторов
    @app.get("/api/admin/users")
    async def get_users(admin: User = Depends(get_current_admin_user)):
        """Получение списка пользователей (только для админов)"""
        users = []
        for username, user_data in security_manager.users.items():
            users.append({
                "username": user_data["user"].username,
                "email": user_data["user"].email,
                "role": user_data["user"].role,
                "is_active": user_data["user"].is_active
            })
        return {"users": users}
    
    @app.post("/api/admin/users/{username}/deactivate")
    async def deactivate_user(username: str, admin: User = Depends(get_current_admin_user)):
        """Деактивация пользователя (только для админов)"""
        if username not in security_manager.users:
            raise HTTPException(status_code=404, detail="Пользователь не найден")
            
        security_manager.users[username]["user"].is_active = False
        return {"message": f"Пользователь {username} деактивирован"}

# Пример использования
if __name__ == "__main__":
    # Создание тестовых пользователей
    create_test_users()
    
    # Тестирование аутентификации
    user = security_manager.authenticate_user("admin", "admin123")
    if user:
        token = security_manager.create_access_token(user)
        print(f"Токен создан: {token[:50]}...")
        
        # Проверка токена
        payload = security_manager.verify_token(token)
        print(f"Проверка токена: {payload}")
    else:
        print("Ошибка аутентификации")