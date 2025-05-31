"""
配置模块，负责加载环境变量和提供全局配置
"""

import os
from typing import Dict, Optional, Any
from dotenv import load_dotenv

class Config:
    
    # 数据库配置
    PGVECTOR_URL = os.getenv("PGVECTOR_URL")
    
    # 应用配置
    APP_PORT=7860
    APP_HOST="0.0.0.0"
    
    # 文档处理配置
    CHUNK_SIZE=1024
    CHUNK_OVERLAP=200
    