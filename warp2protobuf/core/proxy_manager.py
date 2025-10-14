# protobuf2openai/proxy_manager.py
import asyncio
import random
import os
import httpx
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AsyncProxyManager:
    """异步代理管理器"""

    def __init__(self):
        self.used_identifiers = {}
        self.lock = asyncio.Lock()

    async def cleanup_expired_identifiers(self):
        """清理过期的IP标识"""
        current_time = datetime.now()
        async with self.lock:
            expired_keys = [k for k, v in self.used_identifiers.items() if v < current_time]
            for key in expired_keys:
                del self.used_identifiers[key]

    async def get_proxy(self) -> Optional[str]:
        """获取代理IP"""
        # 优先读取环境变量，其次读取项目 config.PROXY_URL，最后不使用代理
        # 支持的环境变量：WARP_PROXY、HTTPS_PROXY、HTTP_PROXY、ALL_PROXY
        try:
            # 环境变量优先级
            env_proxy = (
                os.getenv("WARP_PROXY")
                or os.getenv("HTTPS_PROXY")
                or os.getenv("HTTP_PROXY")
                or os.getenv("ALL_PROXY")
                or os.getenv("https_proxy")
                or os.getenv("http_proxy")
                or os.getenv("all_proxy")
            )

            if env_proxy and env_proxy.strip():
                return env_proxy.strip()

            # 尝试从项目配置读取
            try:
                import config  # 项目根目录下的配置
                proxy_from_config = getattr(config, "PROXY_URL", "")
                if isinstance(proxy_from_config, str) and proxy_from_config.strip():
                    return proxy_from_config.strip()
            except Exception:
                pass

            # 无代理
            return None
        except Exception:
            # 任何异常都回退为不使用代理
            return None

    def format_proxy_for_httpx(self, proxy_str: str) -> Optional[str]:
        """格式化代理为httpx格式"""
        if not proxy_str:
            return None

        try:
            # 如果已经是完整的URL格式（http://或socks5://），直接返回
            if proxy_str.startswith(('http://', 'https://', 'socks5://', 'socks4://')):
                return proxy_str
            
            # 否则按照旧逻辑处理（兼容性）
            if '@' in proxy_str:
                credentials, host_port = proxy_str.split('@')
                user, password = credentials.split(':')
                host, port = host_port.split(':')
                return f"socks5://{user}:{password}@{host}:{port}"
            else:
                parts = proxy_str.split(':')
                if len(parts) == 2:
                    host, port = parts
                    return f"socks5://{host}:{port}"
                else:
                    logger.error(f"代理格式无法识别: {proxy_str}")
                    return None
        except Exception as e:
            logger.error(f"格式化代理失败: {e}", exc_info=True)
            return None
