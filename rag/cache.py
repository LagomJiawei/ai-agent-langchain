"""
语义结果缓存
使用 Redis 缓存相似查询的答案
"""
import hashlib
import json
from typing import Optional
from loguru import logger

try:
    import redis
except ImportError:
    redis = None


class RagResultSemanticCache:
    """RAG 语义结果缓存"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 1,
        password: Optional[str] = None,
        prefix: str = "rag_cache:",
        ttl: int = 3600 * 24,  # 默认 24 小时
    ):
        self.prefix = prefix
        self.ttl = ttl
        self._enabled = redis is not None

        if self._enabled:
            try:
                self._client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True,
                )
                # 测试连接
                self._client.ping()
                logger.info("RAG 语义缓存已启用")
            except Exception as e:
                logger.warning(f"Redis 连接失败，缓存已禁用: {e}")
                self._enabled = False
        else:
            logger.warning("Redis 未安装，缓存已禁用")

    def _get_key(self, query: str) -> str:
        """生成缓存键"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{self.prefix}{query_hash}"

    def get(self, query: str) -> Optional[str]:
        """获取缓存结果"""
        if not self._enabled:
            return None

        try:
            key = self._get_key(query)
            value = self._client.get(key)
            if value:
                logger.debug(f"缓存命中: {query}")
                return value
            return None
        except Exception as e:
            logger.debug(f"缓存读取失败: {e}")
            return None

    def put(self, query: str, answer: str) -> None:
        """写入缓存"""
        if not self._enabled:
            return

        try:
            key = self._get_key(query)
            self._client.setex(key, self.ttl, answer)
            logger.debug(f"缓存写入: {query}")
        except Exception as e:
            logger.debug(f"缓存写入失败: {e}")

    def invalidate(self, query: str) -> None:
        """使缓存失效"""
        if not self._enabled:
            return

        try:
            key = self._get_key(query)
            self._client.delete(key)
        except Exception as e:
            logger.debug(f"缓存删除失败: {e}")

    def clear_all(self) -> None:
        """清空所有缓存"""
        if not self._enabled:
            return

        try:
            pattern = f"{self.prefix}*"
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
                logger.info(f"清空缓存: {len(keys)} 条")
        except Exception as e:
            logger.debug(f"缓存清空失败: {e}")

    def stats(self) -> dict:
        """获取缓存统计"""
        if not self._enabled:
            return {"enabled": False, "count": 0}

        try:
            pattern = f"{self.prefix}*"
            keys = self._client.keys(pattern)
            return {"enabled": True, "count": len(keys)}
        except Exception as e:
            return {"enabled": False, "error": str(e)}


# 全局缓存实例
_semantic_cache: Optional[RagResultSemanticCache] = None


def get_semantic_cache() -> Optional[RagResultSemanticCache]:
    """获取全局缓存实例"""
    global _semantic_cache

    from config.settings import settings

    if _semantic_cache is None and settings.redis.enabled:
        _semantic_cache = RagResultSemanticCache(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            password=settings.redis.password,
        )

    return _semantic_cache
