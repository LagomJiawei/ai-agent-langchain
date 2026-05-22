"""
工具限流保护模块
令牌桶 QPS 限制 + 并发控制 + 熔断器
"""
import time
import threading
from typing import Callable, Any, Dict
from functools import wraps
from collections import deque
from loguru import logger


class TokenBucket:
    """令牌桶限流器"""

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: 每秒生成令牌数 (QPS)
            capacity: 桶容量（最大突发请求数）
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1, block: bool = True) -> bool:
        """获取令牌

        Args:
            tokens: 需要的令牌数
            block: 是否阻塞等待

        Returns:
            是否成功获取令牌
        """
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                elif not block:
                    return False

            time.sleep(0.01)


class Semaphore:
    """并发控制信号量"""

    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        self._value = max_concurrent
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def acquire(self, block: bool = True) -> bool:
        with self._lock:
            while self._value <= 0:
                if not block:
                    return False
                self._cond.wait()
            self._value -= 1
            return True

    def release(self) -> None:
        with self._lock:
            self._value += 1
            self._cond.notify()


class CircuitBreaker:
    """熔断器"""

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        """
        Args:
            failure_threshold: 失败次数阈值，超过则熔断
            recovery_timeout: 恢复超时时间（秒），熔断后等待多久进入半开状态
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = self.STATE_CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """执行被保护的函数"""
        with self._lock:
            if self.state == self.STATE_OPEN:
                now = time.time()
                if now - self.last_failure_time > self.recovery_timeout:
                    self.state = self.STATE_HALF_OPEN
                    logger.info(f"熔断器进入半开状态: {func.__name__}")
                else:
                    raise RuntimeError(f"Circuit breaker is open: {func.__name__}")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self) -> None:
        """成功回调"""
        with self._lock:
            if self.state == self.STATE_HALF_OPEN:
                self.state = self.STATE_CLOSED
                logger.info("熔断器恢复闭合状态")
            self.failure_count = 0

    def _on_failure(self) -> None:
        """失败回调"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == self.STATE_HALF_OPEN:
                self.state = self.STATE_OPEN
                logger.warning(f"半开状态再次失败，重新熔断: {self.failure_count}")
            elif self.failure_count >= self.failure_threshold:
                self.state = self.STATE_OPEN
                logger.warning(f"达到失败阈值，熔断器触发: {self.failure_count}")


class RateLimiter:
    """统一限流管理器"""

    def __init__(
        self,
        qps: int = 5,
        max_concurrent: int = 10,
        circuit_breaker_threshold: int = 5,
    ):
        self.token_bucket = TokenBucket(rate=qps, capacity=qps * 2)
        self.semaphore = Semaphore(max_concurrent=max_concurrent)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self._lock = threading.Lock()

    def _get_circuit_breaker(self, tool_name: str) -> CircuitBreaker:
        """获取工具对应的熔断器"""
        with self._lock:
            if tool_name not in self.circuit_breakers:
                self.circuit_breakers[tool_name] = CircuitBreaker(
                    failure_threshold=self.circuit_breaker_threshold
                )
            return self.circuit_breakers[tool_name]

    def execute(self, tool_name: str, func: Callable, *args, **kwargs) -> Any:
        """执行工具（带限流保护）"""
        # 1. QPS 限流
        if not self.token_bucket.acquire(block=False):
            logger.warning(f"工具调用被 QPS 限流: {tool_name}")
            raise RuntimeError(f"Rate limit exceeded for tool: {tool_name}")

        # 2. 并发控制
        if not self.semaphore.acquire(block=True):
            logger.warning(f"工具调用被并发限制: {tool_name}")
            raise RuntimeError(f"Too many concurrent calls for tool: {tool_name}")

        try:
            # 3. 熔断器保护
            breaker = self._get_circuit_breaker(tool_name)
            return breaker.call(func, *args, **kwargs)
        finally:
            self.semaphore.release()


# 全局限流器实例
_global_rate_limiter: RateLimiter = None


def get_rate_limiter() -> RateLimiter:
    """获取全局限流器"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        from config.settings import settings

        _global_rate_limiter = RateLimiter(
            qps=settings.tool_rate_limit.qps,
            max_concurrent=settings.tool_rate_limit.max_concurrent,
            circuit_breaker_threshold=settings.tool_rate_limit.circuit_breaker_threshold,
        )
    return _global_rate_limiter


def rate_limited(tool_name: str = None):
    """限流装饰器"""

    def decorator(func: Callable) -> Callable:
        nonlocal tool_name
        if tool_name is None:
            tool_name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            from config.settings import settings

            if not settings.tool_rate_limit.enabled:
                return func(*args, **kwargs)

            limiter = get_rate_limiter()
            return limiter.execute(tool_name, func, *args, **kwargs)

        return wrapper

    return decorator
