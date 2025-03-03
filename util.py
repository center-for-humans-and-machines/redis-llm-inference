import redis
import os


def get_connection() -> redis.Redis:
    """Get a connection to the redis server"""
    host = os.getenv("REDIS_HOST", "localhost")
    print("Connecting redis to", host)
    return redis.Redis(
        host=host,
        port=os.getenv("REDIS_PORT", 6379),
        password=os.getenv("REDIS_PASSWORD", None),
    )
