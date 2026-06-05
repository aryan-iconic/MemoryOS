"""Logging configuration for memoryos."""

from logging import getLogger, StreamHandler, Formatter, DEBUG

logger = getLogger("memoryos")
logger.setLevel(DEBUG)

handler = StreamHandler()
formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)
