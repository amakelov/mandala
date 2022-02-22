import logging

from ..core.config import EnvConfig, CoreConfig

if EnvConfig.has_rich and CoreConfig.use_rich:
    from rich.logging import RichHandler
    logging_handler = RichHandler(enable_link_path=False)
    FORMAT = "%(message)s"
    logging.basicConfig(
        level='INFO', format=FORMAT, datefmt="[%X]", handlers=[logging_handler]
    )
else:
    logging_handler = logging.Handler()

LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}


def set_logging_level(level='debug'):
    logging.getLogger().setLevel(LEVELS[level])
