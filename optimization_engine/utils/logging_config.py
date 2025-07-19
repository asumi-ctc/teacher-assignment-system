# ==============================================================================
# 5. utils/logging_config.py (ロギング設定)
# ==============================================================================
import logging
import os
from typing import Optional, List
from logging.config import dictConfig

_is_main_logging_configured = False
LOG_DIR = "logs"
APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")
GATEWAY_LOG_FILE = os.path.join(LOG_DIR, "optimization_gateway.log")
SOLVER_LOG_FILE = os.path.join(LOG_DIR, "optimization_solver.log")

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'app_file': {
            'class': 'logging.FileHandler',
            'mode': 'w',
            'filename': os.path.abspath(APP_LOG_FILE),
            'encoding': 'utf-8',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'gateway_file': {
            'class': 'logging.FileHandler',
            'mode': 'w',
            'filename': os.path.abspath(GATEWAY_LOG_FILE),
            'encoding': 'utf-8',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'solver_file': {
            'class': 'logging.FileHandler',
            'mode': 'w',
            'filename': os.path.abspath(SOLVER_LOG_FILE),
            'encoding': 'utf-8',
            'formatter': 'standard',
            'level': 'INFO',
        },
    },
    'loggers': {
        'app': {
            'handlers': ['app_file'],
            'level': 'INFO',
            'propagate': True,
        },
        'optimization_gateway': {
            'handlers': ['gateway_file'],
            'level': 'INFO',
            'propagate': True,
        },
        'optimization_solver': {
            'handlers': ['solver_file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}

def setup_logging():
    """
    アプリケーション全体のロギングを設定する。
    """
    global _is_main_logging_configured
    if _is_main_logging_configured:
        return
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    all_logger_names = list(LOGGING_CONFIG['loggers'].keys())
    all_logger_names.append('') 

    for logger_name in all_logger_names:
        logger_obj = logging.getLogger(logger_name)
        for handler in logger_obj.handlers[:]:
            logger_obj.removeHandler(handler)

    dictConfig(LOGGING_CONFIG)
    _is_main_logging_configured = True
