# ==============================================================================
# 5. utils/logging_config.py (ロギング設定)
# ==============================================================================
import logging
import os
from typing import Optional, List
from logging.config import dictConfig

_is_main_logging_configured = False

# --- [修正] プロジェクトルートを基準にログパスを定義 ---
# このファイルの場所からプロジェクトのルートディレクトリを特定
# /workspaces/teacher-assignment-system/optimization_engine/utils/logging_config.py -> /workspaces/teacher-assignment-system
try:
    # __file__ が定義されている通常のPython環境
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    # Streamlit Cloudなど、__file__が定義されていない環境向けのフォールバック
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

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
            'filename': os.path.join(LOG_DIR, "app.log"),
            'encoding': 'utf-8',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'gateway_file': {
            'class': 'logging.FileHandler',
            'mode': 'w',
            'filename': os.path.join(LOG_DIR, "optimization_gateway.log"),
            'encoding': 'utf-8',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'solver_file': {
            'class': 'logging.FileHandler',
            'mode': 'w',
            'filename': os.path.join(LOG_DIR, "optimization_solver.log"),
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

# --- [修正] UIから参照できるように、絶対パスの定数をエクスポート ---
APP_LOG_FILE = LOGGING_CONFIG['handlers']['app_file']['filename']
GATEWAY_LOG_FILE = LOGGING_CONFIG['handlers']['gateway_file']['filename']
SOLVER_LOG_FILE = LOGGING_CONFIG['handlers']['solver_file']['filename']

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
