import logging
import os
from logging.config import dictConfig

_is_main_logging_configured = False

# --- [修正] streamlit run app.py で実行されることを前提に、カレントディレクトリをルートとする ---
# streamlit run app.py で実行すると、カレントワーキングディレクトリは
# プロジェクトのルートになるため、それを基準にするのが最も確実です。
PROJECT_ROOT = os.getcwd()
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'solver_formatter': {
            'format': '%(message)s'
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
            'formatter': 'solver_formatter',
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
            'propagate': False,
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}

# --- UIから参照できるように、絶対パスの定数をエクスポート ---
APP_LOG_FILE = LOGGING_CONFIG['handlers']['app_file']['filename']
GATEWAY_LOG_FILE = LOGGING_CONFIG['handlers']['gateway_file']['filename']
SOLVER_LOG_FILE = LOGGING_CONFIG['handlers']['solver_file']['filename']

def setup_logging():
    """
    アプリケーション全体のロギングを設定します。
    """
    global _is_main_logging_configured
    if _is_main_logging_configured:
        return

    # ログディレクトリが存在しない場合は作成します
    os.makedirs(LOG_DIR, exist_ok=True)

    # 既存のハンドラをクリアして、設定の重複を防ぎます
    all_logger_names = list(LOGGING_CONFIG['loggers'].keys())
    all_logger_names.append('')  # ルートロガーも対象

    for logger_name in all_logger_names:
        logger_obj = logging.getLogger(logger_name)
        for handler in logger_obj.handlers[:]:
            logger_obj.removeHandler(handler)

    # 設定を適用します
    dictConfig(LOGGING_CONFIG)
    _is_main_logging_configured = True