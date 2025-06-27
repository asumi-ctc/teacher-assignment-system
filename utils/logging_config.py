import logging
import os
from typing import Optional, List
from logging.config import dictConfig

# Global flag to prevent repeated full configuration in the main process
# This flag is specific to the process it's running in.
_is_main_logging_configured = False
LOG_DIR = "logs"
APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")
GATEWAY_LOG_FILE = os.path.join(LOG_DIR, "optimization_gateway.log")
SOLVER_LOG_FILE = os.path.join(LOG_DIR, "optimization_solver.log")


# --- 静的ロギング設定辞書 ---
# Djangoのsettings.pyに容易に移植できるよう、静的な辞書として定義。
# Streamlit環境では、setup_logging関数がこの辞書を適用する。
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
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        }
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}

def setup_logging(target_loggers: Optional[List[str]] = None):
    """
    アプリケーション全体のロギングを設定する。
    Streamlit環境での重複設定を防ぎつつ、Djangoにも移植しやすい静的な設定辞書を使用します。

    Args:
        target_loggers: (Streamlit/multiprocessing環境用) この引数は主にメインプロセスと
                        子プロセスを区別するために使用されます。Noneの場合はメインプロセスと見なします。
    """
    global _is_main_logging_configured

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Prevent repeated full configuration in Streamlit's main process ---
    # If target_loggers is None, it's the main application's full setup.
    # We only want this to happen once per process.
    if target_loggers is None:
        if _is_main_logging_configured:
            # Full logging setup already done in this process, return immediately.
            return
        _is_main_logging_configured = True # Mark as configured

        # 初回実行時に既存のハンドラーをクリアして、クリーンな状態を保証します。
        # これはStreamlitの再実行モデルにおいて重要です。
        all_logger_names = list(LOGGING_CONFIG['loggers'].keys())
        all_logger_names.append('') # ルートロガー('')を追加

        for logger_name in all_logger_names:
            logger_obj = logging.getLogger(logger_name)
            for handler in logger_obj.handlers[:]:
                logger_obj.removeHandler(handler)

    # ロギング設定を適用
    dictConfig(LOGGING_CONFIG)