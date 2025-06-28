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
                        指定された場合、そのロガーに関連する設定のみを適用します。
    """
    global _is_main_logging_configured

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    if target_loggers is None:
        # --- メインプロセス用の完全な設定 ---
        if _is_main_logging_configured:
            return
        _is_main_logging_configured = True

        # 初回実行時に既存のハンドラーをクリアして、クリーンな状態を保証します。
        # これはStreamlitの再実行モデルにおいて重要です。
        all_logger_names = list(LOGGING_CONFIG['loggers'].keys())
        all_logger_names.append('') # ルートロガー('')を追加

        for logger_name in all_logger_names:
            logger_obj = logging.getLogger(logger_name)
            for handler in logger_obj.handlers[:]:
                logger_obj.removeHandler(handler)

        # 完全なロギング設定を適用
        dictConfig(LOGGING_CONFIG)
    else:
        # --- 子プロセス用の部分的な設定 ---
        # 子プロセス用に、指定されたロガーの設定のみを含む部分的な設定辞書を作成します。
        # これにより、他のプロセスのログファイルを上書きするのを防ぎます。
        partial_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': LOGGING_CONFIG['formatters'],
            'handlers': {},
            'loggers': {},
        }
        for logger_name in target_loggers:
            if logger_name in LOGGING_CONFIG['loggers']:
                logger_config = LOGGING_CONFIG['loggers'][logger_name]
                partial_config['loggers'][logger_name] = logger_config
                for handler_name in logger_config.get('handlers', []):
                    if handler_name in LOGGING_CONFIG['handlers']:
                        # ハンドラーがまだ追加されていなければ追加
                        if handler_name not in partial_config['handlers']:
                            partial_config['handlers'][handler_name] = LOGGING_CONFIG['handlers'][handler_name]
        
        if partial_config['loggers']:
            dictConfig(partial_config)