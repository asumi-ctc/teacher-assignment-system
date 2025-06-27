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

def setup_logging(target_loggers: Optional[List[str]] = None):
    """
    アプリケーション全体のロギングを設定する。

    Args:
        target_loggers: 設定対象のロガー名のリスト。
                        Noneの場合は 'app', 'optimization_gateway', 'optimization_solver' の全てを設定する。
                        子プロセスから呼び出す際に、特定のロガーのみを再設定するために使用する。
    """
    global _is_main_logging_configured

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # ターゲットロガーとファイルパスの辞書
    all_loggers_files = { # ロガー名と対応するファイルパスのマップ
        'app': APP_LOG_FILE,
        'optimization_gateway': GATEWAY_LOG_FILE,
        'optimization_solver': SOLVER_LOG_FILE # optimization_engine.log から変更済み
    }

    # 設定対象のロガーを決定
    loggers_to_configure_map = all_loggers_files if target_loggers is None else {
        name: path for name, path in all_loggers_files.items() if name in target_loggers
    }
    
    # --- Prevent repeated full configuration in Streamlit's main process ---
    # If target_loggers is None, it's the main application's full setup.
    # We only want this to happen once per process.
    if target_loggers is None:
        if _is_main_logging_configured:
            # Full logging setup already done in this process, return immediately.
            return
        _is_main_logging_configured = True # Mark as configured

        # For the very first full configuration, clear root logger handlers
        # to ensure a clean slate for the main app's console output.
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # --- Clear handlers for specific loggers being configured ---
    # This applies to both the main setup (for app, gateway, solver loggers)
    # and child process setups (for optimization_solver logger).
    # This is necessary because disable_existing_loggers is False.
    for logger_name in loggers_to_configure_map.keys():
        logger_obj = logging.getLogger(logger_name)
        for handler in logger_obj.handlers[:]: # ハンドラーリストのコピーをイテレート
            logger_obj.removeHandler(handler)

    # dictConfig 形式のロギング設定辞書を構築
    LOGGING_CONFIG_DICT = {
        'version': 1,
        'disable_existing_loggers': False, # 既存ロガーの設定を無効化しない (Djangoとの互換性のため)

        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': { # コンソール出力用ハンドラー
                'class': 'logging.StreamHandler', # クラスは文字列で指定
                'formatter': 'standard'
            },
            # 各モジュール用のファイルハンドラーを動的に追加
            **{
                f'{name}_file': { # ハンドラー名は '{ロガー名}_file' とする
                    'class': 'logging.FileHandler', # RotatingFileHandler から FileHandler に変更
                    'mode': 'w', # オリジナルの挙動に合わせて 'w' (上書き) に設定
                    'filename': os.path.abspath(log_file), # 各ロガーのファイルパス (絶対パスに変換)
                    'formatter': 'standard',
                    'level': 'INFO', # ファイルへの書き込みレベルは INFO で固定 (必要に応じて調整)
                    'maxBytes': 1024 * 1024 * 5, # 5 MB
                    'backupCount': 5, # 5世代のバックアップ
                } for name, log_file in loggers_to_configure_map.items()
            }
        },
        'loggers': {
            # 各モジュールロガーの設定
            **{
                name: {
                    'handlers': ['console', f'{name}_file'], # コンソールとファイルの両方に出力
                    'level': 'INFO', # ロガーのレベルは INFO で固定 (必要に応じて調整)
                    'propagate': False, # ルートロガーへの伝播を停止
                } for name in loggers_to_configure_map.keys()
            },
            'django': { # Django 自身のログ（settings.pyへの移植時に必要）
                'handlers': ['console'], # Djangoのログはsettings.pyで別途ファイルハンドラを持つことが多い
                'level': 'INFO',
                'propagate': False,
            }
        },
        'root': { # ルートロガー: 上記で捕捉されない全てのログのデフォルト
            'handlers': ['console'], # ルートはコンソールのみに設定し、各ロガーがファイルハンドラを持つ
            'level': 'INFO', # ルートロガーの最低レベル
            # propagate: True はデフォルトなので不要
        },
    }

    # ロギング設定を適用
    dictConfig(LOGGING_CONFIG_DICT)