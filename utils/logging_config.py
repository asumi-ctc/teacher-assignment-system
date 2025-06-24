import logging
import os
from typing import Optional, List # 追加

LOG_DIR = "logs"
APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")
GATEWAY_LOG_FILE = os.path.join(LOG_DIR, "optimization_gateway.log")
ENGINE_LOG_FILE = os.path.join(LOG_DIR, "optimization_engine.log")

def setup_logging(target_loggers: Optional[List[str]] = None):
    """
    アプリケーション全体のロギングを設定する。

    Args:
        target_loggers: 設定対象のロガー名のリスト。
                        Noneの場合は 'app', 'optimization_gateway', 'optimization_engine' の全てを設定する。
                        子プロセスから呼び出す際に、特定のロガーのみを再設定するために使用する。
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # ターゲットロガーとファイルパスの辞書
    all_loggers = {
        'app': APP_LOG_FILE,
        'optimization_gateway': GATEWAY_LOG_FILE,
        'optimization_engine': ENGINE_LOG_FILE
    }

    # 設定対象のロガーを決定
    loggers_to_configure = all_loggers if target_loggers is None else {
        name: path for name, path in all_loggers.items() if name in target_loggers
    }

    # ルートロガーにコンソール出力を設定
    root_logger = logging.getLogger() # ルートロガーを取得
    root_logger.setLevel(logging.INFO)
    
    # ルートロガーの既存ハンドラーを全て削除
    # これにより、Streamlitのリロード時にハンドラーが重複するのを防ぐ
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # ルートロガーにコンソールハンドラを追加
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 各モジュール用のロガーを設定
    for name, log_file in loggers_to_configure.items():
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        # ルートロガーへの伝播を防ぎ、コンソールへの二重出力を回避
        logger.propagate = False
        
        # 既存のハンドラーを全て削除
        # これにより、Streamlitのリロードや子プロセスでの再設定時にハンドラーが重複するのを防ぐ
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        
        # Add new file handler
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # コンソールハンドラを追加
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)