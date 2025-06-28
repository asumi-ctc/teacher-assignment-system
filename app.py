import streamlit as st
# import streamlit.components.v1 as components # 削除
import pandas as pd
import datetime # 日付処理用に追加
import time # 処理時間測定用に追加
import google.generativeai as genai # Gemini API 用
# from streamlit_oauth import OAuth2Component # OIDC認証用 # 削除
# from google.oauth2 import id_token # IDトークン検証用 # 削除
# from google.auth.transport import requests as google_requests # IDトークン検証用 # 削除
import multiprocessing
import random # データ生成用
import os # CPUコア数を取得するために追加
import numpy as np # データ型変換のために追加
# dateutil.relativedelta を使用するため、インストールが必要な場合があります。 (pip install python-dateutil)
from dateutil.relativedelta import relativedelta
import logging # logging モジュールをインポート
from typing import List, Optional, Any, Tuple # TypedDict は optimization_engine に移動

# --- [修正点1] 分離したモジュールをインポート ---
import optimization_gateway
import optimization_solver
from ortools.sat.python import cp_model # solver_raw_status_code の比較等で使用
# ---------------------------------------------

# --- [修正点3] ログ設定を別ファイルに分離し、定数をインポート ---
from utils.logging_config import setup_logging, APP_LOG_FILE, GATEWAY_LOG_FILE, SOLVER_LOG_FILE
# ---


# --- 1. データ定義 (LOG_EXPLANATIONS と _get_log_explanation は削除) --- 
# SolverOutput は optimization_engine.py に移動

# --- Gemini API送信用ログのフィルタリング関数 (グ
