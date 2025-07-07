# app.py (完全・修正版)

import streamlit as st
import pandas as pd
import datetime
import time
import multiprocessing
import random
import os
import numpy as np
from dateutil.relativedelta import relativedelta
import logging

# --- 変更: 必要なモジュールと型定義をインポート ---
import optimization_gateway
from utils.logging_config import setup_logging, APP_LOG_FILE, GATEWAY_LOG_FILE, SOLVER_LOG_FILE
from utils.error_definitions import InvalidInputError
from utils.types import OptimizationInput, SolverParameters, OptimizationWeights
from ortools.sat.python import cp_model

# --- データ生成やUI表示のヘルパー関数群 (変更なし) ---
# (ここに、以前の app.py にあった get_gemini_explanation や
#  generate_*_data, display_*_view などの関数をすべて配置してください)
# ...

# --- 最適化実行のコールバック関数 (新しいインターフェースに対応) ---
def run_optimization():
    logger = logging.getLogger('app')
    logger.info("最適割り当て実行ボタンがクリックされました。")
    
    # 実行前に以前の結果をクリア
    for key in ["solver_result_cache", "optimization_error_message", "optimization_duration"]:
        if key in st.session_state:
            del st.session_state[key]

    try:
        with st.spinner("最適化計算を実行中..."):
            start_time = time.time()
            
            # 1. UIから設定オブジェクトを組み立て
            weights = OptimizationWeights(
                past_assignment_recency=st.session_state.get("weight_past_assignment_exp", 0.5),
                qualification=st.session_state.get("weight_qualification_exp", 0.5),
                travel=st.session_state.get("weight_travel_exp", 0.5),
                age=st.session_state.get("weight_age_exp", 0.5),
                frequency=st.session_state.get("weight_frequency_exp", 0.5),
                assignment_shortage=st.session_state.get("weight_assignment_shortage_exp", 0.5),
                lecturer_concentration=st.session_state.get("weight_lecturer_concentration_exp", 0.5),
                consecutive_assignment=st.session_state.get("weight_consecutive_assignment_exp", 0.5),
            )
            solver_params = SolverParameters(
                weights=weights,
                allow_under_assignment=st.session_state.allow_under_assignment_cb
            )
            
            # 2. 最適化エンジンへの入力オブジェクトを組み立て
            engine_input = OptimizationInput(
                lecturers_data=st.session_state.DEFAULT_LECTURERS_DATA,
                courses_data=st.session_state.DEFAULT_COURSES_DATA,
                classrooms_data=st.session_state.DEFAULT_CLASSROOMS_DATA,
                travel_costs_matrix=st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX,
                solver_params=solver_params,
                today_date=st.session_state.TODAY,
                fixed_assignments=st.session_state.get("fixed_assignments_for_solver"),
                forced_unassignments=st.session_state.get("forced_unassignments_for_solver")
            )

            # 3. 新しいエントリーポイントを呼び出し
            solver_output = optimization_gateway.execute_optimization(engine_input)
            
            # 4. 結果をセッションに保存
            st.session_state.optimization_duration = time.time() - start_time
            st.session_state.solver_result_cache = solver_output
    
    except (InvalidInputError, TimeoutError) as e:
        logger.error(f"処理中にエラーが発生: {e}", exc_info=True)
        st.session_state.optimization_error_message = f"処理中にエラーが発生しました:\n\n{e}"
    except Exception as e:
        logger.error(f"予期せぬシステムエラー: {e}", exc_info=True)
        st.session_state.optimization_error_message = f"予期せぬシステムエラーが発生しました:\n\n{e}"
    
    # 処理完了後、必ず結果表示モードに移行
    st.session_state.solution_executed = True
    st.session_state.view_mode = "optimization_result"

# --- メイン実行ブロック ---
def main():
    setup_logging()
    logger = logging.getLogger('app')
    st.set_page_config(page_title="講師割り当てシステムデモ", layout="wide")

    # 初回起動時にデータ生成
    if "app_data_initialized" not in st.session_state:
        initialize_app_data()
        st.session_state.app_data_initialized = True

    # セッション状態の初期化
    if "view_mode" not in st.session_state: st.session_state.view_mode = "sample_data"
    if "allow_under_assignment_cb" not in st.session_state: st.session_state.allow_under_assignment_cb = True
    
    # --- UI描画 ---
    st.title("講師割り当てシステム(OR-Tools)-プロトタイプ")
    
    # (ここにナビゲーションボタンなどのUIコードを配置)
    # ...

    st.sidebar.button("最適割り当てを実行", type="primary", on_click=run_optimization)
    # (ここにサイドバーの他のUIコードを配置)
    # ...

    # 画面表示の切り替え
    if st.session_state.view_mode == "sample_data":
        display_sample_data_view()
    elif st.session_state.view_mode == "optimization_result":
        display_optimization_result_view(gemini_api_key=st.secrets.get("GEMINI_API_KEY"))
    # ... (他のビュー)

if __name__ == "__main__":
    # (multiprocessing の設定)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()