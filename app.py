# app.py

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

import optimization_gateway
from optimization_gateway import OptimizationInput, SolverParameters, OptimizationWeights
from utils.logging_config import setup_logging, APP_LOG_FILE, GATEWAY_LOG_FILE, SOLVER_LOG_FILE
from utils.error_definitions import InvalidInputError

# ... (データ生成関数などは変更なし) ...

def run_optimization():
    logger = logging.getLogger('app')
    # ... (セッションクリアのロジック) ...

    try:
        with st.spinner("最適化計算を実行中..."):
            start_time = time.time()
            
            # 1. 設定オブジェクトの組み立て
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
            
            # 2. 入力オブジェクト全体の組み立て
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
            
            end_time = time.time()
            st.session_state.optimization_duration = end_time - start_time
            st.session_state.solver_result_cache = solver_output
            st.session_state.solution_executed = True
            st.session_state.view_mode = "optimization_result"

    except (InvalidInputError, TimeoutError) as e:
        logger.error(f"処理中にエラーが発生: {e}", exc_info=True)
        st.session_state.optimization_error_message = f"処理中にエラーが発生しました:\n\n{e}"
        st.session_state.solution_executed = True
        st.session_state.view_mode = "optimization_result"
        # st.rerun() をtry-except-finallyブロックの外で行うか、即時更新が不要なら削除を検討
    except Exception as e:
        logger.error(f"予期せぬシステムエラー: {e}", exc_info=True)
        st.session_state.optimization_error_message = f"予期せぬシステムエラーが発生しました:\n\n{e}"
        st.session_state.solution_executed = True
        st.session_state.view_mode = "optimization_result"
    finally:
        # ... (ログ読み込みロジック) ...
        pass
    
    st.rerun() # 処理の最後に一度だけ実行してUIを更新