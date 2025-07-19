# ==============================================================================
# 3. app_callbacks.py (UIからのコールバック関数)
# ==============================================================================
import streamlit as st
import logging
import os
import time
# [修正] エンジン側のモジュールを正しいパスからインポート
from optimization_engine import optimization_gateway
from optimization_engine.utils.logging_config import GATEWAY_LOG_FILE, SOLVER_LOG_FILE, APP_LOG_FILE
from optimization_engine.utils.error_definitions import InvalidInputError, ProcessExecutionError, ProcessTimeoutError
from app_data_utils import initialize_app_data, generate_invalid_sample_data

def handle_regenerate_sample_data():
    logger = logging.getLogger('app')
    logger.info("Regenerate sample data button clicked, callback triggered.")
    initialize_app_data(force_regenerate=True)
    st.session_state.show_regenerate_success_message = True

def handle_generate_invalid_data():
    logger = logging.getLogger('app')
    logger.info("Generate invalid data button clicked, callback triggered.")
    description = generate_invalid_sample_data()
    st.session_state.show_invalid_data_message = description

def run_optimization():
    logger = logging.getLogger('app')
    keys_to_clear_on_execute = [
        "solver_result_cache", "solver_log_for_download", "optimization_error_message",
        "optimization_gateway_log_for_download", "app_log_for_download", "gemini_explanation", 
        "gemini_api_requested", "gemini_api_error", "last_full_prompt_for_gemini", "optimization_duration"
    ]
    for key in keys_to_clear_on_execute:
        if key in st.session_state:
            del st.session_state[key]
    logger.info("Cleared previous optimization results from session_state.")

    def read_log_file(log_path: str) -> str:
        try:
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to read log file {log_path}: {e}")
        return ""

    try:
        with st.spinner("最適化計算を実行中..."):
            start_time = time.time()
            
            solver_output = optimization_gateway.run_optimization_with_monitoring(
                lecturers_data=st.session_state.DEFAULT_LECTURERS_DATA,
                courses_data=st.session_state.DEFAULT_COURSES_DATA,
                classrooms_data=st.session_state.DEFAULT_CLASSROOMS_DATA,
                weight_past_assignment_recency=st.session_state.get("weight_past_assignment_exp", 0.5),
                weight_qualification=st.session_state.get("weight_qualification_exp", 0.5),
                weight_age=st.session_state.get("weight_age_exp", 0.5),
                weight_frequency=st.session_state.get("weight_frequency_exp", 0.5),
                weight_consecutive_assignment=st.session_state.get("weight_consecutive_assignment_exp", 0.5),
                today_date=st.session_state.TODAY,
                fixed_assignments=st.session_state.get("fixed_assignments_for_solver"),
                forced_unassignments=st.session_state.get("forced_unassignments_for_solver")
            )
            
            end_time = time.time()
            st.session_state.optimization_duration = end_time - start_time
            st.session_state.solver_result_cache = solver_output
            st.session_state.solution_executed = True
            st.session_state.view_mode = "optimization_result"

    except (InvalidInputError, ProcessExecutionError, ProcessTimeoutError) as e:
        logger.error(f"最適化ゲートウェイでエラーが発生しました: {e}", exc_info=True)
        st.session_state.optimization_error_message = f"最適化処理でエラーが発生しました:\n\n{e}"
        st.session_state.solution_executed = True
        st.session_state.view_mode = "optimization_result"
        st.rerun()

    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)
        import traceback
        st.session_state.optimization_error_message = f"最適化処理中にエラーが発生しました:\n\n{traceback.format_exc()}"
        st.session_state.solution_executed = True
        st.session_state.view_mode = "optimization_result"

    finally:
        st.session_state.optimization_gateway_log_for_download = read_log_file(GATEWAY_LOG_FILE)
        engine_log_content = read_log_file(SOLVER_LOG_FILE)
        st.session_state.optimization_engine_log_for_download_from_file = engine_log_content
        st.session_state.app_log_for_download = read_log_file(APP_LOG_FILE)
        solver_log_lines = []
        if engine_log_content:
            solver_log_prefix = "[OR-Tools Solver]"
            for line in engine_log_content.splitlines():
                if solver_log_prefix in line:
                    solver_log_lines.append(line)
        st.session_state.solver_log_for_download = "\n".join(solver_log_lines)

def handle_execute_changes_callback():
    logger = logging.getLogger('app')
    
    current_forced = st.session_state.get("forced_unassignments_for_solver", [])
    if not isinstance(current_forced, list):
        current_forced = []

    if not st.session_state.get("assignments_to_change_list"):
        st.warning("交代する割り当てが選択されていません。")
        return

    st.session_state.pending_change_summary_info = [
        {"lecturer_id": item[0], "course_id": item[1], "lecturer_name": item[2], "course_name": item[3], "classroom_name": item[4]}
        for item in st.session_state.assignments_to_change_list
    ]
    
    newly_forced_unassignments = [(item[0], item[1]) for item in st.session_state.assignments_to_change_list]
    
    for pair in newly_forced_unassignments:
        if pair not in current_forced:
            current_forced.append(pair)
            
    st.session_state.forced_unassignments_for_solver = current_forced
    st.session_state.assignments_to_change_list = []
    
    run_optimization()
