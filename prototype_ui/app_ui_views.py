# ==============================================================================
# 4. app_ui_views.py (UI描画関数)
# ==============================================================================
import streamlit as st
import pandas as pd
import logging
from ortools.sat.python import cp_model
from gemini_utils import get_gemini_explanation, filter_log_for_gemini
from app_callbacks import handle_regenerate_sample_data, handle_generate_invalid_data, handle_execute_changes_callback

logger = logging.getLogger('app')

def display_sample_data_view():
    """「サンプルデータ」ビューを描画する"""
    st.header("入力データ")

    if st.session_state.get("show_regenerate_success_message"):
        st.success("サンプルデータを再生成しました。")
        del st.session_state.show_regenerate_success_message

    if st.session_state.get("show_invalid_data_message"):
        st.warning(f"テスト用の不正データを生成しました: {st.session_state.show_invalid_data_message}")
        del st.session_state.show_invalid_data_message

    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "サンプルデータ再生成",
            key="regenerate_sample_data_button",
            on_click=handle_regenerate_sample_data,
            type="primary",
            help="現在の入力データを破棄し、新しいサンプルデータを生成します。"
        )
    with col2:
        st.button(
            "テスト用不正データ生成",
            key="generate_invalid_data_button",
            on_click=handle_generate_invalid_data,
            help="データバリデーションのテスト用に、意図的に不正なデータを生成します。"
        )

    st.markdown(
        f"**現在の割り当て対象月:** {st.session_state.ASSIGNMENT_TARGET_MONTH_START.strftime('%Y年%m月%d日')} "
        f"～ {st.session_state.ASSIGNMENT_TARGET_MONTH_END.strftime('%Y年%m月%d日')}"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("講師データ (サンプル)")
        df_lecturers_display = pd.DataFrame(st.session_state.DEFAULT_LECTURERS_DATA)
        if 'qualification_special_rank' in df_lecturers_display.columns:
            df_lecturers_display['qualification_special_rank'] = df_lecturers_display['qualification_special_rank'].apply(lambda x: "なし" if x is None else x)
        if 'past_assignments' in df_lecturers_display.columns:
            df_lecturers_display['past_assignments_display'] = df_lecturers_display['past_assignments'].apply(
                lambda assignments: ", ".join([f"{a['classroom_id']}({a['date']})" for a in assignments]) if isinstance(assignments, list) and assignments else "履歴なし"
            )
        if 'availability' in df_lecturers_display.columns:
            df_lecturers_display['availability_display'] = df_lecturers_display['availability'].apply(lambda dates: ", ".join([d.strftime('%Y-%m-%d') for d in dates]) if isinstance(dates, list) else "")
        lecturer_display_columns = ["id", "name", "age", "home_classroom_id", "qualification_general_rank", "qualification_special_rank", "availability_display", "past_assignments_display"]
        st.dataframe(df_lecturers_display[lecturer_display_columns], height=200)
    with col2:
        st.subheader("講座データ (サンプル)")
        df_courses_display = pd.DataFrame(st.session_state.DEFAULT_COURSES_DATA)
        course_display_columns = ["id", "name", "classroom_id", "course_type", "rank", "schedule"]
        st.dataframe(df_courses_display[course_display_columns], height=200)

    st.subheader("教室データ (サンプル)")
    df_classrooms = pd.DataFrame(st.session_state.DEFAULT_CLASSROOMS_DATA)
    st.dataframe(df_classrooms[['id', 'location']])

def display_objective_function_view():
    """「ソルバーとmodelオブジェクト」ビューを描画する"""
    st.header("ソルバーとmodelオブジェクト")
    # (このビューのMarkdownコンテンツは変更がないため、元のapp.pyからコピーしてください)
    st.markdown("...")

def display_optimization_result_view(gemini_api_key):
    """「最適化結果」ビューを描画する"""
    st.header("最適化結果")
    
    if not st.session_state.get("solution_executed", False):
        st.info("サイドバーの「最適割り当てを実行」ボタンを押して最適化を実行してください。")
        return

    if "optimization_error_message" in st.session_state and st.session_state.optimization_error_message:
        st.error("最適化処理でエラーが発生しました。詳細は以下をご確認ください。")
        with st.expander("エラー詳細", expanded=True):
            st.code(st.session_state.optimization_error_message, language=None)
        return

    if "solver_result_cache" not in st.session_state:
        st.warning("最適化結果のデータは現在ありません。再度実行してください。")
        return
        
    solver_result = st.session_state.solver_result_cache
    st.subheader(f"求解ステータス: {solver_result['solution_status']}")
    # (以降の結果表示ロジックは元のapp.pyからコピーしてください)
    st.markdown("...")

def display_change_assignment_view():
    """「割り当ての変更」ビューを描画する"""
    st.header("割り当ての変更")
    if not st.session_state.get("solution_executed", False) or \
       "solver_result_cache" not in st.session_state or \
       not st.session_state.solver_result_cache.get("assignments_df"):
        st.warning("割り当て結果が存在しないため、この機能は利用できません。まず最適化を実行してください。")
        return
    
    # (以降の割り当て変更UIロジックは元のapp.pyからコピーしてください)
    st.markdown("...")
