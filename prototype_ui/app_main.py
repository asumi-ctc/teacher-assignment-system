# ==============================================================================
# 1. app_main.py (アプリケーションのエントリーポイント)
# ==============================================================================
import sys
import os
import streamlit as st
import multiprocessing
import logging

# --- [修正] プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
# これにより、app_main.pyから見て上の階層にあるoptimization_engineフォルダを
# importできるようになります。
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# --------------------------------------------------------------------

# --- 分離したモジュールをインポート ---
from optimization_engine.utils.logging_config import setup_logging
from app_data_utils import initialize_app_data
from app_callbacks import (
    run_optimization, 
    handle_regenerate_sample_data, 
    handle_generate_invalid_data
)
from app_ui_views import (
    display_sample_data_view, 
    display_objective_function_view, 
    display_optimization_result_view, 
    display_change_assignment_view
)

def main():
    # --- ロガーやデータ初期化など ---
    setup_logging()
    logger = logging.getLogger('app')
    st.set_page_config(page_title="講師割り当てシステムデモ", layout="wide")
    initialize_app_data() # 初回呼び出し
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

    # --- セッション状態の初期化 ---
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "sample_data"
    if "assignments_to_change_list" not in st.session_state:
        st.session_state.assignments_to_change_list = []
    if "solution_executed" not in st.session_state:
        st.session_state.solution_executed = False

    # --- UIの描画 ---
    st.title("講師割り当てシステム(OR-Tools)-プロトタイプ")

    # --- ナビゲーションボタン ---
    nav_cols = st.columns([2, 2, 2, 1])
    with nav_cols[0]:
        if st.button("サンプルデータ", use_container_width=True, type="primary" if st.session_state.view_mode == "sample_data" else "secondary"):
            st.session_state.view_mode = "sample_data"
            st.rerun()
    with nav_cols[1]:
        if st.button("ソルバーとmodelオブジェクト", use_container_width=True, type="primary" if st.session_state.view_mode == "objective_function" else "secondary"):
            st.session_state.view_mode = "objective_function"
            st.rerun()
    with nav_cols[2]:
        if st.session_state.get("solution_executed", False):
            if st.button("最適化結果", use_container_width=True, type="primary" if st.session_state.view_mode == "optimization_result" else "secondary"):
                st.session_state.view_mode = "optimization_result"
                st.rerun()

    # --- サイドバー ---
    st.sidebar.markdown(
        "【制約】【許容条件】【最適化目標】を設定すれば、数理モデル最適化手法により自動的に最適な講師割り当てを実行します。"
        "また最適化目標に重み付けすることで割り当て結果をチューニングすることができます。"
    )
    st.sidebar.button("最適割り当てを実行", type="primary", on_click=run_optimization)
    st.sidebar.markdown("---")

    if st.session_state.get("solution_executed", False) and \
       st.session_state.get("solver_result_cache") and \
       st.session_state.solver_result_cache.get("assignments_df"):
        if st.sidebar.button("割り当て結果を変更", key="change_assignment_view_button", type="secondary" if st.session_state.view_mode != "change_assignment_view" else "primary"):
            st.session_state.view_mode = "change_assignment_view"
            st.rerun()
    st.sidebar.markdown("---")

    with st.sidebar.expander("【制約】", expanded=False):
        st.markdown("- 1.講師は、資格ランクを超える講座への割り当てはできない")
        st.markdown("- 2.講師は、個人スケジュールに適合しない講座への割り当てはできない。")

    with st.sidebar.expander("【最適化目標】", expanded=False):
        st.caption(
            "各最適化目標の相対的な重要度を重みで設定します。\n"
            "不要な最適化目標は重みを0にしてください（最適化目標から除外されます）。"
        )
        st.markdown("**年齢の若い人を優先**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど年齢が若い人を重視します。", key="weight_age_exp")
        st.markdown("**割り当て頻度の少ない人を優先**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど全講座割当回数が少ない人を重視します。", key="weight_frequency_exp")
        st.markdown("**講師資格が高い人を優先**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど講師資格ランクが高い人が重視されます。", key="weight_qualification_exp")
        st.markdown("**同教室への割り当て実績が無い人を優先**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど同教室への割り当て実績が無い人、或いは最後に割り当てられた日からの経過日数が長い人が重視されます。", key="weight_past_assignment_exp")
        st.markdown("**連日講座への連続割り当てを優先**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど、特別資格を持つ講師が一般講座と特別講座の連日ペアをまとめて担当することを重視します（報酬が増加）。", key="weight_consecutive_assignment_exp")

    # --- メインエリアの表示制御 ---
    logger.info(f"Starting main area display. Current view_mode: {st.session_state.view_mode}")
    if st.session_state.view_mode == "sample_data":
        display_sample_data_view()
    elif st.session_state.view_mode == "objective_function":
        display_objective_function_view()
    elif st.session_state.view_mode == "optimization_result":
        display_optimization_result_view(gemini_api_key=GEMINI_API_KEY)
    elif st.session_state.view_mode == "change_assignment_view":
        display_change_assignment_view()
    else:
        logger.warning(f"Unexpected view_mode: {st.session_state.view_mode}. Displaying fallback info.")
        st.info("サイドバーから表示するデータを選択してください。")
    logger.info("Exiting main function.")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
