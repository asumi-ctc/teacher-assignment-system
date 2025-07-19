# ==============================================================================
# 4. app_ui_views.py (UI描画関数)
# ==============================================================================
import streamlit as st
import pandas as pd
import logging
from ortools.sat.python import cp_model

# 関連モジュールから必要な関数をインポート
from gemini_utils import get_gemini_explanation
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
        # 表示用にデータを加工
        if 'qualification_special_rank' in df_lecturers_display.columns:
            df_lecturers_display['qualification_special_rank'] = df_lecturers_display['qualification_special_rank'].apply(lambda x: "なし" if pd.isna(x) else x)
        if 'past_assignments' in df_lecturers_display.columns:
            df_lecturers_display['past_assignments_display'] = df_lecturers_display['past_assignments'].apply(
                lambda assignments: f"{len(assignments)}件" if isinstance(assignments, list) else "0件"
            )
        if 'availability' in df_lecturers_display.columns:
            df_lecturers_display['availability_display'] = df_lecturers_display['availability'].apply(lambda dates: f"{len(dates)}日" if isinstance(dates, list) else "0日")
        
        lecturer_display_columns = ["id", "name", "age", "qualification_general_rank", "qualification_special_rank", "availability_display", "past_assignments_display"]
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
    st.header("ソルバーとmodelオブジェクトの概要")
    st.markdown("""
    このアプリケーションは、Googleのオープンソース数理最適化ライブラリ **OR-Tools** の中の **CP-SATソルバー** を使用しています。
    CP-SATソルバーは、**制約充足問題 (Constraint Satisfaction Problem, CSP)** や **組合せ最適化問題** を解くのに非常に強力です。

    ### 1. 変数 (Variables)
    - 各「講師」と各「講座」の組み合わせに対して、割り当ての有無を表すブール変数 `x(講師, 講座)` を作成します。
    - `x(講師, 講座) = 1` であれば割り当て、`0` であれば非割り当てを意味します。

    ### 2. 制約 (Constraints)
    モデルには、絶対に守らなければならないルールである「ハード制約」を追加します。
    - **資格制約**: 講師は、自身の資格ランクより高いランクの講座には割り当てられません。
    - **スケジュール制約**: 講師は、自身の対応可能日以外の講座には割り当てられません。
    - **重複割当制約**: 1つの講座には、最大1人の講師しか割り当てられません。

    ### 3. 目的関数 (Objective Function)
    最適化の目標を数式で表現したものです。CP-SATソルバーは、この目的関数の値を**最小化**（または最大化）しようとします。
    このアプリでは、複数の評価指標（コスト）を重み付けして合計した値を目的関数としています。

    $Minimize(\sum Cost_{total})$

    - **各割り当てのコスト**:
      - **年齢**: 年齢が高いほどコスト増
      - **過去の割当頻度**: 過去の割り当て回数が多いほどコスト増
      - **資格**: 講座ランクに対して資格ランクが高い（オーバースペック）ほどコスト増
      - **同教室での経験**: 同じ教室での割り当て経験が最近であるほどコスト増
    - **特別報酬（負のコスト）**:
      - **連続割り当て**: 特定の条件を満たす連日の講座に同じ講師を割り当てると報酬（コスト減）
    - **ペナルティ**:
      - **未割り当て講座**: 講座が割り当てられない場合に大きなペナルティコストを追加

    サイドバーの「最適化目標」にあるスライダーは、これらの各コストに乗算される**重み**を調整するためのものです。重みを変更することで、どの要素をより重視するかをチューニングできます。
    """)

def display_optimization_result_view(gemini_api_key):
    """「最適化結果」ビューを描画する（完全版）"""
    st.header("最適化結果")

    if not st.session_state.get("solution_executed", False):
        st.info("サイドバーの「最適割り当てを実行」ボタンを押して最適化を実行してください。")
        return

    if "optimization_error_message" in st.session_state and st.session_state.optimization_error_message:
        st.error("最適化処理でエラーが発生しました。詳細は以下をご確認ください。")
        with st.expander("エラー詳細", expanded=True):
            st.code(st.session_state.optimization_error_message, language=None)
        return

    if "solver_result_cache" not in st.session_state or not st.session_state.solver_result_cache:
        st.warning("最適化結果のデータは現在ありません。再度実行してください。")
        return

    solver_result = st.session_state.solver_result_cache
    solution_status = solver_result.get("solution_status", "不明")
    objective_value = solver_result.get("objective_value")

    st.subheader(f"求解ステータス: {solution_status}")

    if objective_value is not None:
        st.metric(label="目的関数値 (総コスト)", value=f"{objective_value:,.2f}")
    else:
        st.metric(label="目的関数値 (総コスト)", value="N/A")

    if "optimization_duration" in st.session_state:
        st.caption(f"計算時間: {st.session_state.optimization_duration:.2f} 秒")
    
    st.markdown("---")
    
    assignments_df = pd.DataFrame(solver_result.get("assignments_df", []))

    if not assignments_df.empty:
        st.subheader("割り当て結果一覧")
        display_columns = [
            "講師名", "講座名", "教室名", "スケジュール", "講師一般ランク", "講師特別ランク", "講座ランク"
        ]
        display_columns = [col for col in display_columns if col in assignments_df.columns]
        st.dataframe(assignments_df[display_columns])
        
        csv = assignments_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
             label="割り当て結果をCSVでダウンロード",
             data=csv,
             file_name='assignment_results.csv',
             mime='text/csv',
        )
    else:
        st.warning("割り当て可能な結果が見つかりませんでした。")

    st.markdown("---")

    with st.expander("詳細ログとAIによる解説", expanded=False):
        st.info("最適化エンジンの生ログと、それを基にしたGeminiによる解説を確認できます。")
        
        log_tabs = st.tabs(["Geminiによる解説", "ソルバーログ", "ゲートウェイログ", "アプリケーションログ"])
        
        with log_tabs[0]:
            if not gemini_api_key:
                st.warning("Gemini APIキーが設定されていません。")
            else:
                if st.button("ログをGeminiに送信して解説を生成"):
                    st.session_state.gemini_api_requested = True
                    engine_log = st.session_state.get("optimization_engine_log_for_download_from_file", "")
                    
                    with st.spinner("Geminiに解説をリクエスト中..."):
                        try:
                            explanation, full_prompt = get_gemini_explanation(
                                log_text=engine_log,
                                api_key=gemini_api_key,
                                solver_status=solution_status,
                                objective_value=objective_value,
                                assignments_summary=assignments_df
                            )
                            st.session_state.gemini_explanation = explanation
                            st.session_state.last_full_prompt_for_gemini = full_prompt
                            st.session_state.gemini_api_error = None
                        except Exception as e:
                            st.session_state.gemini_api_error = f"Gemini APIの呼び出し中にエラーが発生しました: {e}"
                            st.session_state.gemini_explanation = None

                if st.session_state.get("gemini_api_requested"):
                    if st.session_state.get("gemini_explanation"):
                        st.markdown(st.session_state.gemini_explanation)
                    elif st.session_state.get("gemini_api_error"):
                        st.error(st.session_state.gemini_api_error)

        with log_tabs[1]:
            st.code(st.session_state.get("solver_log_for_download", "ソルバーログはありません。"), language="text")
        with log_tabs[2]:
            st.code(st.session_state.get("optimization_gateway_log_for_download", "ゲートウェイログはありません。"), language="text")
        with log_tabs[3]:
            st.code(st.session_state.get("app_log_for_download", "アプリケーションログはありません。"), language="text")

def display_change_assignment_view():
    """「割り当ての変更」ビューを描画する"""
    st.header("割り当ての変更")

    if not st.session_state.get("solution_executed", False) or "solver_result_cache" not in st.session_state:
        st.warning("割り当て結果が存在しないため、この機能は利用できません。まず最適化を実行してください。")
        return
        
    assignments_df = pd.DataFrame(st.session_state.solver_result_cache.get("assignments_df", []))
    if assignments_df.empty:
        st.warning("割り当て結果が空のため、変更する対象がありません。")
        return
    
    st.info("交代させたい講師の割り当てを選択し、「選択した割り当てを交代して再実行」ボタンを押してください。選択された講師と講座のペアは、次の最適化で強制的に割り当て対象外となります。")
    
    # st.data_editor を使ってインタラクティブなテーブルを作成
    edited_df = st.data_editor(
        assignments_df,
        column_config={
            "交代選択": st.column_config.CheckboxColumn(
                "交代させる",
                default=False,
            )
        },
        disabled=assignments_df.columns, # 既存の列は編集不可
        hide_index=True,
        key="assignment_change_editor"
    )

    selected_rows = edited_df[edited_df["交代選択"]]

    if not selected_rows.empty:
        st.write("以下の割り当てを交代対象として選択しました：")
        st.dataframe(selected_rows[["講師名", "講座名", "教室名", "スケジュール"]])
        
        # コールバックに渡すための情報をセッション状態に保存
        # 必要なのは「講師ID」と「講座ID」のペア
        st.session_state.assignments_to_change_list = [
            (row['講師ID'], row['講座ID'], row['講師名'], row['講座名'], row['教室名'])
            for index, row in selected_rows.iterrows()
        ]
    else:
        st.session_state.assignments_to_change_list = []
        
    st.button(
        "選択した割り当てを交代して再実行",
        on_click=handle_execute_changes_callback,
        disabled=selected_rows.empty,
        type="primary"
    )
    
    if "pending_change_summary_info" in st.session_state and st.session_state.pending_change_summary_info:
        st.success("以下の割り当てを交代対象として設定し、再計算を実行しました。")
        summary_df = pd.DataFrame(st.session_state.pending_change_summary_info)
        st.dataframe(summary_df)
        del st.session_state.pending_change_summary_info
        
    if "forced_unassignments_for_solver" in st.session_state and st.session_state.forced_unassignments_for_solver:
        with st.expander("現在設定されている交代対象（強制割当除外リスト）"):
            st.write(st.session_state.forced_unassignments_for_solver)