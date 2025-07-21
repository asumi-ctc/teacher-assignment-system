import streamlit as st
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import time
import json
from pathlib import Path # ファイル読み込み用に追加

# Streamlit環境で安全にmultiprocessingを使用するため、'spawn'メソッドを強制的に設定
# これはアプリケーションの起動時に一度だけ実行されるべき。
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Streamlitの内部的な再実行サイクルなどで既に設定されている場合があるので無視する
    pass

import logging # logging モジュールをインポート

# --- 分離したモジュールをインポート ---
from optimization_engine import optimization_gateway
from optimization_engine.utils.error_definitions import InvalidInputError, ProcessExecutionError, ProcessTimeoutError
from optimization_engine.utils.logging_config import setup_logging, APP_LOG_FILE, GATEWAY_LOG_FILE, SOLVER_LOG_FILE
from ortools.sat.python import cp_model # solver_raw_status_code の比較等で使用
from typing import Optional, Any, Tuple, List, Dict # 型ヒント用

# --- ロギング設定をアプリケーション起動時に一度だけ実行 ---
setup_logging()
logger = logging.getLogger('app')

# --- 外部ファイルからデータを読み込む関数 ---
def load_initial_data_from_files(data_dir: str = "data"):
    """
    指定されたディレクトリから講師、講座、教室のデータをJSONファイルとして読み込む。
    データが読み込めない場合はエラーを発生させる。
    """
    data_path = Path(data_dir)
    lecturers_file = data_path / "lecturers.json"
    courses_file = data_path / "courses.json"
    classrooms_file = data_path / "classrooms.json"

    try:
        lecturers_data = json.loads(lecturers_file.read_text(encoding="utf-8"))
        courses_data = json.loads(courses_file.read_text(encoding="utf-8"))
        classrooms_data = json.loads(classrooms_file.read_text(encoding="utf-8"))

        # 日付文字列をdatetime.dateオブジェクトに変換（必要に応じて）
        # LecturerDataのavailabilityとpast_assignments['date']、CourseDataのschedule
        for lecturer in lecturers_data:
            if 'availability' in lecturer and isinstance(lecturer['availability'], list):
                lecturer['availability'] = [datetime.date.fromisoformat(d) if isinstance(d, str) else d for d in lecturer['availability']]
            if 'past_assignments' in lecturer and isinstance(lecturer['past_assignments'], list):
                for pa in lecturer['past_assignments']:
                    if 'date' in pa and isinstance(pa['date'], str):
                        pa['date'] = datetime.date.fromisoformat(pa['date'])
        for course in courses_data:
            if 'schedule' in course and isinstance(course['schedule'], str):
                course['schedule'] = datetime.date.fromisoformat(course['schedule'])

        logger.info(f"Initial data loaded successfully from {data_dir}.")
        return lecturers_data, courses_data, classrooms_data

    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e.filename}. Please ensure data files are generated in '{data_dir}'.", exc_info=True)
        st.error(f"エラー: 必要なデータファイルが見つかりません。`{data_dir}` ディレクトリにデータファイルが生成されていることを確認してください。")
        st.stop() # アプリの実行を停止
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from data files: {e}", exc_info=True)
        st.error(f"エラー: データファイルの読み込み中にJSON形式のエラーが発生しました。ファイルが破損している可能性があります。")
        st.stop()
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading initial data: {e}", exc_info=True)
        st.error(f"エラー: 初期データの読み込み中に予期せぬエラーが発生しました: {e}")
        st.stop()

# --- コールバック関数 ---
def run_optimization():
    """最適化を実行し、結果をセッション状態に保存するコールバック関数"""
    logger = logging.getLogger('app')
    keys_to_clear_on_execute = [
        "solver_result_cache",
        "solver_log_for_download", "optimization_error_message",
        "optimization_gateway_log_for_download",
        "app_log_for_download", "optimization_duration" # Gemini関連のキーを削除
    ]
    for key in keys_to_clear_on_execute:
        if key in st.session_state:
            del st.session_state[key]
    logger.info("Cleared previous optimization results from session_state.")

    def read_log_file(log_path: str) -> str:
        """ログファイルを読み込んで内容を返す。存在しない場合は空文字列を返す。"""
        try:
            if Path(log_path).exists():
                with open(log_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to read log file {log_path}: {e}")
        return ""

    try:
        with st.spinner("最適化計算を実行中..."):
            start_time = time.time() # 処理時間測定開始
            
            logger.info("Starting optimization calculation (optimization_gateway.run_optimization_with_monitoring).")
            solver_output = optimization_gateway.run_optimization_with_monitoring(
                lecturers_data=st.session_state.DEFAULT_LECTURERS_DATA,
                courses_data=st.session_state.DEFAULT_COURSES_DATA,
                classrooms_data=st.session_state.DEFAULT_CLASSROOMS_DATA,
                weight_past_assignment_recency=st.session_state.get("weight_past_assignment_exp", 0.5),
                weight_qualification=st.session_state.get("weight_qualification_exp", 0.5),
                weight_age=st.session_state.get("weight_age_exp", 0.5),
                weight_frequency=st.session_state.get("weight_frequency_exp", 0.5),
                weight_lecturer_concentration=st.session_state.get("weight_lecturer_concentration_exp", 0.5),
                weight_consecutive_assignment=st.session_state.get("weight_consecutive_assignment_exp", 0.5),
                today_date=st.session_state.TODAY,
                fixed_assignments=st.session_state.get("fixed_assignments_for_solver"),
                forced_unassignments=st.session_state.get("forced_unassignments_for_solver")
            )
            
            end_time = time.time() # 処理時間測定終了
            elapsed_time = end_time - start_time
            logger.info(f"Optimization process took {elapsed_time:.2f} seconds.")
            st.session_state.optimization_duration = elapsed_time # 結果をセッションに保存

            logger.info("solve_assignment completed.")

            if not isinstance(solver_output, dict):
                raise TypeError(f"最適化関数の戻り値が不正です。型: {type(solver_output).__name__}")

            required_keys = ["status", "message", "solution_status", "objective_value", "assignments_df", "lecturer_course_counts", "course_assignment_counts", "course_remaining_capacity", "raw_solver_status_code"]
            missing_keys = [key for key in required_keys if key not in solver_output]
            if missing_keys:
                raise KeyError(f"最適化関数の戻り値に必要なキーが不足しています。不足キー: {missing_keys}")

            st.session_state.solver_result_cache = solver_output

            if "fixed_assignments_for_solver" in st.session_state: del st.session_state.fixed_assignments_for_solver
            if "forced_unassignments_for_solver" in st.session_state: del st.session_state.forced_unassignments_for_solver

            st.session_state.solution_executed = True
            st.session_state.view_mode = "optimization_result"

    except (InvalidInputError, ProcessExecutionError, ProcessTimeoutError) as e:
        logger.error(f"最適化ゲートウェイでエラーが発生しました: {e}", exc_info=True)
        # エラーの種類に応じてユーザーへのメッセージを調整
        if isinstance(e, InvalidInputError):
            error_message = f"入力データの検証中にエラーが発生しました:\n\n{e}"
        elif isinstance(e, ProcessTimeoutError):
            error_message = f"最適化処理がタイムアウトしました:\n\n{e}"
        elif isinstance(e, ProcessExecutionError):
            error_message = f"最適化プロセスの実行中にエラーが発生しました:\n\n{e}"
        else:
            error_message = f"予期せぬ最適化エラーが発生しました:\n\n{e}"

        st.session_state.optimization_error_message = error_message
        # ログダウンロード用に空の文字列を設定
        st.session_state.solver_log_for_download = ""
        st.session_state.app_log_for_download = ""
        # UIにエラーを表示するための設定
        st.session_state.solution_executed = True
        st.session_state.view_mode = "optimization_result"
        st.rerun() # UIを即時更新してエラーを表示

    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)
        import traceback
        error_trace = traceback.format_exc()
        st.session_state.optimization_error_message = f"最適化処理中にエラーが発生しました:\n\n{error_trace}"
        st.session_state.solver_log_for_download = ""
        st.session_state.app_log_for_download = ""
        st.session_state.solution_executed = True
        st.session_state.view_mode = "optimization_result"

    finally:
        # 処理の最後にログファイルを読み込む
        logger.info("Reading log files to store in session state.")
        st.session_state.optimization_gateway_log_for_download = read_log_file(GATEWAY_LOG_FILE)
        # optimization_engine のログは直接ファイルから読み込む
        engine_log_content = read_log_file(SOLVER_LOG_FILE)
        st.session_state.optimization_engine_log_for_download_from_file = engine_log_content
        st.session_state.app_log_for_download = read_log_file(APP_LOG_FILE)

        # OR-Toolsソルバーログを抽出してダウンロード用に設定
        solver_log_lines = []
        if engine_log_content:
            solver_log_prefix = "[OR-Tools Solver]"
            for line in engine_log_content.splitlines():
                if solver_log_prefix in line:
                    solver_log_lines.append(line)
        st.session_state.solver_log_for_download = "\n".join(solver_log_lines)
        logger.info(f"Extracted {len(solver_log_lines)} lines of OR-Tools solver log for download.")

        logger.info("Finished reading log files.")

def handle_execute_changes_callback():
    """
    選択された割り当ての講師を変更し、再最適化を実行するコールバック関数。
    この関数は、UIからの割り当て変更要求を処理します。
    """
    logger = logging.getLogger('app')
    logger.info(
        f"Callback: Executing changes for {len(st.session_state.get('assignments_to_change_list', []))} selected assignments."
    )
    
    current_forced = st.session_state.get("forced_unassignments_for_solver", [])
    # current_forcedがリストでない場合やNoneの場合に初期化
    if not isinstance(current_forced, list):
        current_forced = []
        logger.warning("forced_unassignments_for_solver was not a list or None, re-initialized to empty list.")

    if not st.session_state.get("assignments_to_change_list"):
        st.warning("交代する割り当てが選択されていません。")
        logger.warning("handle_execute_changes_callback called with empty assignments_to_change_list.")
        return

    # 変更サマリー表示用に情報を保存
    st.session_state.pending_change_summary_info = [
        {
            "lecturer_id": item[0], "course_id": item[1],
            "lecturer_name": item[2], "course_name": item[3],
            "classroom_name": item[4] # 教室名も追加
        }
        for item in st.session_state.assignments_to_change_list
    ]
    logger.info(f"Pending change summary info: {st.session_state.pending_change_summary_info}")

    # ソルバーに渡す強制非割り当てリストを準備
    newly_forced_unassignments = [
        (item[0], item[1]) for item in st.session_state.assignments_to_change_list
    ]
    
    # 既存の強制非割り当てと結合し、重複を避ける
    for pair in newly_forced_unassignments:
        if pair not in current_forced:
            current_forced.append(pair)
            
    st.session_state.forced_unassignments_for_solver = current_forced # セッション状態を更新
    logger.info(f"forced_unassignments_for_solver updated to: {st.session_state.forced_unassignments_for_solver}")
    
    # 選択リストをクリア
    st.session_state.assignments_to_change_list = []

    # メインの最適化ロジックをトリガー
    run_optimization()

# --- Streamlit UI 表示関数 ---
def display_optimization_result_view():
    """「最適化結果」ビューを描画する"""
    logger = logging.getLogger('app')
    st.header("最適化結果")
    logger.info("Displaying optimization result.")

    if not st.session_state.get("solution_executed", False):
        st.info("サイドバーの「最適割り当てを実行」ボタンを押して最適化を実行してください。")
    else: # solution_executed is True
        if "solver_result_cache" not in st.session_state:
            if "optimization_error_message" in st.session_state and st.session_state.optimization_error_message:
                logger.warning("Optimization error occurred. Displaying error message.")
                st.error("最適化処理でエラーが発生しました。詳細は以下をご確認ください。")
                with st.expander("エラー詳細", expanded=True):
                    st.code(st.session_state.optimization_error_message, language=None)
            else:
                logger.info("No solver_result_cache and no optimization_error_message. Prompting user to run optimization.")
                st.warning(
                    "最適化結果のデータは現在ありません。\n"
                    "再度結果を表示するには、サイドバーの「最適割り当てを実行」ボタンを押してください。"
                )
        else: # solution_executed is True and solver_result_cache exists
            logger.info("solver_result_cache found. Displaying results.")
            solver_result = st.session_state.solver_result_cache
            st.subheader(f"求解ステータス: {solver_result['solution_status']}")

            metric_cols = st.columns(2)
            with metric_cols[0]:
                if solver_result['objective_value'] is not None:
                    st.metric("総コスト (目的値)", f"{solver_result['objective_value']:.2f}")
            with metric_cols[1]:
                if 'optimization_duration' in st.session_state:
                    st.metric("処理時間", f"{st.session_state.optimization_duration:.2f} 秒", help="データ準備から最適化完了までの時間です。")

            if solver_result['raw_solver_status_code'] in [cp_model.FEASIBLE, cp_model.UNKNOWN]:
                st.warning(
                    """
                    時間制限(Time Limit)内に最適解が見つかりませんでした。現在の最良の解を表示します。

                    もう一度やり直す場合は、余り必要としない最適化目標の重みを0.0に設定することで、その最適化目標が除外されて計算時間が短縮される可能性があります。
                    """
                )

            if solver_result.get('assignments_df'):
                results_df = pd.DataFrame(solver_result['assignments_df'])

                assigned_course_ids = set(results_df["講座ID"])
                unassigned_courses = [c for c in st.session_state.DEFAULT_COURSES_DATA if c["id"] not in assigned_course_ids]
                if not unassigned_courses:
                    st.success("全ての講座が割り当てられました。")

                st.subheader("割り当て結果サマリー")
                summary_data = []
                summary_data.append(("**総割り当て件数**", f"{len(results_df)}件"))

                assigned_lecturer_ids = results_df["講師ID"].unique()
                temp_assigned_lecturers = [l for l in st.session_state.DEFAULT_LECTURERS_DATA if l["id"] in assigned_lecturer_ids]

                if temp_assigned_lecturers:
                    avg_age = sum(l.get("age", 0) for l in temp_assigned_lecturers) / len(temp_assigned_lecturers)
                    summary_data.append(("**平均年齢**", f"{avg_age:.1f}才"))
                    avg_frequency = sum(len(l.get("past_assignments", [])) for l in temp_assigned_lecturers) / len(temp_assigned_lecturers)
                    summary_data.append(("**平均頻度**", f"{avg_frequency:.1f}回"))

                    summary_data.append(("**一般資格ランク別割り当て**", "(講師が保有する一般資格ランク / 全講師中の同ランク保有者数)"))
                    general_rank_total_counts = {i: 0 for i in range(1, 6)}
                    for lecturer in st.session_state.DEFAULT_LECTURERS_DATA:
                        rank = lecturer.get("qualification_general_rank")
                        if rank in general_rank_total_counts:
                            general_rank_total_counts[rank] += 1
                    assigned_general_rank_counts = {i: 0 for i in range(1, 6)}
                    for l_assigned in temp_assigned_lecturers:
                        rank = l_assigned.get("qualification_general_rank")
                        if rank in assigned_general_rank_counts:
                            assigned_general_rank_counts[rank] += 1
                    for rank_num in range(1, 6):
                        summary_data.append((f"　一般ランク{rank_num}", f"{assigned_general_rank_counts.get(rank_num, 0)}人 / {general_rank_total_counts.get(rank_num, 0)}人中"))

                    summary_data.append(("**特別資格ランク別割り当て**", "(講師が保有する特別資格ランク / 全講師中の同ランク保有者数)"))
                    special_rank_total_counts = {i: 0 for i in range(1, 6)}
                    for lecturer in st.session_state.DEFAULT_LECTURERS_DATA:
                        rank = lecturer.get("qualification_special_rank")
                        if rank is not None and rank in special_rank_total_counts:
                            special_rank_total_counts[rank] += 1
                    assigned_special_rank_counts = {i: 0 for i in range(1, 6)}
                    for l_assigned in temp_assigned_lecturers:
                        rank = l_assigned.get("qualification_special_rank")
                        if rank is not None and rank in assigned_special_rank_counts:
                            assigned_special_rank_counts[rank] += 1
                    for rank_num in range(1, 6):
                        summary_data.append((f"　特別ランク{rank_num}", f"{assigned_special_rank_counts.get(rank_num, 0)}人 / {special_rank_total_counts.get(rank_num, 0)}人中"))

                if '今回の割り当て回数' in results_df.columns:
                    counts_of_lecturers_by_assignment_num = results_df['講師ID'].value_counts().value_counts().sort_index()
                    summary_data.append(("**講師の割り当て回数別**", "(今回の最適化での担当講座数)"))
                    for num_assignments, num_lecturers in counts_of_lecturers_by_assignment_num.items():
                        if num_assignments >= 1:
                            summary_data.append((f"　{num_assignments}回 担当した講師", f"{num_lecturers}人"))

                past_assignment_new_count = results_df[results_df["当該教室最終割当日からの日数"] < 0].shape[0]
                past_assignment_existing_count = results_df.shape[0] - past_assignment_new_count
                summary_data.append(("**同教室への過去の割り当て**", "(実績なし優先コスト計算に基づく)"))
                summary_data.append(("　新規", f"{past_assignment_new_count}人"))
                summary_data.append(("　割当て実績あり", f"{past_assignment_existing_count}人"))

                markdown_table = "| 項目 | 値 |\n| :---- | :---- |\n"
                for item, value in summary_data:
                    markdown_table += f"| {item} | {value} |\n"
                st.markdown(markdown_table)
                st.markdown("---")

                # --- 割り当て変更サマリーの表示 ---
                if "pending_change_summary_info" in st.session_state and st.session_state.pending_change_summary_info:
                    st.subheader("今回の割り当て変更による影響")
                    change_details_markdown = ""
                    for change_item in st.session_state.pending_change_summary_info:
                        original_lecturer_id = change_item['lecturer_id']
                        original_lecturer_name = change_item['lecturer_name']
                        course_id_changed = change_item['course_id']
                        course_name_changed = change_item['course_name']

                        new_assignment_for_course_df = results_df[results_df['講座ID'] == course_id_changed]

                        new_assignment_str = "割り当てなし"
                        if not new_assignment_for_course_df.empty:
                            new_lecturers_info = [f"{new_row['講師名']} (`{new_row['講師ID']}`)" for _, new_row in new_assignment_for_course_df.iterrows()]
                            new_assignment_str = "、".join(new_lecturers_info)

                        change_details_markdown += f"- **講座:** {course_name_changed} (`{course_id_changed}`)\n  - **変更前:** {original_lecturer_name} (`{original_lecturer_id}`)\n  - **変更後:** {new_assignment_str}\n"

                    if change_details_markdown:
                        st.markdown(change_details_markdown)
                    st.markdown("---")
                    del st.session_state.pending_change_summary_info # 表示後にクリア

                # --- 詳細結果と未割り当て講座 ---
                st.subheader("割り当て結果詳細")
                st.dataframe(results_df)
                st.markdown("---")

                if unassigned_courses:
                    st.subheader("割り当てられなかった講座")
                    st.dataframe(pd.DataFrame(unassigned_courses))
                    st.caption("上記の講座は、制約（資格ランクなど）により割り当て可能な講師が見つからなかったか、または他の割り当てと比較してコストが高すぎると判断された可能性があります。")

            else: # solver_result['assignments'] が存在しないか、空の場合
                if solver_result['raw_solver_status_code'] in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                    st.error("解が見つかりましたが、実際の割り当ては行われませんでした。")
                    st.warning(
                        """
                        考えられる原因:
                        - 割り当て可能な講師と講座のペアが元々存在しない (制約が厳しすぎる、データ不適合)。
                        - 結果として、総コスト 0.00 (何も割り当てない) が最適と判断された可能性があります。
                        """
                    )
                    st.subheader("全ての講座が割り当てられませんでした")
                    st.dataframe(pd.DataFrame(st.session_state.DEFAULT_COURSES_DATA))
                elif solver_result['raw_solver_status_code'] == cp_model.INFEASIBLE:
                    st.error("実行不可能な割り当てです。")
                    st.warning(
                        """
                        指定された条件では、制約を満たす割り当てパターンが見つかりませんでした。

                        **考えられる主な原因:**
                        - **制約の競合:** 「全ての講座に必ず1名割り当てる」というルールが厳格化されたため、一人の講師が担当できる唯一の講座が複数存在する場合などに、制約が満たせなくなります。

                        **対処法の例:**
                        - 講師の対応可能日を増やす。
                        - 割り当て対象の講座を減らす。
                        - 「割り当ての変更」機能で、競合していそうな講師の割り当てを強制的に交代させてみる。
                        """
                    )
                else:
                    st.error(solver_result['solution_status_str'])

            # ログダウンロードセクション
            if st.session_state.get("solution_executed"):
                st.markdown("---") # 区切り線
                st.subheader("ログのダウンロード")
                dl_cols_1 = st.columns(2)
                dl_cols_2 = st.columns(2)

                with dl_cols_1[0]:
                    if st.session_state.get("optimization_gateway_log_for_download"):
                        st.download_button(
                            label="最適化ゲートウェイのログ",
                            data=st.session_state.optimization_gateway_log_for_download,
                            file_name="optimization_gateway.log",
                            mime="text/plain",
                            key="download_optimization_gateway_log_button",
                            help="データバリデーション、プロセス監視、最適化エンジン呼び出しに関するログです。"
                        )
                with dl_cols_1[1]:
                    if st.session_state.get("optimization_engine_log_for_download_from_file"):
                        st.download_button(
                            label="最適化エンジン内部ログ",
                            data=st.session_state.optimization_engine_log_for_download_from_file,
                            file_name="optimization_engine_internal.log",
                            mime="text/plain",
                            key="download_engine_internal_log_button",
                            help="最適化エンジンの内部でキャプチャされた、割り当て候補のフィルタリングやコスト計算、制約構築などの詳細ログです。"
                        )
                with dl_cols_2[0]:
                    if st.session_state.get("solver_log_for_download"):
                        st.download_button(
                            label="OR-Toolsソルバーのログ",
                            data=st.session_state.solver_log_for_download,
                            file_name="solver_log.txt",
                            mime="text/plain",
                            key="download_solver_log_button",
                            help="OR-Toolsソルバーが生成した、求解過程に関する技術的なログです。"
                        )
                with dl_cols_2[1]:
                    if st.session_state.get("app_log_for_download"):
                        st.download_button(
                            label="その他のシステムログ",
                            data=st.session_state.app_log_for_download,
                            file_name="app.log",
                            mime="text/plain",
                            key="download_app_log_button",
                            help="UI操作やアプリケーション全体の高レベルなイベントに関するログです。"
                        )
            logger.info("Optimization result display complete.")

def display_change_assignment_view():
    """「割り当ての変更」ビューを描画する"""
    logger = logging.getLogger('app')
    st.header("割り当ての変更")
    logger.info("Displaying change assignment view.")

    if not st.session_state.get("solution_executed", False) or \
       "solver_result_cache" not in st.session_state or \
       not st.session_state.solver_result_cache.get("assignments_df"):
        st.warning("割り当て結果が存在しないため、この機能は利用できません。まず最適化を実行してください。")
    else: # 割り当て結果が存在する場合
        solver_result = st.session_state.solver_result_cache
        results_df = pd.DataFrame(solver_result['assignments_df'])

        if results_df.empty:
            st.info("変更対象の割り当てがありません。")
        else:
            st.markdown("交代させたい講師の割り当てを選択し、「交代リスト」に追加してください。リスト作成後、「選択した割り当ての講師を変更して再最適化」ボタンで実行します。")

            # --- 検索フィルター ---
            st.subheader("割り当て検索フィルター")
            filter_cols = st.columns(3)
            with filter_cols[0]:
                search_lecturer_name = st.text_input("講師名で絞り込み", key="change_search_lecturer_name").lower()
            with filter_cols[1]:
                search_course_name = st.text_input("講座名で絞り込み", key="change_search_course_name").lower()
            with filter_cols[2]:
                search_classroom_name = st.text_input("教室名で絞り込み", key="change_search_classroom_name").lower()

            filtered_assignments_df = results_df.copy()
            if search_lecturer_name:
                filtered_assignments_df = filtered_assignments_df[filtered_assignments_df['講師名'].str.lower().str.contains(search_lecturer_name, na=False)]
            if search_course_name:
                filtered_assignments_df = filtered_assignments_df[filtered_assignments_df['講座名'].str.lower().str.contains(search_course_name, na=False)]
            if search_classroom_name: # 教室名で検索 (results_df に '教室名' がある前提)
                if '教室名' in filtered_assignments_df.columns:
                    filtered_assignments_df = filtered_assignments_df[filtered_assignments_df['教室名'].str.lower().str.contains(search_classroom_name, na=False)]
                else:
                    st.warning("結果データに教室名列が見つかりません。教室IDでの絞り込みを試みてください。")

            st.markdown("---")
            st.subheader("現在の割り当て一覧 (フィルター結果)")
            if filtered_assignments_df.empty:
                st.info("検索条件に一致する割り当てがありません。")
            else:
                for index, row in filtered_assignments_df.iterrows():
                    item_tuple = (
                        row['講師ID'], row['講座ID'],
                        row['講師名'], row['講座名'],
                        row['教室名'], row['スケジュール'] # 教室名とスケジュールもタプルに含める
                    )
                    is_selected = item_tuple in st.session_state.assignments_to_change_list

                    checkbox_label = f"講師: {row['講師名']} (`{row['講師ID']}`), 講座: {row['講座名']} (`{row['講座ID']}`), 教室: {row['教室名']} @ {row['スケジュール']}"

                    # チェックボックスの状態変更で直接リストを更新
                    if st.checkbox(checkbox_label, value=is_selected, key=f"cb_change_{row['講師ID']}_{row['講座ID']}"):
                        if not is_selected: # 以前選択されていなくて、今チェックされた
                            st.session_state.assignments_to_change_list.append(item_tuple)
                    else: # チェックが外された場合
                        if is_selected: # 以前選択されていて、今チェックが外された
                            st.session_state.assignments_to_change_list.remove(item_tuple)
                    st.markdown("---")

            # --- 交代リストの表示と管理 ---
            st.sidebar.markdown("---")
            st.sidebar.subheader("交代予定の割り当てリスト")
            if not st.session_state.assignments_to_change_list:
                st.sidebar.info("交代する割り当てはありません。")
            else:
                for i, item in enumerate(st.session_state.assignments_to_change_list):
                    # item: (lecturer_id, course_id, lecturer_name, course_name, classroom_name, schedule)
                    st.sidebar.markdown(f"- **講師:** {item[2]}, **講座:** {item[3]}")
                    if st.sidebar.button(f"リストから削除 ({item[2]}-{item[3]})", key=f"remove_change_{item[0]}_{item[1]}_{i}"):
                        st.session_state.assignments_to_change_list.pop(i)
                        st.rerun() # リスト変更を即時反映
                st.sidebar.markdown("---")

            # --- 変更実行ボタン ---
            if st.button("選択した割り当ての講師を変更して再最適化",
                            type="primary",
                            use_container_width=True,
                            disabled=not st.session_state.assignments_to_change_list,
                            on_click=handle_execute_changes_callback):
                pass # on_click で処理される
    logger.info("Change assignment view display complete.")

def main():
    # --- Streamlitアプリの基本設定 ---
    st.set_page_config(page_title="講師割り当てシステムデモ", layout="wide")
    st.title("講師割り当てシステム(OR-Tools)-プロトタイプ")

    # --- セッション状態の初期化と初期データ読み込み ---
    # アプリケーション起動時に一度だけ実行
    if "app_data_initialized" not in st.session_state:
        logger.info("Initializing app data and loading from files.")
        st.session_state.DEFAULT_LECTURERS_DATA, \
        st.session_state.DEFAULT_COURSES_DATA, \
        st.session_state.DEFAULT_CLASSROOMS_DATA = load_initial_data_from_files()
        
        # 割り当て対象月の設定 (現在の4ヶ月後)
        st.session_state.TODAY = datetime.date.today()
        assignment_target_month_start_val = (st.session_state.TODAY + relativedelta(months=4)).replace(day=1)
        st.session_state.ASSIGNMENT_TARGET_MONTH_START = assignment_target_month_start_val
        next_month_val = assignment_target_month_start_val + relativedelta(months=1)
        st.session_state.ASSIGNMENT_TARGET_MONTH_END = next_month_val - datetime.timedelta(days=1)
        logger.info(f"Assignment target month set: {st.session_state.ASSIGNMENT_TARGET_MONTH_START} to {st.session_state.ASSIGNMENT_TARGET_MONTH_END}")

        st.session_state.app_data_initialized = True
        logger.info("'app_data_initialized' set to True.")
    else:
        logger.info("App data already initialized. Skipping data loading.")

    # その他のセッション状態変数
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "optimization_result" # デフォルトビュー
    if "assignments_to_change_list" not in st.session_state:
        st.session_state.assignments_to_change_list = []
    if "solution_executed" not in st.session_state:
        st.session_state.solution_executed = False

    # --- UIの描画 ---
    # ナビゲーションボタン (最適化結果と割り当て変更のみ)
    nav_cols = st.columns([1, 1]) # ボタン数を2つに調整
    with nav_cols[0]:
        if st.button("最適化結果", use_container_width=True, type="primary" if st.session_state.view_mode == "optimization_result" else "secondary"):
            st.session_state.view_mode = "optimization_result"
            st.rerun()
    with nav_cols[1]:
        # 「割り当て結果を変更」ボタンは、最適化が実行されて結果がある場合のみ表示
        if st.session_state.get("solution_executed", False) and \
           st.session_state.get("solver_result_cache") and \
           st.session_state.solver_result_cache.get("assignments_df"):
            if st.button("割り当て結果を変更", use_container_width=True, type="primary" if st.session_state.view_mode == "change_assignment_view" else "secondary"):
                st.session_state.view_mode = "change_assignment_view"
                st.rerun()
        else:
            # 最適化がまだ実行されていない、または結果がない場合はボタンを無効化または非表示
            st.button("割り当て結果を変更", use_container_width=True, disabled=True, help="最適化を実行すると有効になります。")
    
    st.sidebar.markdown(
        "【制約】【許容条件】【最適化目標】を設定すれば、数理モデル最適化手法により自動的に最適な講師割り当てを実行します。"
        "また最適化目標に重み付けすることで割り当て結果をチューニングすることができます。"
    )
    st.sidebar.button("最適割り当てを実行", type="primary", on_click=run_optimization)
    st.sidebar.markdown("---")

    with st.sidebar.expander("【制約】", expanded=False):
        st.markdown("- 1.講師は、資格ランクを超える講座への割り当てはできない")
        st.markdown("- 2.講師は、個人スケジュールに適合しない講座への割り当てはできない。")
        st.markdown("- 3.講座には、出来るだけ1名を割り当てる。（ソフト制約）")

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
        st.markdown("**講師の割り当て集中度を低くする（今回の割り当て内）**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど、一人の講師が今回の最適化で複数の講座を担当することへのペナルティが大きくなります。", key="weight_lecturer_concentration_exp")

        st.markdown("**連日講座への連続割り当てを優先**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど、特別資格を持つ講師が一般講座と特別講座の連日ペアをまとめて担当することを重視します（報酬が増加）。", key="weight_consecutive_assignment_exp")

    # --- メインエリアの表示制御 ---
    logger.info(f"Starting main area display. Current view_mode: {st.session_state.view_mode}")
    if st.session_state.view_mode == "optimization_result":
        display_optimization_result_view()
    elif st.session_state.view_mode == "change_assignment_view":
        display_change_assignment_view()
    else: # view_mode が予期せぬ値の場合 (フォールバック)
        logger.warning(f"Unexpected view_mode: {st.session_state.view_mode}. Displaying fallback info.")
        st.info("サイドバーから表示するデータを選択してください。")
    logger.info("Exiting main function.")

if __name__ == "__main__":
    main()
