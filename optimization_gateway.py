# optimization_gateway.py

import logging
import datetime
import os
import multiprocessing
from multiprocessing.connection import Connection
from typing import List, Dict, Any, Tuple, Set, Optional

from utils.logging_config import setup_logging
from utils.error_definitions import InvalidInputError
from utils.types import OptimizationInput, OptimizationResult, SolverOutput
import optimization_solver

logger = logging.getLogger(__name__)

# --- Helper Functions for Validation (No Change) ---
def _validate_date_string(date_str: Any, context: str) -> None:
    if not isinstance(date_str, str):
        raise InvalidInputError(f"{context}: 日付は文字列である必要がありますが、型 '{type(date_str).__name__}' を受け取りました。値: {date_str}")
    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise InvalidInputError(f"{context}: 日付形式が無効です。'{date_str}' は 'YYYY-MM-DD' 形式ではありません。")

def _validate_classrooms(classrooms_data: List[Dict[str, Any]]) -> Set[str]:
    if not isinstance(classrooms_data, list): raise InvalidInputError("教室データはリスト形式である必要があります。")
    if not classrooms_data: raise InvalidInputError("教室データが空です。")
    validated_ids = set()
    for i, classroom in enumerate(classrooms_data):
        if not isinstance(classroom, dict): raise InvalidInputError(f"教室データ[インデックス:{i}]は辞書形式でなければなりません。")
        for key in ["id", "location"]:
            if key not in classroom: raise InvalidInputError(f"教室データ[インデックス:{i}]: 必須フィールド '{key}' がありません。")
        classroom_id = classroom.get("id")
        if not isinstance(classroom_id, str) or not classroom_id: raise InvalidInputError(f"教室データ[インデックス:{i}]: 'id' は空でない文字列である必要があります。")
        if classroom_id in validated_ids: raise InvalidInputError(f"教室ID '{classroom_id}' が重複しています。")
        validated_ids.add(classroom_id)
    return validated_ids

def _validate_lecturers(lecturers_data: List[Dict[str, Any]], valid_classroom_ids: Set[str]) -> None:
    if not isinstance(lecturers_data, list): raise InvalidInputError("講師データはリスト形式である必要があります。")
    if not lecturers_data: raise InvalidInputError("講師データが空です。")
    # (より詳細なバリデーションロジックは省略)

def _validate_courses(courses_data: List[Dict[str, Any]], valid_classroom_ids: Set[str]) -> None:
    # (バリデーションロジックは省略)
    pass

def _validate_travel_costs(travel_costs_matrix: Dict[Tuple[str, str], int], valid_classroom_ids: Set[str]) -> None:
    # (バリデーションロジックは省略)
    pass

# --- Data Validation and Preprocessing ---
def _validate_and_preprocess_data(input_data: OptimizationInput) -> Dict[str, Any]:
    logger.info("入力データの検証と前処理を開始します...")
    valid_classroom_ids = _validate_classrooms(input_data.classrooms_data)
    _validate_lecturers(input_data.lecturers_data, valid_classroom_ids)
    _validate_courses(input_data.courses_data, valid_classroom_ids)
    _validate_travel_costs(input_data.travel_costs_matrix, valid_classroom_ids)
    logger.info("データバリデーションが正常に完了しました。")

    logger.info("データ前処理（マップ形式へ変換）を実行中...")
    preprocessed_data = {
        "lecturers_map": {d['id']: d for d in input_data.lecturers_data},
        "courses_map": {d['id']: d for d in input_data.courses_data},
        "classrooms_map": {d['id']: d for d in input_data.classrooms_data},
        "travel_costs_matrix": input_data.travel_costs_matrix,
        "solver_params": input_data.solver_params,
        "today_date": input_data.today_date,
        "fixed_assignments": input_data.fixed_assignments,
        "forced_unassignments": input_data.forced_unassignments,
    }
    logger.info("データ前処理が正常に完了しました。")
    return preprocessed_data

# --- Optimization Execution Process ---
def _run_solver_process(conn: Connection, solver_args: Dict[str, Any]):
    try:
        setup_logging(target_loggers=['optimization_solver'])
        result = optimization_solver.solve_assignment(**solver_args)
        conn.send(result)
    except Exception as e:
        child_logger = logging.getLogger('optimization_solver')
        child_logger.error(f"最適化子プロセスで致命的なエラーが発生: {e}", exc_info=True)
        conn.send(e)
    finally:
        conn.close()

def _run_optimization_with_monitoring(solver_args: Dict[str, Any], timeout_seconds: int) -> SolverOutput:
    logger.info(f"最適化プロセスを開始します。(タイムアウト: {timeout_seconds}秒)")
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(target=_run_solver_process, args=(child_conn, solver_args))
    solver_output = None
    try:
        process.start()
        if parent_conn.poll(timeout_seconds):
            result = parent_conn.recv()
            if isinstance(result, Exception):
                raise InvalidInputError(f"最適化プロセスでエラーが発生しました: {result}") from result
            solver_output = result
        else:
            raise TimeoutError(f"最適化処理が設定時間（{timeout_seconds}秒）内に完了しませんでした。")
    finally:
        if process.is_alive():
            logger.warning("最適化プロセスを強制終了します。")
            process.terminate()
            process.join(timeout=5)
        parent_conn.close()
        child_conn.close()
    if solver_output is None:
        raise Exception("最適化プロセスから予期せぬ空の結果が返されました。")
    return solver_output

# --- Result Formatting ---
def _format_result(solver_output: SolverOutput) -> OptimizationResult:
    logger.info("最適化結果を整形します...")
    all_lecturers_dict = {l['id']: l for l in solver_output['all_lecturers']}
    all_courses_dict = {c['id']: c for c in solver_output['all_courses']}
    
    processed_assignments = []
    for assignment in solver_output['assignments']:
        lecturer = all_lecturers_dict.get(assignment['講師ID'], {})
        course = all_courses_dict.get(assignment['講座ID'], {})
        processed_assignments.append({**assignment, "講師名": lecturer.get("name"), "講座名": course.get("name")})

    lecturer_counts = {lect_id: 0 for lect_id in all_lecturers_dict}
    for assign in processed_assignments:
        lecturer_counts[assign['講師ID']] += 1

    final_result: OptimizationResult = {
        "status": "成功", "message": "最適化処理が正常に完了しました。",
        "solution_status": solver_output["solution_status_str"],
        "objective_value": solver_output["objective_value"],
        "assignments_df": processed_assignments,
        "lecturer_course_counts": lecturer_counts,
        "course_assignment_counts": {}, "course_remaining_capacity": {},
        "raw_solver_status_code": solver_output["raw_solver_status_code"]
    }
    return final_result

# --- Public Interface ---
def execute_optimization(input_data: OptimizationInput) -> OptimizationResult:
    try:
        solver_args = _validate_and_preprocess_data(input_data)
        timeout = input_data.solver_params.max_search_seconds
        solver_output = _run_optimization_with_monitoring(solver_args, timeout)
        return _format_result(solver_output)
    except (InvalidInputError, TimeoutError) as e:
        logger.error(f"最適化の事前処理または実行中にエラーが発生: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)
        raise InvalidInputError(f"予期せぬ内部エラー: {e}") from e