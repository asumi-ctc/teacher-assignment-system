import logging
import datetime
import os
import multiprocessing
from multiprocessing.connection import Connection
from typing import List, Dict, Any, Tuple, Set, TypedDict, Optional, Union

from utils.logging_config import setup_logging
from utils.error_definitions import InvalidInputError 
import optimization_solver

logger = logging.getLogger(__name__)

class OptimizationResult(TypedDict):
    status: str
    message: str
    solution_status: str
    objective_value: Optional[float]
    assignments_df: List[Dict[str, Union[str, int, float, None]]]
    lecturer_course_counts: Dict[str, int]
    course_assignment_counts: Dict[str, int]
    course_remaining_capacity: Dict[str, int]
    raw_solver_status_code: int

RETRY_LIMIT = 2
PROCESS_TIMEOUT_SECONDS = 90

def _run_solver_process(conn: Connection, solver_args: Dict[str, Any]):
    """子プロセスで実行されるソルバー呼び出しラッパー"""
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

def run_optimization_with_monitoring(
    lecturers_data: List[Dict[str, Any]],
    courses_data: List[Dict[str, Any]],
    classrooms_data: List[Dict[str, Any]],
    travel_costs_matrix: Dict[Tuple[str, str], int],
    **kwargs: Any
) -> OptimizationResult:
    """
    最適化エンジンを監視付きの別プロセスで実行し、タイムアウトや再試行を管理する。
    """
    # --- 1. 入力データの検証と前処理 ---
    try:
        logger.info("Starting input data validation and preprocessing.")
        if not isinstance(lecturers_data, list): raise InvalidInputError("`lecturers_data` must be a list.")
        if not isinstance(courses_data, list): raise InvalidInputError("`courses_data` must be a list.")
        if not isinstance(classrooms_data, list): raise InvalidInputError("`classrooms_data` must be a list.")
        if not isinstance(travel_costs_matrix, dict): raise InvalidInputError("`travel_costs_matrix` must be a dict.")

        if lecturers_data:
            required_keys = {'id', 'availability', 'home_classroom_id', 'qualification_general_rank'}
            for i, lecturer in enumerate(lecturers_data):
                if not isinstance(lecturer, dict):
                    raise InvalidInputError(f"Item at index {i} in `lecturers_data` is not a dict.")
                missing_keys = required_keys - lecturer.keys()
                if missing_keys:
                    lecturer_id = lecturer.get('id', 'N/A')
                    raise InvalidInputError(f"Lecturer at index {i} (id: {lecturer_id}) is missing keys: {missing_keys}")

        if courses_data:
            required_keys = {'id', 'classroom_id', 'course_type', 'rank', 'schedule'}
            for i, course in enumerate(courses_data):
                if not isinstance(course, dict):
                    raise InvalidInputError(f"Item at index {i} in `courses_data` is not a dict.")
                missing_keys = required_keys - course.keys()
                if missing_keys:
                    course_id = course.get('id', 'N/A')
                    raise InvalidInputError(f"Course at index {i} (id: {course_id}) is missing keys: {missing_keys}")

        # データ前処理: リストをIDをキーとする辞書に変換
        lecturers_dict = {lecturer['id']: lecturer for lecturer in lecturers_data}
        courses_dict = {course['id']: course for course in courses_data}
        classrooms_dict = {classroom['id']: classroom for classroom in classrooms_data}
        # パフォーマンスを安定させるため、元のリストの順序をIDのリストとしてソルバーに渡す
        lecturer_ids_in_order = [lecturer['id'] for lecturer in lecturers_data]
        course_ids_in_order = [course['id'] for course in courses_data]
        logger.info("Input data validation and preprocessing successful.")

    except InvalidInputError:
        logger.error("Input data validation failed.", exc_info=True)
        # この例外は呼び出し元 (app.py) で捕捉されるように再送出する
        raise

    solver_args = {
        # ソルバーには前処理済みの辞書形式のデータを渡す
        "lecturers_dict": lecturers_dict,
        "courses_dict": courses_dict,
        "classrooms_dict": classrooms_dict,
        "travel_costs_matrix": travel_costs_matrix,
        "lecturer_ids_in_order": lecturer_ids_in_order,
        "course_ids_in_order": course_ids_in_order,
        **kwargs
    }

    for attempt in range(RETRY_LIMIT):
        logger.info(f"最適化プロセスを開始します。(試行 {attempt + 1}/{RETRY_LIMIT})")
        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        process = multiprocessing.Process(target=_run_solver_process, args=(child_conn, solver_args))
        solver_output = None
        try:
            process.start()
            if parent_conn.poll(PROCESS_TIMEOUT_SECONDS):
                result = parent_conn.recv()
                if isinstance(result, Exception):
                    raise InvalidInputError(f"最適化プロセスでエラーが発生しました: {result}") from result
                solver_output = result
                logger.info("最適化プロセスから結果を正常に受信しました。")
            else:
                logger.error(f"最適化プロセスがタイムアウトしました ({PROCESS_TIMEOUT_SECONDS}秒)。")
        except (IOError, EOFError) as e:
            logger.error(f"プロセス間通信中にエラーが発生しました: {e}", exc_info=True)
        finally:
            if process.is_alive():
                logger.error("最適化プロセスが応答しません。強制終了します。")
                process.terminate()
                process.join(timeout=5)
            parent_conn.close()
            child_conn.close()
        
        if solver_output is not None:
            logger.info("最適化プロセスが正常に完了しました。")
            logger.info("最適化結果を整形します...")
            all_lecturers_dict = {l['id']: l for l in solver_output['all_lecturers']}
            all_courses_dict = {c['id']: c for c in solver_output['all_courses']}
            all_classrooms_dict = solver_args['classrooms_dict'] # classrooms_dictを直接使用
            processed_assignments = []
            for assignment in solver_output['assignments']:
                lecturer = all_lecturers_dict.get(assignment['講師ID'], {})
                course = all_courses_dict.get(assignment['講座ID'], {})
                classroom = all_classrooms_dict.get(course.get('classroom_id'), {})
                processed_assignments.append({
                    **assignment,
                    "講師名": lecturer.get("name", "不明"),
                    "講座名": course.get("name", "不明"),
                    "教室ID": course.get("classroom_id", "不明"),
                    "スケジュール": course.get("schedule", "不明"),
                    "教室名": classroom.get("location", "不明"),
                    "講師一般ランク": lecturer.get("qualification_general_rank"),
                    "講師特別ランク": lecturer.get("qualification_special_rank", "なし"),
                    "講座タイプ": course.get("course_type"),
                    "講座ランク": course.get("rank"),
                })
            
            lecturer_course_counts = {lect_id: 0 for lect_id in all_lecturers_dict}
            for assign in processed_assignments:
                lecturer_course_counts[assign['講師ID']] += 1
            
            course_assignment_counts, course_remaining_capacity = {}, {}
            TARGET_PREFECTURES = ["東京都", "愛知県", "大阪府"]
            for course in solver_output['all_courses']:
                cid = course['id']
                assigned_count = sum(1 for a in processed_assignments if a['講座ID'] == cid)
                course_assignment_counts[cid] = assigned_count
                location = all_classrooms_dict.get(course.get('classroom_id'), {}).get('location')
                capacity = 2 if location in TARGET_PREFECTURES else 1
                course_remaining_capacity[cid] = capacity - assigned_count

            final_result: OptimizationResult = {
                "status": "成功", "message": "最適化処理が正常に完了しました。",
                "solution_status": solver_output["solution_status_str"],
                "objective_value": solver_output["objective_value"],
                "assignments_df": processed_assignments,
                "lecturer_course_counts": lecturer_course_counts,
                "course_assignment_counts": course_assignment_counts,
                "course_remaining_capacity": course_remaining_capacity,
                "raw_solver_status_code": solver_output["solver_raw_status_code"]
            }
            return final_result
        
        if attempt < RETRY_LIMIT - 1:
            logger.info("再試行します...")
            continue
            
    raise InvalidInputError("最適化処理が複数回の試行でも設定時間内に完了しませんでした。")
