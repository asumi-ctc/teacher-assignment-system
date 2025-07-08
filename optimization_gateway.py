import logging
import datetime
import os
import multiprocessing
from multiprocessing.connection import Connection
from typing import List, Dict, Any, Tuple, Set, TypedDict, Optional, Union

from utils.logging_config import setup_logging
from utils.error_definitions import InvalidInputError, ProcessExecutionError, ProcessTimeoutError
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
    logger = logging.getLogger(__name__)

    solver_args = {
        "lecturers_data": lecturers_data,
        "courses_data": courses_data,
        "classrooms_data": classrooms_data,
        "travel_costs_matrix": travel_costs_matrix,
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
                    raise ProcessExecutionError(f"最適化プロセスでエラーが発生しました: {result}") from result
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
            all_classrooms_dict = {c['id']: c for c in solver_args['classrooms_data']}
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
            
    raise ProcessTimeoutError("最適化処理が複数回の試行でも設定時間内に完了しませんでした。")
