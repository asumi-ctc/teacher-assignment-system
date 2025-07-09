import logging
import datetime
from typing import List, Dict, Any, Tuple, Optional

from utils.logging_config import setup_logging
from utils.error_definitions import InvalidInputError, ProcessExecutionError, ProcessTimeoutError
from utils.types import OptimizationResult, AssignmentResultRow, LecturerData, CourseData, ClassroomData, SolverOutput
import optimization_solver

logger = logging.getLogger(__name__)

def _preprocess_input_data(
    lecturers_data: List[Dict[str, Any]],
    courses_data: List[Dict[str, Any]]
) -> Tuple[List[LecturerData], List[CourseData]]:
    """
    入力データの型を検証・変換する。特に日付文字列をdatetime.dateオブジェクトに変換する。
    変換に失敗した場合は ValueError を送出する。
    """
    processed_lecturers: List[LecturerData] = []
    for i, lect in enumerate(lecturers_data):
        new_lect = lect.copy()
        if 'availability' in new_lect and isinstance(new_lect['availability'], list):
            new_lect['availability'] = [
                datetime.datetime.strptime(d, "%Y-%m-%d").date() if isinstance(d, str) else d
                for d in new_lect['availability']
            ]
        if 'past_assignments' in new_lect and isinstance(new_lect['past_assignments'], list):
            new_past_assignments = []
            for pa in new_lect['past_assignments']:
                new_pa = pa.copy()
                if 'date' in new_pa and isinstance(new_pa['date'], str):
                    new_pa['date'] = datetime.datetime.strptime(new_pa['date'], "%Y-%m-%d").date()
                new_past_assignments.append(new_pa)
            new_lect['past_assignments'] = new_past_assignments
        processed_lecturers.append(new_lect)  # type: ignore

    processed_courses: List[CourseData] = []
    for i, course in enumerate(courses_data):
        new_course = course.copy()
        if 'schedule' in new_course and isinstance(new_course['schedule'], str):
            new_course['schedule'] = datetime.datetime.strptime(new_course['schedule'], "%Y-%m-%d").date()
        processed_courses.append(new_course)  # type: ignore

    return processed_lecturers, processed_courses

def run_optimization_with_monitoring(
    lecturers_data: List[Dict[str, Any]],
    courses_data: List[Dict[str, Any]],
    classrooms_data: List[Dict[str, Any]],
    travel_costs_matrix: Dict[Tuple[str, str], int],
    **kwargs: Any
) -> OptimizationResult:
    """
    最適化エンジンを直接呼び出し、結果を整形する。
    入力データの前処理（日付変換など）もここで行う。
    """
    logger = logging.getLogger(__name__)

    solver_args = {
        "lecturers_data": lecturers_data,
        "courses_data": courses_data,
        "classrooms_data": classrooms_data,
        "travel_costs_matrix": travel_costs_matrix,
        **kwargs
    }

    try:
        # 1. 入力データの前処理 (レベル1: 検証と整形)
        processed_lecturers, processed_courses = _preprocess_input_data(
            solver_args["lecturers_data"], solver_args["courses_data"]
        )
        solver_args["lecturers_data"] = processed_lecturers
        solver_args["courses_data"] = processed_courses
        logger.info("入力データの前処理（日付変換）が完了しました。")

        # 2. ソルバーの直接呼び出し
        logger.info("最適化ソルバーを直接呼び出します...")
        solver_output = optimization_solver.solve_assignment(**solver_args)
        logger.info("最適化ソルバーが完了しました。")

    except (ValueError, TypeError) as e:
        logger.error(f"入力データの前処理またはソルバー呼び出し中にエラーが発生しました: {e}", exc_info=True)
        raise InvalidInputError(f"入力データ形式または値に誤りがあります: {e}") from e
    except Exception as e:
        logger.error(f"最適化処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
        # ProcessExecutionErrorはもはや適切ではないが、既存のUIエラーハンドリングのために利用
        raise ProcessExecutionError(f"最適化処理中に予期せぬエラーが発生しました: {e}") from e

    # 3. 結果の整形
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
