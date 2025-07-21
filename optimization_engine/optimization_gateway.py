import logging
import datetime
from typing import List, Dict, Any, Tuple, Optional
 
from .utils.error_definitions import InvalidInputError, ProcessExecutionError
from .utils.types import OptimizationResult, AssignmentResultRow, LecturerData, CourseData, ClassroomData, SolverOutput
from . import optimization_solver

logger = logging.getLogger(__name__)

def run_optimization_with_monitoring(
    lecturers_data: List[LecturerData],
    courses_data: List[CourseData],
    classrooms_data: List[ClassroomData],
    **kwargs: Any
) -> OptimizationResult:
    """
    最適化エンジンを直接呼び出し、結果を整形する。
    入力データの前処理（日付変換など）もここで行う。
    レキシコグラフィカル法に対応した SolverOutput を処理する。
    """
    logger = logging.getLogger('optimization_gateway')

    kwargs.pop("allow_under_assignment", None)
    kwargs.pop("weight_assignment_shortage", None)
    kwargs.pop("weight_travel", None)

    # weight_lecturer_concentration は solver.solve_assignment の引数から削除されたため、popする
    kwargs.pop("weight_lecturer_concentration", None)

    solver_args = {
        "lecturers_data": lecturers_data,
        "courses_data": courses_data,
        "classrooms_data": classrooms_data,
        **kwargs
    }

    try:
        logger.info("最適化ソルバーを直接呼び出します (レキシコグラフィカル法、ユーザー設定上限適用)...")
        solver_output: SolverOutput = optimization_solver.solve_assignment(**solver_args)
        logger.info("最適化ソルバーが完了しました。")

    except (ValueError, TypeError) as e:
        logger.error(f"入力データの前処理またはソルバー呼び出し中にエラーが発生しました: {e}", exc_info=True)
        raise InvalidInputError(f"入力データ形式または値に誤りがあります: {e}") from e
    except Exception as e:
        logger.error(f"最適化処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
        raise ProcessExecutionError(f"最適化処理中に予期せぬエラーが発生しました: {e}") from e

    logger.info("最適化結果を整形します...")
    all_lecturers_dict = {l['id']: l for l in solver_output['all_lecturers']}
    all_courses_dict = {c['id']: c for c in solver_output['all_courses']}
    all_classrooms_dict = {c['id']: c for c in classrooms_data}

    processed_assignments: List[AssignmentResultRow] = []
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
    for course in solver_output['all_courses']:
        cid = course['id']
        assigned_count = sum(1 for a in processed_assignments if a['講座ID'] == cid)
        course_assignment_counts[cid] = assigned_count
        capacity = 1
        course_remaining_capacity[cid] = capacity - assigned_count

    final_result: OptimizationResult = {
        "status": "成功",
        "message": "最適化処理が正常に完了しました。",
        "solution_status": solver_output["solution_status_str"],
        "objective_value": solver_output["objective_value"],
        "assignments_df": processed_assignments,
        "lecturer_course_counts": lecturer_course_counts,
        "course_assignment_counts": course_assignment_counts,
        "course_remaining_capacity": course_remaining_capacity,
        "raw_solver_status_code": solver_output["solver_raw_status_code"],
        "unassigned_courses": solver_output.get("unassigned_courses", []),
        "min_max_assignments_per_lecturer": solver_output.get("min_max_assignments_per_lecturer")
    }
    
    if final_result["solution_status"] in ["PARTIALLY_ASSIGNED", "NO_ASSIGNMENT_POSSIBLE", "INFEASIBLE_PHASE1", "UNKNOWN_FEASIBILITY"]:
        final_result["status"] = "失敗"
        if final_result["solution_status"] == "PARTIALLY_ASSIGNED":
            final_result["status"] = "警告"
        final_result["message"] = f"フェーズ1で全ての講座を割り当てることはできませんでした。ステータス: {final_result['solution_status']}"

    return final_result

