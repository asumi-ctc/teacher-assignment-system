import logging
import datetime
from typing import List, Dict, Any, Tuple, Optional
 
from .utils.error_definitions import InvalidInputError, ProcessExecutionError
from .utils.types import OptimizationResult, AssignmentResultRow, LecturerData, CourseData, ClassroomData, SolverOutput
# optimization_solver は optimization_solver.py の関数を直接呼び出すため、相対インポート
from . import optimization_solver

logger = logging.getLogger('optimization_gateway')

def run_optimization_with_monitoring(
    lecturers_data: List[LecturerData],
    courses_data: List[CourseData],
    classrooms_data: List[ClassroomData],
    **kwargs: Any
) -> OptimizationResult:
    """
    最適化ソルバーを呼び出し、その結果をStreamlit UI表示用に整形します。
    ソルバーへの入力データの前処理や、ソルバーからの出力の後処理を行います。
    レキシコグラフィカル法に対応した SolverOutput を処理します。

    Args:
        lecturers_data (List[LecturerData]): 講師データのリスト。
        courses_data (List[CourseData]): 講座データのリスト。
        classrooms_data (List[ClassroomData]): 教室データのリスト。
        **kwargs: ソルバーに渡すその他の引数（重み、固定/強制非割り当て、タイムリミットなど）。

    Returns:
        OptimizationResult: 最適化処理の最終結果、ステータス、メッセージ、割り当て詳細などを含む辞書。
    Raises:
        InvalidInputError: 入力データ形式または値が不正な場合。
        ProcessExecutionError: 最適化処理中に予期せぬエラーが発生した場合。
    """
    # ソルバーに直接渡すべきではない、またはUI側で計算された引数をkwargsから削除
    kwargs.pop("allow_under_assignment", None)
    kwargs.pop("weight_assignment_shortage", None)
    kwargs.pop("weight_travel", None)
    kwargs.pop("weight_lecturer_concentration", None) # UIで処理するため削除
    kwargs.pop("solver_time_limit", None) # app.pyから削除されたため、ここでもpopする

    solver_args = {
        "lecturers_data": lecturers_data,
        "courses_data": courses_data,
        "classrooms_data": classrooms_data,
        **kwargs # その他の引数 (重み、fixed_assignments, forced_unassignments, max_assignments_per_lecturerなど)
    }

    try:
        logger.info("最適化ソルバーを呼び出します (レキシコグラフィカル法)...")
        # optimization_solver.solve_assignment を呼び出す
        solver_output: SolverOutput = optimization_solver.solve_assignment(**solver_args)
        logger.info("最適化ソルバーが完了しました。")

    except (ValueError, TypeError) as e:
        logger.error(f"入力データの前処理またはソルバー呼び出し中にエラーが発生しました: {e}", exc_info=True)
        raise InvalidInputError(f"入力データ形式または値に誤りがあります: {e}") from e
    except Exception as e:
        logger.error(f"最適化処理中に予期せぬエラーが発生しました: {e}", exc_info=True)
        raise ProcessExecutionError(f"最適化処理中に予期せぬエラーが発生しました: {e}") from e

    # --- ソルバー出力の結果整形 ---
    logger.info("最適化結果を整形します...")
    # 高速なルックアップのために辞書を作成
    all_lecturers_dict = {l['id']: l for l in solver_output['all_lecturers']}
    all_courses_dict = {c['id']: c for c in solver_output['all_courses']}
    all_classrooms_dict = {c['id']: c for c in classrooms_data} # classrooms_data は solver_args から取得

    processed_assignments: List[AssignmentResultRow] = []
    
    # 各講師の割り当て回数を計算するための仮の辞書
    temp_lecturer_course_counts: Dict[str, int] = {l_id: 0 for l_id in all_lecturers_dict.keys()}
    for assignment in solver_output['assignments']:
        temp_lecturer_course_counts[assignment['講師ID']] += 1

    # 各割り当てについて、講師名、講座名、教室名などの詳細情報を追加
    for assignment in solver_output['assignments']:
        lecturer = all_lecturers_dict.get(assignment['講師ID'], {})
        course = all_courses_dict.get(assignment['講座ID'], {})
        classroom = all_classrooms_dict.get(course.get('classroom_id'), {})
        processed_assignments.append({
            **assignment, # ソルバーからの基本割り当て情報
            "講師名": lecturer.get("name", "不明"),
            "講座名": course.get("name", "不明"),
            "教室ID": course.get("classroom_id", "不明"),
            "スケジュール": course.get("schedule", "不明"),
            "教室名": classroom.get("location", "不明"),
            "講師一般ランク": lecturer.get("qualification_general_rank"),
            "講師特別ランク": lecturer.get("qualification_special_rank", "なし"),
            "講座タイプ": course.get("course_type"),
            "講座ランク": course.get("rank"),
            "今回の割り当て回数": temp_lecturer_course_counts.get(assignment['講師ID'], 0) # ここで正確な割り当て回数を設定
        })

    # 最終的な講師ごとの割り当て回数を再計算（processed_assignmentsから）
    lecturer_course_counts = {lect_id: 0 for lect_id in all_lecturers_dict}
    for assign in processed_assignments:
        lecturer_course_counts[assign['講師ID']] += 1

    # 講座ごとの割り当て回数と残りキャパシティ
    course_assignment_counts, course_remaining_capacity = {}, {}
    for course in solver_output['all_courses']:
        cid = course['id']
        assigned_count = sum(1 for a in processed_assignments if a['講座ID'] == cid)
        # 各講座の割り当てキャパシティは1と仮定（複数割り当ては想定されない）
        assigned_count = min(assigned_count, 1) # 念のため1に上限設定
        course_assignment_counts[cid] = assigned_count
        capacity = 1 
        course_remaining_capacity[cid] = capacity - assigned_count

    # 最終的な OptimizationResult 辞書を構築
    final_result: OptimizationResult = {
        "status": "成功", # デフォルトは成功
        "message": "Optimization process completed successfully.",
        "solution_status": solver_output["solution_status_str"],
        "objective_value": solver_output["objective_value"],
        "assignments_df": processed_assignments,
        "lecturer_course_counts": lecturer_course_counts,
        "course_assignment_counts": course_assignment_counts,
        "course_remaining_capacity": course_remaining_capacity,
        "raw_solver_status_code": solver_output["solver_raw_status_code"],
        "unassigned_courses": solver_output.get("unassigned_courses", []), # フェーズ1で割り当てられなかった講座
        "min_max_assignments_per_lecturer": solver_output.get("min_max_assignments_per_lecturer") # フェーズ2の出力
    }
    
    # フェーズ1のステータスに基づいて、最終結果のステータスとメッセージを調整
    if final_result["solution_status"] in ["PARTIALLY_ASSIGNED", "NO_ASSIGNMENT_POSSIBLE", "INFEASIBLE_PHASE1", "UNKNOWN_FEASIBILITY"]:
        final_result["status"] = "失敗" # フェーズ1で割り当て不能なら最終的に失敗
        if final_result["solution_status"] == "PARTIALLY_ASSIGNED":
            final_result["status"] = "警告" # 部分的に割り当てられた場合は警告
        final_result["message"] = f"Phase 1 failed to assign all courses. Status: {final_result['solution_status']}"

    return final_result
