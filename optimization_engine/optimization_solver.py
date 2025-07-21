import logging
import datetime
from typing import List, Dict, Any, Tuple, Optional, Callable
import pandas as pd
from ortools.sat.python import cp_model

from .utils.error_definitions import InvalidInputError, ProcessExecutionError, ProcessTimeoutError, SolverError
from .utils.types import LecturerData, CourseData, ClassroomData, SolverOutput, SolverAssignment

logger = logging.getLogger(__name__)

DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT = 365 * 10 # 10年

def solve_assignment(
    lecturers_data: List[LecturerData],
    courses_data: List[CourseData],
    classrooms_data: List[ClassroomData],
    weight_past_assignment_recency: float,
    weight_qualification: float,
    weight_age: float,
    weight_frequency: float,
    # weight_lecturer_concentration: float, # この重みはUIスライダーの制御ロジックに移行するため、solverからは削除
    weight_consecutive_assignment: float,
    today_date: datetime.date,
    fixed_assignments: Optional[List[Tuple[str, str]]] = None,
    forced_unassignments: Optional[List[Tuple[str, str]]] = None,
    time_limit_seconds: int = 60,
    max_assignments_per_lecturer: Optional[int] = None # UIの集中度スライダーから計算された上限値
) -> SolverOutput:
    """
    レキシコグラフィカル法（3フェーズ）を用いて講師割り当て問題を解くメイン関数。
    フェーズ1: 全講座割り当て可能性の確認
    フェーズ2: 講師割り当て回数の最小化（min_max_assignmentsの決定）
    フェーズ3: ユーザー設定の「集中度」から計算された割り当て上限を制約とした最終最適化
    """
    logger.info("レキシコグラフィカル法による最適化を開始します。")

    # --- 1. データの前処理と準備（既存ロジックを維持） ---
    lecturers_dict = {l['id']: l for l in lecturers_data}
    courses_dict = {c['id']: c for c in courses_data}
    classrooms_dict = {c['id']: c for c in classrooms_data}

    def parse_date_if_str(date_obj: Any) -> Optional[datetime.date]:
        if isinstance(date_obj, str):
            try:
                return datetime.date.fromisoformat(date_obj)
            except ValueError:
                logger.warning(f"Invalid date format encountered: {date_obj}. Returning None.")
                return None
        return date_obj

    for lecturer_id, lecturer in lecturers_dict.items():
        if 'availability' in lecturer and isinstance(lecturer['availability'], list):
            lecturer['availability'] = [parse_date_if_str(d) for d in lecturer['availability']]
            lecturer['availability'] = [d for d in lecturer['availability'] if d is not None]
        if 'past_assignments' in lecturer and isinstance(lecturer['past_assignments'], list):
            for pa in lecturer['past_assignments']:
                if 'date' in pa:
                    pa['date'] = parse_date_if_str(pa['date'])
    
    for course_id, course in courses_dict.items():
        if 'schedule' in course:
            course['schedule'] = parse_date_if_str(course['schedule'])

    fixed_assignments_set = set(fixed_assignments) if fixed_assignments else set()
    forced_unassignments_set = set(forced_unassignments) if forced_unassignments else set()

    # --- 2. 割り当て候補の生成とコスト計算（既存ロジックを維持） ---
    possible_assignments_details = []
    
    consecutive_day_pairs = []
    courses_by_date_classroom = {}
    for course_id, course in courses_dict.items():
        if course['schedule'] and course['classroom_id']:
            key = (course['schedule'], course['classroom_id'])
            if key not in courses_by_date_classroom:
                courses_by_date_classroom[key] = []
            courses_by_date_classroom[key].append(course_id)

    for course_id_1, course_1 in courses_dict.items():
        if course_1['schedule'] and course_1['course_type'] == 'general':
            next_day_date = course_1['schedule'] + datetime.timedelta(days=1)
            if (next_day_date, course_1['classroom_id']) in courses_by_date_classroom:
                for course_id_2 in courses_by_date_classroom[(next_day_date, course_1['classroom_id'])]:
                    course_2 = courses_dict[course_id_2]
                    if course_2['course_type'] == 'special':
                        for lecturer_id, lecturer in lecturers_dict.items():
                            if lecturer['qualification_special_rank'] is not None:
                                if course_1['schedule'] in lecturer['availability'] and \
                                   course_2['schedule'] in lecturer['availability'] and \
                                   lecturer['qualification_special_rank'] <= course_1['rank'] and \
                                   lecturer['qualification_special_rank'] <= course_2['rank']:
                                    consecutive_day_pairs.append({
                                        "lecturer_id": lecturer_id,
                                        "course1_id": course_id_1,
                                        "course2_id": course_id_2,
                                        "classroom_id": course_1['classroom_id'],
                                        "dates": (course_1['schedule'], course_2['schedule'])
                                    })
    
    MAX_AGE_COST = 65 - 22
    MAX_FREQUENCY_COST = 12
    MAX_QUALIFICATION_COST = 5
    MAX_RECENCY_DAYS = 365 * 2

    for lecturer_id, lecturer in lecturers_dict.items():
        for course_id, course in courses_dict.items():
            if course['schedule'] not in lecturer['availability']:
                continue

            is_qualified = False
            if course['course_type'] == 'general':
                if lecturer['qualification_general_rank'] <= course['rank']:
                    is_qualified = True
                elif lecturer['qualification_special_rank'] is not None:
                    is_qualified = True
            elif course['course_type'] == 'special':
                if lecturer['qualification_special_rank'] is not None and lecturer['qualification_special_rank'] <= course['rank']:
                    is_qualified = True
            
            if not is_qualified:
                continue

            assignment_pair = (lecturer_id, course_id)
            if assignment_pair in forced_unassignments_set:
                continue
            
            # --- コスト計算 ---
            age_cost = (lecturer['age'] - 22) / MAX_AGE_COST if MAX_AGE_COST > 0 else 0
            frequency_cost = len(lecturer.get('past_assignments', [])) / MAX_FREQUENCY_COST if MAX_FREQUENCY_COST > 0 else 0

            qualification_cost = 0
            if course['course_type'] == 'general':
                qualification_cost = (lecturer['qualification_general_rank'] - course['rank']) / MAX_QUALIFICATION_COST if MAX_QUALIFICATION_COST > 0 else 0
            elif course['course_type'] == 'special' and lecturer['qualification_special_rank'] is not None:
                qualification_cost = (lecturer['qualification_special_rank'] - course['rank']) / MAX_QUALIFICATION_COST if MAX_QUALIFICATION_COST > 0 else 0
            qualification_cost = max(0, qualification_cost)

            recency_cost = 0
            last_assignment_date_in_classroom = None
            for pa in lecturer.get('past_assignments', []):
                if pa['classroom_id'] == course['classroom_id'] and pa['date'] is not None:
                    if last_assignment_date_in_classroom is None or pa['date'] > last_assignment_date_in_classroom:
                        last_assignment_date_in_classroom = pa['date']
            
            days_since_last_assignment = DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT
            if last_assignment_date_in_classroom:
                days_since_last_assignment = (today_date - last_assignment_date_in_classroom).days
            
            recency_cost = max(0, (MAX_RECENCY_DAYS - days_since_last_assignment) / MAX_RECENCY_DAYS) if MAX_RECENCY_DAYS > 0 else 0

            total_scaled_cost = (
                age_cost * weight_age +
                frequency_cost * weight_frequency +
                qualification_cost * weight_qualification +
                recency_cost * weight_past_assignment_recency
            ) * 1000

            cost = int(round(total_scaled_cost))

            possible_assignments_details.append({
                "lecturer_id": lecturer_id,
                "course_id": course_id,
                "cost": cost,
                "age_cost_raw": age_cost,
                "frequency_cost_raw": frequency_cost,
                "qualification_cost_raw": qualification_cost,
                "recency_days": days_since_last_assignment,
                "is_fixed": assignment_pair in fixed_assignments_set
            })

    # --- 3. レキシコグラフィカル法による求解 ---
    final_assignments: List[SolverAssignment] = []
    final_objective_value: Optional[float] = None
    final_solution_status_str: str = "UNKNOWN"
    final_raw_solver_status_code: int = cp_model.UNKNOWN
    unassigned_courses_list: List[Dict[str, Any]] = []
    determined_min_max_assignments: Optional[int] = None # フェーズ2の出力

    # --- フェーズ1: 全講座割り当て可能性の確認 ---
    logger.info("フェーズ1開始: 全講座割り当て可能性の確認...")
    model_phase1 = cp_model.CpModel()
    solver_phase1 = cp_model.CpSolver()
    solver_phase1.parameters.log_search_progress = True
    solver_phase1.parameters.max_time_in_seconds = time_limit_seconds

    x_vars_phase1: Dict[Tuple[str, str], cp_model.BoolVar] = {}
    
    for assign_detail in possible_assignments_details:
        lecturer_id = assign_detail["lecturer_id"]
        course_id = assign_detail["course_id"]
        
        x_var = model_phase1.NewBoolVar(f'x_{lecturer_id}_{course_id}_P1')
        x_vars_phase1[(lecturer_id, course_id)] = x_var
        
        if assign_detail["is_fixed"]:
            model_phase1.Add(x_var == 1)

    # 制約: 各講座には必ず1名の講師を割り当てる (ハード制約)
    for course_id, course in courses_dict.items():
        possible_x_vars_for_course = [
            x_vars_phase1[(l_id, course_id)] for l_id in lecturers_dict.keys()
            if (l_id, course_id) in x_vars_phase1
        ]
        if possible_x_vars_for_course:
            model_phase1.Add(sum(possible_x_vars_for_course) == 1)
        else:
            logger.warning(f"講座 {course_id} に割り当て可能な講師がありません。")
            unassigned_courses_list.append(courses_dict[course_id])
            status_phase1 = cp_model.INFEASIBLE # 割り当て候補がなければ即座にINFEASIBLEと判断
            logger.info(f"フェーズ1結果: {solver_phase1.StatusName(status_phase1)} (講座 {course_id} に割り当て候補なし)")
            return {
                "solution_status_str": "NO_ASSIGNMENT_POSSIBLE",
                "objective_value": None,
                "assignments": [],
                "all_courses": courses_data,
                "all_lecturers": lecturers_data,
                "solver_raw_status_code": status_phase1,
                "unassigned_courses": unassigned_courses_list,
                "min_max_assignments_per_lecturer": None # フェーズ1失敗時は設定しない
            }

    status_phase1 = solver_phase1.Solve(model_phase1)
    logger.info(f"フェーズ1結果: {solver_phase1.StatusName(status_phase1)}")

    if status_phase1 not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        final_solution_status_str = solver_phase1.StatusName(status_phase1)
        final_raw_solver_status_code = status_phase1
        
        if status_phase1 == cp_model.FEASIBLE or status_phase1 == cp_model.UNKNOWN:
            assigned_courses_count = 0
            for course_id in courses_dict.keys():
                course_assigned = False
                for lecturer_id in lecturers_dict.keys():
                    if (lecturer_id, course_id) in x_vars_phase1 and solver_phase1.Value(x_vars_phase1[(lecturer_id, course_id)]) == 1:
                        course_assigned = True
                        break
                if course_assigned:
                    assigned_courses_count += 1
                else:
                    unassigned_courses_list.append(courses_dict[course_id])
            
            if assigned_courses_count > 0 and len(unassigned_courses_list) > 0:
                final_solution_status_str = "PARTIALLY_ASSIGNED"
            elif assigned_courses_count == 0 and len(unassigned_courses_list) > 0:
                 final_solution_status_str = "NO_ASSIGNMENT_POSSIBLE"
            else:
                 final_solution_status_str = "UNKNOWN_FEASIBILITY"

        elif status_phase1 == cp_model.INFEASIBLE:
            final_solution_status_str = "INFEASIBLE_PHASE1"

        logger.warning(f"フェーズ1終了: 全ての講座を割り当てることはできませんでした。ステータス: {final_solution_status_str}")
        return {
            "solution_status_str": final_solution_status_str,
            "objective_value": None,
            "assignments": [],
            "all_courses": courses_data,
            "all_lecturers": lecturers_data,
            "solver_raw_status_code": final_raw_solver_status_code,
            "unassigned_courses": unassigned_courses_list,
            "min_max_assignments_per_lecturer": None
        }
    
    logger.info("フェーズ1成功: 全ての講座を割り当て可能です。")

    # --- フェーズ2: 講師割り当て回数の最大値の最小化 ---
    logger.info("フェーズ2開始: 講師割り当て回数の最大値の最小化...")
    model_phase2 = cp_model.CpModel()
    solver_phase2 = cp_model.CpSolver()
    solver_phase2.parameters.log_search_progress = True
    solver_phase2.parameters.max_time_in_seconds = time_limit_seconds

    x_vars_phase2: Dict[Tuple[str, str], cp_model.BoolVar] = {}
    assignments_by_lecturer_phase2: Dict[str, List[cp_model.BoolVar]] = {l_id: [] for l_id in lecturers_dict.keys()}

    for assign_detail in possible_assignments_details:
        lecturer_id = assign_detail["lecturer_id"]
        course_id = assign_detail["course_id"]
        x_var = model_phase2.NewBoolVar(f'x_{lecturer_id}_{course_id}_P2')
        x_vars_phase2[(lecturer_id, course_id)] = x_var
        assignments_by_lecturer_phase2[lecturer_id].append(x_var)
        
        if (lecturer_id, course_id) in fixed_assignments_set:
             model_phase2.Add(x_var == 1)

    # 制約: 各講座には必ず1名の講師を割り当てる (ハード制約)
    for course_id, course in courses_dict.items():
        possible_x_vars_for_course = [
            x_vars_phase2[(l_id, course_id)] for l_id in lecturers_dict.keys()
            if (l_id, course_id) in x_vars_phase2
        ]
        if possible_x_vars_for_course:
            model_phase2.Add(sum(possible_x_vars_for_course) == 1)
        else:
            raise SolverError(f"講座 {course_id} に割り当て可能な講師がありません。フェーズ1で検知されるべき問題です。")

    max_assignments_var = model_phase2.NewIntVar(0, len(courses_data), 'max_assignments')
    for lecturer_id, x_vars in assignments_by_lecturer_phase2.items():
        num_total_assignments_l = model_phase2.NewIntVar(0, len(courses_data), f'num_total_assignments_{lecturer_id}_P2')
        model_phase2.Add(num_total_assignments_l == sum(x_vars))
        model_phase2.Add(num_total_assignments_l <= max_assignments_var)

    model_phase2.Minimize(max_assignments_var)

    status_phase2 = solver_phase2.Solve(model_phase2)
    logger.info(f"フェーズ2結果: {solver_phase2.StatusName(status_phase2)}")

    if status_phase2 not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        logger.error(f"フェーズ2失敗: 講師割り当て回数の最大値を決定できませんでした。ステータス: {solver_phase2.StatusName(status_phase2)}")
        final_solution_status_str = solver_phase2.StatusName(status_phase2)
        final_raw_solver_status_code = status_phase2
        raise SolverError(f"フェーズ2で解が見つかりませんでした: {final_solution_status_str}")

    determined_min_max_assignments = int(solver_phase2.ObjectiveValue())
    logger.info(f"フェーズ2成功: 決定された講師の最小最大割り当て回数 = {determined_min_max_assignments}")

    # --- フェーズ3: 最終的な最適化 ---
    logger.info("フェーズ3開始: ユーザー設定の割り当て上限を制約として最終最適化...")
    model_phase3 = cp_model.CpModel()
    solver_phase3 = cp_model.CpSolver()
    solver_phase3.parameters.log_search_progress = True
    solver_phase3.parameters.max_time_in_seconds = time_limit_seconds

    x_vars_phase3: Dict[Tuple[str, str], cp_model.BoolVar] = {}
    possible_assignments_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
    assignments_by_lecturer_phase3: Dict[str, List[cp_model.BoolVar]] = {l_id: [] for l_id in lecturers_dict.keys()}
    
    consecutive_assignment_pair_vars_details: List[Dict[str, Any]] = []

    for assign_detail in possible_assignments_details:
        lecturer_id = assign_detail["lecturer_id"]
        course_id = assign_detail["course_id"]
        cost = assign_detail["cost"]

        x_var = model_phase3.NewBoolVar(f'x_{lecturer_id}_{course_id}_P3')
        x_vars_phase3[(lecturer_id, course_id)] = x_var
        assignments_by_lecturer_phase3[lecturer_id].append(x_var)
        
        possible_assignments_dict[(lecturer_id, course_id)] = {
            "variable": x_var,
            "cost": cost,
            **assign_detail
        }

        if assign_detail["is_fixed"]:
             model_phase3.Add(x_var == 1)

    # 制約: 各講座には必ず1名の講師を割り当てる (ハード制約)
    for course_id, course in courses_dict.items():
        possible_x_vars_for_course = [
            x_vars_phase3[(l_id, course_id)] for l_id in lecturers_dict.keys()
            if (l_id, course_id) in x_vars_phase3
        ]
        if possible_x_vars_for_course:
            model_phase3.Add(sum(possible_x_vars_for_course) == 1)

    # ユーザー設定の「最大割り当て回数上限」を制約として追加
    # max_assignments_per_lecturer が None でない場合のみ適用
    if max_assignments_per_lecturer is not None:
        for lecturer_id, x_vars in assignments_by_lecturer_phase3.items():
            model_phase3.Add(sum(x_vars) <= max_assignments_per_lecturer)
        logger.info(f"制約追加: 各講師の割り当て回数は最大 {max_assignments_per_lecturer} 回（ユーザー設定）")
    else:
        # max_assignments_per_lecturer が None の場合、集中度に関する上限制約は適用しない
        # これはユーザーが「集中度を低くする」スライダーを0.0に設定した場合などに該当
        logger.warning("max_assignments_per_lecturer が設定されていません。講師の割り当て集中度に関する明示的な上限制約は適用されません。")
        # ただし、フェーズ2で得られた determined_min_max_assignments は、
        # UIで情報提供として利用されるため、ここでの制約には直接影響しない。


    # 連日割り当ての報酬に関する変数定義
    actual_reward_for_consecutive = int(weight_consecutive_assignment * 500)
    if weight_consecutive_assignment > 0 and actual_reward_for_consecutive > 0:
        for pair_detail in consecutive_day_pairs:
            lecturer_id = pair_detail["lecturer_id"]
            course1_id = pair_detail["course1_id"]
            course2_id = pair_detail["course2_id"]

            if (lecturer_id, course1_id) in x_vars_phase3 and \
               (lecturer_id, course2_id) in x_vars_phase3:
                
                pair_var = model_phase3.NewBoolVar(f'y_{lecturer_id}_{course1_id}_{course2_id}_P3')
                individual_var_c1 = x_vars_phase3[(lecturer_id, course1_id)]
                individual_var_c2 = x_vars_phase3[(lecturer_id, course2_id)]
                
                model_phase3.Add(pair_var <= individual_var_c1)
                model_phase3.Add(pair_var <= individual_var_c2)
                
                consecutive_assignment_pair_vars_details.append({
                    "variable": pair_var,
                    "lecturer_id": lecturer_id,
                    "course1_id": course1_id,
                    "course2_id": course2_id,
                    "reward": actual_reward_for_consecutive
                })

    # 目的関数の構築
    objective_terms = []
    for detail in possible_assignments_dict.values():
        objective_terms.append(detail["variable"] * detail["cost"])

    # 講師の割り当て集中度に関するペナルティ項は、ユーザー設定のハード制約に役割を譲るため、ここでは削除
    # if weight_lecturer_concentration > 0 and actual_penalty_concentration > 0:
    #     for lecturer_id_loop, lecturer_vars in assignments_by_lecturer_phase3.items():
    #         if not lecturer_vars or len(lecturer_vars) <= 1:
    #             continue
    #         extra_assign_var_name = f'extra_assign_{lecturer_id_loop}_P3'
    #         if model_phase3.HasVar(extra_assign_var_name):
    #             extra_assignments_l = model_phase3.GetVarFromName(extra_assign_var_name)
    #             objective_terms.append(extra_assignments_l * actual_penalty_concentration)

    if weight_consecutive_assignment > 0 and actual_reward_for_consecutive > 0:
        for pair_detail in consecutive_assignment_pair_vars_details:
            objective_terms.append(pair_detail["variable"] * -pair_detail["reward"])

    if objective_terms:
        model_phase3.Minimize(sum(objective_terms))
    else:
        model_phase3.Minimize(0)

    status_phase3 = solver_phase3.Solve(model_phase3)
    logger.info(f"フェーズ3結果: {solver_phase3.StatusName(status_phase3)}")

    # --- 4. 結果の抽出と整形 ---
    if status_phase3 in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        final_solution_status_str = solver_phase3.StatusName(status_phase3)
        final_objective_value = solver_phase3.ObjectiveValue()

        for detail in possible_assignments_details:
            lecturer_id = detail["lecturer_id"]
            course_id = detail["course_id"]
            if solver_phase3.Value(x_vars_phase3[(lecturer_id, course_id)]) == 1:
                lecturer_assign_count = int(solver_phase3.Value(model_phase3.GetVarFromName(f'num_total_assignments_{lecturer_id}_P3'))) if model_phase3.HasVar(f'num_total_assignments_{lecturer_id}_P3') else 1

                is_consecutive_pair = "なし"
                for pair_var_detail in consecutive_assignment_pair_vars_details:
                    if pair_var_detail["lecturer_id"] == lecturer_id and \
                       (pair_var_detail["course1_id"] == course_id or pair_var_detail["course2_id"] == course_id) and \
                       solver_phase3.Value(pair_var_detail["variable"]) == 1:
                        is_consecutive_pair = f"あり ({pair_var_detail['course1_id']}-{pair_var_detail['course2_id']})"
                        break

                final_assignments.append({
                    "講師ID": lecturer_id,
                    "講座ID": course_id,
                    "算出コスト(x100)": detail["cost"],
                    "年齢コスト(元)": detail["age_cost_raw"],
                    "頻度コスト(元)": detail["frequency_cost_raw"],
                    "資格コスト(元)": detail["qualification_cost_raw"],
                    "当該教室最終割当日からの日数": detail["recency_days"],
                    "今回の割り当て回数": lecturer_assign_count,
                    "連続ペア割当": is_consecutive_pair,
                })
    else:
        final_solution_status_str = solver_phase3.StatusName(status_phase3)
        final_objective_value = None
        logger.error(f"最終最適化フェーズで解が見つかりませんでした。ステータス: {final_solution_status_str}")

    final_raw_solver_status_code = status_phase3

    return {
        "solution_status_str": final_solution_status_str,
        "objective_value": final_objective_value,
        "assignments": final_assignments,
        "all_courses": courses_data,
        "all_lecturers": lecturers_data,
        "solver_raw_status_code": final_raw_solver_status_code,
        "unassigned_courses": unassigned_courses_list,
        "min_max_assignments_per_lecturer": determined_min_max_assignments
    }
