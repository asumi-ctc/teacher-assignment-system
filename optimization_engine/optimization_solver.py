import logging
import datetime
from typing import List, Dict, Any, Tuple, Optional, Callable
import pandas as pd
from ortools.sat.python import cp_model
import sys
import time # timeモジュールをインポートし、sleepを使用
from pathlib import Path # Pathモジュールをインポートし、ファイルクリアに使用

# .utils から型定義とエラー定義をインポート
from .utils.error_definitions import InvalidInputError, ProcessExecutionError, ProcessTimeoutError, SolverError
from .utils.types import LecturerData, CourseData, ClassroomData, SolverOutput, SolverAssignment
# logging_config から SOLVER_LOG_FILE をインポート (StreamToLoggerが使用)
from .utils.logging_config import SOLVER_LOG_FILE 

logger = logging.getLogger('optimization_solver')

# OR-Toolsの標準出力/標準エラー出力をPythonロガーにリダイレクトするためのクラス
class StreamToLogger:
    """
    Standard stream (stdout/stderr) redirector to a Python logger.
    OR-Toolsのログを捕捉し、指定されたプレフィックスを付けてロガーに出力します。
    """
    def __init__(self, logger_obj, log_level=logging.INFO, prefix=""):
        self.logger = logger_obj
        self.log_level = log_level
        self.prefix = prefix
        self.linebuf = '' # 行バッファ

    def write(self, buf):
        # 行ごとに処理するためにバッファリング
        for line in buf.rstrip().splitlines():
            if line.strip(): # 空行は無視
                self.logger.log(self.log_level, f"{self.prefix}{line.strip()}")

    def flush(self):
        # flushは何もしない (ロガーのハンドラがフラッシュを処理するため)
        pass

# 過去の割り当てがない、または日付パース不能な場合に設定するデフォルトの経過日数 (ペナルティ計算上、十分に大きい値)
DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT = 365 * 10 # 10年

def solve_assignment(
    lecturers_data: List[LecturerData],
    courses_data: List[CourseData],
    classrooms_data: List[ClassroomData],
    weight_past_assignment_recency: float,
    weight_qualification: float,
    weight_age: float,
    weight_frequency: float,
    weight_consecutive_assignment: float,
    today_date: datetime.date,
    fixed_assignments: Optional[List[Tuple[str, str]]] = None,
    forced_unassignments: Optional[List[Tuple[str, str]]] = None,
    time_limit_seconds: int = 60, # 各ソルバーフェーズのデフォルトのタイムリミット（UIからは制御されない）
    max_assignments_per_lecturer: Optional[int] = None # UIの集中度スライダーから計算された上限値
) -> SolverOutput:
    """
    講師割り当て問題を3段階のレキシコグラフィカル最適化手法で解決します。

    フェーズ1: 全ての講座に講師を割り当てることが可能かを確認します。
    フェーズ2: 全ての講座が割り当て可能であるという前提のもと、講師一人あたりの最大割り当て回数を最小化し、
               その「最小の最大割り当て回数 (M_min)」を決定します。
    フェーズ3: ユーザーが設定した「講師の最大割り当て回数上限」を制約として適用し、
               その他の目的（年齢、頻度、資格、実績、連日割り当て報酬）の重み付き合計を最小化します。

    Args:
        lecturers_data (List[LecturerData]): 講師データのリスト。
        courses_data (List[CourseData]): 講座データのリスト。
        classrooms_data (List[ClassroomData]): 教室データのリスト。
        weight_past_assignment_recency (float): 過去の割り当て実績コストの重み。
        weight_qualification (float): 資格コストの重み。
        weight_age (float): 年齢コストの重み。
        weight_frequency (float): 割り当て頻度コストの重み。
        weight_consecutive_assignment (float): 連日割り当て報酬の重み。
        today_date (datetime.date): 割り当て日数の計算に使用する現在の日付。
        fixed_assignments (Optional[List[Tuple[str, str]]]): 強制的に割り当てる (講師ID, 講座ID) のペアのリスト。
        forced_unassignments (Optional[List[Tuple[str, str]]]): 強制的に割り当てない (講師ID, 講座ID) のペアのリスト。
        time_limit_seconds (int): 各ソルバーフェーズの最大実行時間（秒）。
        max_assignments_per_lecturer (Optional[int]): ユーザーがUIで設定した講師一人あたりの最大割り当て回数上限。

    Returns:
        SolverOutput: ソルバーの実行結果、解決ステータス、目的値、割り当てリスト、
                      フェーズ2で決定された最小最大割り当て回数などを含む辞書。
    Raises:
        SolverError: ソルバーが期待通りに解を見つけられなかった場合。
        InvalidInputError: 入力データが不正な場合。
        ProcessExecutionError: 予期せぬ実行エラーが発生した場合。
    """
    logger.info("レキシコグラフィカル最適化を開始します。")

    # 最適化開始時にSOLVER_LOG_FILEをクリア
    try:
        if Path(SOLVER_LOG_FILE).exists():
            with open(SOLVER_LOG_FILE, 'w', encoding='utf-8') as f:
                f.truncate(0)
            logger.info(f"Cleared SOLVER_LOG_FILE: {SOLVER_LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to clear SOLVER_LOG_FILE {SOLVER_LOG_FILE}: {e}")

    # --- データの前処理と準備 ---
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

    # --- 割り当て候補の生成とコスト計算 ---
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

    # --- レキシコグラフィカル最適化フェーズ ---
    final_assignments: List[SolverAssignment] = []
    final_objective_value: Optional[float] = None
    final_solution_status_str: str = "UNKNOWN"
    final_raw_solver_status_code: int = cp_model.UNKNOWN
    unassigned_courses_list: List[Dict[str, Any]] = []
    determined_min_max_assignments: Optional[int] = None

    # OR-Toolsソルバーのログ出力をPythonロガーにリダイレクト
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    # StreamToLoggerのprefixをnew_app.pyのフィルタリングと完全に一致させる
    sys.stdout = StreamToLogger(logger, logging.INFO, prefix="[OR-Tools] ") # プレフィックスを"[OR-Tools] "に修正
    sys.stderr = StreamToLogger(logger, logging.ERROR, prefix="[OR-Tools ERROR] ") # エラーログのプレフィックスも修正

    try: # try-finallyブロックで確実にストリームを元に戻す
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
                logger.warning(f"講座 {course_id} に割り当て可能な講師が、スケジュールや資格の制約により見つかりませんでした。")
                unassigned_courses_list.append(courses_dict[course_id])
                status_phase1 = cp_model.INFEASIBLE
                logger.info(f"フェーズ1結果: {solver_phase1.StatusName(status_phase1)} (講座 {course_id} に割り当て候補なし)")
                return {
                    "solution_status_str": "NO_ASSIGNMENT_POSSIBLE",
                    "objective_value": None,
                    "assignments": [],
                    "all_courses": courses_data,
                    "all_lecturers": lecturers_data,
                    "solver_raw_status_code": status_phase1,
                    "unassigned_courses": unassigned_courses_list,
                    "min_max_assignments_per_lecturer": None
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

        # --- フェーズ2: 講師割り当て回数の最大値の最小化 (M_minの決定) ---
        logger.info("フェーズ2開始: 講師割り当て回数の最大値の最小化 (M_min決定)...")
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

        # 制約: 各講座には必ず1名の講師を割り当てる (フェーズ1の成功を引き継ぐハード制約)
        for course_id, course in courses_dict.items():
            possible_x_vars_for_course = [
                x_vars_phase2[(l_id, course_id)] for l_id in lecturers_dict.keys()
                if (l_id, course_id) in x_vars_phase2
            ]
            if possible_x_vars_for_course:
                model_phase2.Add(sum(possible_x_vars_for_course) == 1)
            else:
                logger.error(f"内部エラー: 講座 {course_id} に割り当て可能な講師がフェーズ2モデルで見つかりませんでした。")
                raise SolverError(f"内部エラー: 講座 {course_id} に割り当て可能な講師がフェーズ2モデルで見つかりませんでした。")

        max_assignments_var = model_phase2.NewIntVar(0, len(courses_data), 'max_assignments')
        for lecturer_id, x_vars in assignments_by_lecturer_phase2.items():
            num_total_assignments_l = model_phase2.NewIntVar(0, len(courses_data), f'num_total_assignments_{lecturer_id}_P2')
            model_phase2.Add(num_total_assignments_l == sum(x_vars))
            model_phase2.Add(num_total_assignments_l <= max_assignments_var)

        model_phase2.Minimize(max_assignments_var)

        status_phase2 = solver_phase2.Solve(model_phase2)
        logger.info(f"フェーズ2結果: {solver_phase2.StatusName(status_phase2)}")

        if status_phase2 not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            logger.error(f"フェーズ2失敗: 講師割り当て回数の最小最大値を決定できませんでした。ステータス: {solver_phase2.StatusName(status_phase2)}")
            final_solution_status_str = solver_phase2.StatusName(status_phase2)
            final_raw_solver_status_code = status_phase2
            raise SolverError(f"フェーズ2で解が見つかりませんでした: {final_solution_status_str}")

        determined_min_max_assignments = int(solver_phase2.ObjectiveValue())
        logger.info(f"フェーズ2成功: 決定された講師の最小最大割り当て回数 (M_min) = {determined_min_max_assignments}")

        # --- フェーズ3: 最終的な最適化（ユーザー設定の集中度制約とコスト最小化） ---
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

        # ユーザーがUIで設定した max_assignments_per_lecturer をハード制約として適用
        if max_assignments_per_lecturer is not None:
            for lecturer_id, x_vars in assignments_by_lecturer_phase3.items():
                num_total_assignments_l_p3 = model_phase3.NewIntVar(0, len(courses_data), f'num_total_assignments_{lecturer_id}_P3')
                model_phase3.Add(num_total_assignments_l_p3 == sum(x_vars))
                model_phase3.Add(num_total_assignments_l_p3 <= max_assignments_per_lecturer)
            logger.info(f"制約追加: 各講師の割り当て回数は最大 {max_assignments_per_lecturer} 回（ユーザー設定）")
        else:
            logger.info("講師の最大割り当て回数上限は設定されていません（ユーザーが集中度を最も許容する設定）。")


        # 連日割り当ての報酬に関する変数定義と制約
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

        if weight_consecutive_assignment > 0 and actual_reward_for_consecutive > 0:
            for pair_detail in consecutive_assignment_pair_vars_details:
                objective_terms.append(pair_detail["variable"] * -pair_detail["reward"])

        if objective_terms:
            model_phase3.Minimize(sum(objective_terms))
        else:
            model_phase3.Minimize(0)

        status_phase3 = solver_phase3.Solve(model_phase3)
        logger.info(f"フェーズ3結果: {solver_phase3.StatusName(status_phase3)}")

        # --- 結果の抽出と整形 ---
        if status_phase3 in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            final_solution_status_str = solver_phase3.StatusName(status_phase3)
            final_objective_value = solver_phase3.ObjectiveValue()

            for detail in possible_assignments_details:
                lecturer_id = detail["lecturer_id"]
                course_id = detail["course_id"]
                if solver_phase3.Value(x_vars_phase3[(lecturer_id, course_id)]) == 1:
                    lecturer_assign_count = 1 

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
    finally: # 確実にストリームを元に戻し、ロガーをフラッシュ
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        for handler in logger.handlers:
            handler.flush()
        time.sleep(0.1) # ファイルシステムへの書き込みが完了するのを少し待つ
