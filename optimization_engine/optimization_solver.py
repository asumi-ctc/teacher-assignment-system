import logging
import datetime
from typing import List, Dict, Any, Tuple, Optional, Callable
import pandas as pd # Used for internal data handling if needed, though primarily for output formatting
from ortools.sat.python import cp_model

# Import custom error definitions and types from .utils
from .utils.error_definitions import InvalidInputError, ProcessExecutionError, ProcessTimeoutError, SolverError
from .utils.types import LecturerData, CourseData, ClassroomData, SolverOutput, SolverAssignment

logger = logging.getLogger('optimization_solver')

# Default value for days since last assignment if no past assignment or invalid date
DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT = 365 * 10 # Represents a very old assignment (10 years)

def solve_assignment(
    lecturers_data: List[LecturerData],
    courses_data: List[CourseData],
    classrooms_data: List[ClassroomData],
    weight_past_assignment_recency: float,
    weight_qualification: float,
    weight_age: float,
    weight_frequency: float,
    # weight_lecturer_concentration is handled by max_assignments_per_lecturer calculation in app.py
    weight_consecutive_assignment: float,
    today_date: datetime.date,
    fixed_assignments: Optional[List[Tuple[str, str]]] = None,
    forced_unassignments: Optional[List[Tuple[str, str]]] = None,
    time_limit_seconds: int = 60, # Default time limit for each solver phase (not controlled by UI slider)
    max_assignments_per_lecturer: Optional[int] = None # Calculated from UI's concentration slider (can be None)
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

    # --- データの前処理と準備 ---
    # 辞書形式に変換して高速なルックアップを可能にする
    lecturers_dict = {l['id']: l for l in lecturers_data}
    courses_dict = {c['id']: c for c in courses_data}
    classrooms_dict = {c['id']: c for c in classrooms_data}

    # 日付文字列をdatetime.dateオブジェクトに変換するヘルパー関数
    def parse_date_if_str(date_obj: Any) -> Optional[datetime.date]:
        if isinstance(date_obj, str):
            try:
                return datetime.date.fromisoformat(date_obj)
            except ValueError:
                logger.warning(f"Invalid date format encountered: {date_obj}. Returning None.")
                return None
        return date_obj

    # 講師データのavailabilityとpast_assignmentsの日付を変換
    for lecturer_id, lecturer in lecturers_dict.items():
        if 'availability' in lecturer and isinstance(lecturer['availability'], list):
            lecturer['availability'] = [parse_date_if_str(d) for d in lecturer['availability']]
            lecturer['availability'] = [d for d in lecturer['availability'] if d is not None] # 無効な日付をフィルタリング
        if 'past_assignments' in lecturer and isinstance(lecturer['past_assignments'], list):
            for pa in lecturer['past_assignments']:
                if 'date' in pa:
                    pa['date'] = parse_date_if_str(pa['date'])
    
    # 講座データのスケジュール日付を変換
    for course_id, course in courses_dict.items():
        if 'schedule' in course:
            course['schedule'] = parse_date_if_str(course['schedule'])

    # 固定割り当てと強制非割り当てのペアをセットとして準備（高速な検索のため）
    fixed_assignments_set = set(fixed_assignments) if fixed_assignments else set()
    forced_unassignments_set = set(forced_unassignments) if forced_unassignments else set()

    # --- 割り当て候補の生成とコスト計算 ---
    # ソルバー変数を作成する前に、可能な講師-講座の組み合わせをフィルタリングし、コストを計算する
    possible_assignments_details = []
    
    # 連日割り当てのペアを事前に特定（講師が特別資格を持ち、両講座を担当可能な場合のみ）
    consecutive_day_pairs = []
    # 全ての講座を日付と教室IDでグループ化して、翌日の講座を効率的に検索できるようにする
    courses_by_date_classroom = {}
    for course_id, course in courses_dict.items():
        if course['schedule'] and course['classroom_id']:
            key = (course['schedule'], course['classroom_id'])
            if key not in courses_by_date_classroom:
                courses_by_date_classroom[key] = []
            courses_by_date_classroom[key].append(course_id)

    # 各一般講座について、翌日に同じ教室で特別講座があるかを確認
    for course_id_1, course_1 in courses_dict.items():
        if course_1['schedule'] and course_1['course_type'] == 'general':
            next_day_date = course_1['schedule'] + datetime.timedelta(days=1)
            # 同じ教室で翌日に特別講座があるか
            if (next_day_date, course_1['classroom_id']) in courses_by_date_classroom:
                for course_id_2 in courses_by_date_classroom[(next_day_date, course_1['classroom_id'])]:
                    course_2 = courses_dict[course_id_2]
                    if course_2['course_type'] == 'special':
                        # この連日ペアを担当可能な講師を特定
                        for lecturer_id, lecturer in lecturers_dict.items():
                            # 講師が特別資格を持ち、かつ両方の講座のスケジュールと資格ランクを満たすか
                            if lecturer['qualification_special_rank'] is not None and \
                               course_1['schedule'] in lecturer['availability'] and \
                               course_2['schedule'] in lecturer['availability'] and \
                               lecturer['qualification_special_rank'] <= course_1['rank'] and \
                               lecturer['qualification_special_rank'] <= course_2['rank']:
                                consecutive_day_pairs.append({
                                    "lecturer_id": lecturer_id, # この講師がこのペアを担当可能
                                    "course1_id": course_id_1,
                                    "course2_id": course_id_2,
                                    "classroom_id": course_1['classroom_id'],
                                    "dates": (course_1['schedule'], course_2['schedule'])
                                })
    
    # コスト計算の正規化に使用する最大値（ハードコードではなく、データから動的に計算することも可能）
    MAX_AGE_COST = 65 - 22 # 講師の年齢範囲 (22歳から65歳を想定)
    MAX_FREQUENCY_COST = 12 # 過去の割り当て履歴の最大数 (ダミーデータ生成ロジックに基づく)
    MAX_QUALIFICATION_COST = 5 # 資格ランクの範囲 (1から5)
    MAX_RECENCY_DAYS = 365 * 2 # 過去2年間の日数 (ダミーデータ生成ロジックに基づく)

    # 各講師-講座の割り当て候補について、適格性チェックとコスト計算を行う
    for lecturer_id, lecturer in lecturers_dict.items():
        for course_id, course in courses_dict.items():
            # ハード制約: 講師のスケジュールが講座のスケジュールに適合しているか
            if course['schedule'] not in lecturer['availability']:
                continue # この割り当ては不可能なので、候補から除外

            # ハード制約: 講師の資格ランクが講座の要求ランクを満たしているか
            is_qualified = False
            if course['course_type'] == 'general':
                # 一般講座: 講師の一般資格ランクが講座ランク以下、または特別資格を持つ場合
                if lecturer['qualification_general_rank'] <= course['rank']:
                    is_qualified = True
                elif lecturer['qualification_special_rank'] is not None: # 特別資格があれば一般講座は担当可能
                    is_qualified = True
            elif course['course_type'] == 'special':
                # 特別講座: 講師が特別資格を持ち、その特別資格ランクが講座ランク以下の場合
                if lecturer['qualification_special_rank'] is not None and lecturer['qualification_special_rank'] <= course['rank']:
                    is_qualified = True
            
            if not is_qualified:
                continue # この割り当ては不可能なので、候補から除外

            # 強制非割り当ての適用
            assignment_pair = (lecturer_id, course_id)
            if assignment_pair in forced_unassignments_set:
                continue # この割り当ては不可能なので、候補から除外
            
            # --- コスト計算 ---
            # 各コストは0-1の範囲に正規化され、重み付けされる
            
            # 年齢コスト: 若いほど低コスト（年齢が高いほどペナルティ）
            # 最小年齢を22歳と仮定し、年齢が高いほどコストが増加
            age_cost = (lecturer['age'] - 22) / MAX_AGE_COST if MAX_AGE_COST > 0 else 0
            
            # 頻度コスト: 過去の割り当てが少ないほど低コスト（多いほどペナルティ）
            # 過去の総割り当て回数が多いほどコストが増加
            frequency_cost = len(lecturer.get('past_assignments', [])) / MAX_FREQUENCY_COST if MAX_FREQUENCY_COST > 0 else 0

            # 資格コスト: ランクが高いほど低コスト（ランクが低いほどペナルティ）
            # 講師の資格ランクと講座の要求ランクの差に基づいてコストを計算
            qualification_cost = 0
            if course['course_type'] == 'general':
                qualification_cost = (lecturer['qualification_general_rank'] - course['rank']) / MAX_QUALIFICATION_COST if MAX_QUALIFICATION_COST > 0 else 0
            elif course['course_type'] == 'special' and lecturer['qualification_special_rank'] is not None:
                qualification_cost = (lecturer['qualification_special_rank'] - course['rank']) / MAX_QUALIFICATION_COST if MAX_QUALIFICATION_COST > 0 else 0
            qualification_cost = max(0, qualification_cost) # コストは0未満にならないように

            # 過去の割り当て実績コスト: 同じ教室への割り当てが最近でないほど低コスト（最近ほどペナルティ）
            # または過去の割り当てがない場合はペナルティ
            recency_cost = 0
            last_assignment_date_in_classroom = None
            for pa in lecturer.get('past_assignments', []):
                if pa['classroom_id'] == course['classroom_id'] and pa['date'] is not None:
                    if last_assignment_date_in_classroom is None or pa['date'] > last_assignment_date_in_classroom:
                        last_assignment_date_in_classroom = pa['date']
            
            days_since_last_assignment = DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT # デフォルトは十分に大きい値
            if last_assignment_date_in_classroom:
                days_since_last_assignment = (today_date - last_assignment_date_in_classroom).days
            
            # 日数が少ないほどペナルティ（コスト大）、多いほどコスト小
            # 正規化: (MAX_RECENCY_DAYS - 経過日数) / MAX_RECENCY_DAYS
            recency_cost = max(0, (MAX_RECENCY_DAYS - days_since_last_assignment) / MAX_RECENCY_DAYS) if MAX_RECENCY_DAYS > 0 else 0

            # 全てのコスト要素を統合し、重みを適用後、整数にスケール（CP-SATの目的関数は整数が望ましい）
            # UIの重みは0.0-1.0なので、そのまま乗算
            total_scaled_cost = (
                age_cost * weight_age +
                frequency_cost * weight_frequency +
                qualification_cost * weight_qualification +
                recency_cost * weight_past_assignment_recency
            ) * 1000 # 全体のスケールファクター (最大1000点)

            cost = int(round(total_scaled_cost))

            possible_assignments_details.append({
                "lecturer_id": lecturer_id,
                "course_id": course_id,
                "cost": cost, # スケール後の整数コスト
                "age_cost_raw": age_cost, # デバッグ用の生データ
                "frequency_cost_raw": frequency_cost, # デバッグ用の生データ
                "qualification_cost_raw": qualification_cost, # デバッグ用の生データ
                "recency_days": days_since_last_assignment, # デバッグ用の生データ
                "is_fixed": assignment_pair in fixed_assignments_set # 固定割り当てかどうか
            })

    # --- レキシコグラフィカル最適化フェーズ ---
    # 最終的な結果を格納する変数
    final_assignments: List[SolverAssignment] = []
    final_objective_value: Optional[float] = None
    final_solution_status_str: str = "UNKNOWN"
    final_raw_solver_status_code: int = cp_model.UNKNOWN
    unassigned_courses_list: List[Dict[str, Any]] = [] # フェーズ1で割り当てられなかった講座
    determined_min_max_assignments: Optional[int] = None # フェーズ2の出力 (M_min)

    # --- フェーズ1: 全講座割り当て可能性の確認 ---
    logger.info("フェーズ1開始: 全講座割り当て可能性の確認...")
    model_phase1 = cp_model.CpModel()
    solver_phase1 = cp_model.CpSolver()
    solver_phase1.parameters.log_search_progress = True
    solver_phase1.parameters.max_time_in_seconds = time_limit_seconds
    # OR-Toolsのログをキャプチャするためのコールバックを設定
    solver_phase1.log_callback = lambda message: logger.info(f"[OR-Tools] {message}")

    x_vars_phase1: Dict[Tuple[str, str], cp_model.BoolVar] = {}
    
    # 各割り当て候補に対してブール変数を作成
    for assign_detail in possible_assignments_details:
        lecturer_id = assign_detail["lecturer_id"]
        course_id = assign_detail["course_id"]
        
        x_var = model_phase1.NewBoolVar(f'x_{lecturer_id}_{course_id}_P1')
        x_vars_phase1[(lecturer_id, course_id)] = x_var
        
        # 固定割り当ては強制的に1に設定
        if assign_detail["is_fixed"]:
            model_phase1.Add(x_var == 1)

    # 制約: 各講座には必ず1名の講師を割り当てる (ハード制約)
    # これがフェーズ1の主要な目的
    for course_id, course in courses_dict.items():
        possible_x_vars_for_course = [
            x_vars_phase1[(l_id, course_id)] for l_id in lecturers_dict.keys()
            if (l_id, course_id) in x_vars_phase1 # 可能な割り当て候補のみを考慮
        ]
        if possible_x_vars_for_course:
            # 講座に割り当て可能な講師が1人以上いる場合、その中から1人だけ割り当てる
            model_phase1.Add(sum(possible_x_vars_for_course) == 1)
        else:
            # 割り当て候補が全くない講座は、このフェーズ1で割り当て不能と判断し、即座に結果を返す
            logger.warning(f"講座 {course_id} に割り当て可能な講師が、スケジュールや資格の制約により見つかりませんでした。")
            unassigned_courses_list.append(courses_dict[course_id])
            status_phase1 = cp_model.INFEASIBLE # 即座に実行不可能と判断
            logger.info(f"フェーズ1結果: {solver_phase1.StatusName(status_phase1)} (講座 {course_id} に割り当て候補なし)")
            return {
                "solution_status_str": "NO_ASSIGNMENT_POSSIBLE", # 全く割り当てられない
                "objective_value": None,
                "assignments": [],
                "all_courses": courses_data,
                "all_lecturers": lecturers_data,
                "solver_raw_status_code": status_phase1,
                "unassigned_courses": unassigned_courses_list,
                "min_max_assignments_per_lecturer": None # フェーズ1失敗時はM_minは計算されない
            }

    # フェーズ1のモデルを求解
    status_phase1 = solver_phase1.Solve(model_phase1)
    logger.info(f"フェーズ1結果: {solver_phase1.StatusName(status_phase1)}")

    if status_phase1 not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        # フェーズ1が最適解または実行可能解を見つけられなかった場合（実行不可能、タイムアウトなど）
        final_solution_status_str = solver_phase1.StatusName(status_phase1)
        final_raw_solver_status_code = status_phase1
        
        # 実行可能だが最適ではない、または不明なステータスの場合、部分的に割り当てられた講座があるか確認
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
                final_solution_status_str = "PARTIALLY_ASSIGNED" # 一部割り当てられた
            elif assigned_courses_count == 0 and len(unassigned_courses_list) > 0:
                 final_solution_status_str = "NO_ASSIGNMENT_POSSIBLE" # 全く割り当てられなかった
            else: # 例: タイムアウトで結果が不明だが、割り当てられない講座が特定できない場合
                 final_solution_status_str = "UNKNOWN_FEASIBILITY"

        elif status_phase1 == cp_model.INFEASIBLE:
            # 制約が矛盾しているため解が見つからない
            final_solution_status_str = "INFEASIBLE_PHASE1" # フェーズ1での実行不可能を明示

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
    # OR-Toolsのログをキャプチャするためのコールバックを設定
    solver_phase2.log_callback = lambda message: logger.info(f"[OR-Tools] {message}")

    x_vars_phase2: Dict[Tuple[str, str], cp_model.BoolVar] = {}
    assignments_by_lecturer_phase2: Dict[str, List[cp_model.BoolVar]] = {l_id: [] for l_id in lecturers_dict.keys()}

    # フェーズ2のモデルでも割り当て候補変数を作成
    for assign_detail in possible_assignments_details:
        lecturer_id = assign_detail["lecturer_id"]
        course_id = assign_detail["course_id"]
        x_var = model_phase2.NewBoolVar(f'x_{lecturer_id}_{course_id}_P2')
        x_vars_phase2[(lecturer_id, course_id)] = x_var
        assignments_by_lecturer_phase2[lecturer_id].append(x_var)
        
        # 固定割り当てはここでも強制
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
            # このケースはフェーズ1で既に処理されているはず。もし発生すれば内部エラー。
            logger.error(f"内部エラー: 講座 {course_id} に割り当て可能な講師がフェーズ2モデルで見つかりませんでした。")
            raise SolverError(f"内部エラー: 講座 {course_id} に割り当て可能な講師がフェーズ2モデルで見つかりませんでした。")

    # 講師ごとの割り当て回数を計算し、その最大値を最小化する
    max_assignments_var = model_phase2.NewIntVar(0, len(courses_data), 'max_assignments') # 最大値は総講座数まで
    for lecturer_id, x_vars in assignments_by_lecturer_phase2.items():
        num_total_assignments_l = model_phase2.NewIntVar(0, len(courses_data), f'num_total_assignments_{lecturer_id}_P2')
        model_phase2.Add(num_total_assignments_l == sum(x_vars)) # 各講師の総割り当て回数を定義
        model_phase2.Add(num_total_assignments_l <= max_assignments_var) # 各講師の割り当て回数は最大値以下

    model_phase2.Minimize(max_assignments_var) # 全体での最大割り当て回数を最小化

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
    # OR-Toolsのログをキャプチャするためのコールバックを設定
    solver_phase3.log_callback = lambda message: logger.info(f"[OR-Tools] {message}")

    x_vars_phase3: Dict[Tuple[str, str], cp_model.BoolVar] = {}
    possible_assignments_dict: Dict[Tuple[str, str], Dict[str, Any]] = {} # コスト情報も持つ
    assignments_by_lecturer_phase3: Dict[str, List[cp_model.BoolVar]] = {l_id: [] for l_id in lecturers_dict.keys()}
    
    consecutive_assignment_pair_vars_details: List[Dict[str, Any]] = []

    # フェーズ3のモデルでも割り当て候補変数を作成
    for assign_detail in possible_assignments_details:
        lecturer_id = assign_detail["lecturer_id"]
        course_id = assign_detail["course_id"]
        cost = assign_detail["cost"] # スケールされた整数コスト

        x_var = model_phase3.NewBoolVar(f'x_{lecturer_id}_{course_id}_P3')
        x_vars_phase3[(lecturer_id, course_id)] = x_var
        assignments_by_lecturer_phase3[lecturer_id].append(x_var)
        
        possible_assignments_dict[(lecturer_id, course_id)] = {
            "variable": x_var,
            "cost": cost,
            **assign_detail # 元の詳細情報も保持
        }

        # 固定割り当てはここでも強制
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
        # Else: このケースはフェーズ1で既に処理されているはず。

    # ユーザーがUIで設定した max_assignments_per_lecturer をハード制約として適用
    # max_assignments_per_lecturer が None の場合（UIスライダーが0.0に設定された場合など）は、この制約を適用しない
    if max_assignments_per_lecturer is not None:
        for lecturer_id, x_vars in assignments_by_lecturer_phase3.items():
            # 各講師の総割り当て回数を定義
            num_total_assignments_l_p3 = model_phase3.NewIntVar(0, len(courses_data), f'num_total_assignments_{lecturer_id}_P3')
            model_phase3.Add(num_total_assignments_l_p3 == sum(x_vars))
            # ユーザー設定の上限を制約として追加
            model_phase3.Add(num_total_assignments_l_p3 <= max_assignments_per_lecturer)
        logger.info(f"制約追加: 各講師の割り当て回数は最大 {max_assignments_per_lecturer} 回（ユーザー設定）")
    else:
        logger.info("講師の最大割り当て回数上限は設定されていません（ユーザーが集中度を最も許容する設定）。")


    # 連日割り当ての報酬に関する変数定義と制約
    # 報酬は負のコストとして目的関数に加算される
    actual_reward_for_consecutive = int(weight_consecutive_assignment * 500) # 報酬を最大500ポイントにスケール
    if weight_consecutive_assignment > 0 and actual_reward_for_consecutive > 0:
        for pair_detail in consecutive_day_pairs:
            lecturer_id = pair_detail["lecturer_id"]
            course1_id = pair_detail["course1_id"]
            course2_id = pair_detail["course2_id"]

            # この講師が両方の講座に割り当て可能な場合のみ連日ペア変数を考慮
            if (lecturer_id, course1_id) in x_vars_phase3 and \
               (lecturer_id, course2_id) in x_vars_phase3:
                
                pair_var = model_phase3.NewBoolVar(f'y_{lecturer_id}_{course1_id}_{course2_id}_P3')
                individual_var_c1 = x_vars_phase3[(lecturer_id, course1_id)]
                individual_var_c2 = x_vars_phase3[(lecturer_id, course2_id)]
                
                # pair_var が True なら、両方の講座に割り当てられていることを強制
                model_phase3.Add(pair_var <= individual_var_c1)
                model_phase3.Add(pair_var <= individual_var_c2)
                
                consecutive_assignment_pair_vars_details.append({
                    "variable": pair_var,
                    "lecturer_id": lecturer_id,
                    "course1_id": course1_id,
                    "course2_id": course2_id,
                    "reward": actual_reward_for_consecutive # 報酬値
                })

    # 目的関数の構築: 各割り当て候補のコスト合計を最小化
    objective_terms = []
    for detail in possible_assignments_dict.values():
        objective_terms.append(detail["variable"] * detail["cost"])

    # 連日割り当ての報酬項を負のコストとして追加
    if weight_consecutive_assignment > 0 and actual_reward_for_consecutive > 0:
        for pair_detail in consecutive_assignment_pair_vars_details:
            objective_terms.append(pair_detail["variable"] * -pair_detail["reward"]) # 報酬は負のコスト

    if objective_terms:
        model_phase3.Minimize(sum(objective_terms))
    else:
        model_phase3.Minimize(0) # 目的項がない場合（通常は発生しない）

    # フェーズ3のモデルを求解
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
                # 最終的な割り当て回数はGatewayで計算されるため、ここでは仮の値
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
                    "今回の割り当て回数": lecturer_assign_count, # Gatewayで正確な値に置き換えられる
                    "連続ペア割当": is_consecutive_pair,
                })
    else:
        final_solution_status_str = solver_phase3.StatusName(status_phase3)
        final_objective_value = None
        logger.error(f"最終最適化フェーズで解が見つかりませんでした。ステータス: {final_solution_status_str}")

    final_raw_solver_status_code = status_phase3

    # SolverOutputを構築して返す
    return {
        "solution_status_str": final_solution_status_str,
        "objective_value": final_objective_value,
        "assignments": final_assignments,
        "all_courses": courses_data, # 元の全講座データ
        "all_lecturers": lecturers_data, # 元の全講師データ
        "solver_raw_status_code": final_raw_solver_status_code,
        "unassigned_courses": unassigned_courses_list, # フェーズ1で割り当てられなかった講座があればここに
        "min_max_assignments_per_lecturer": determined_min_max_assignments # フェーズ2の出力 (M_min)
    }
