# optimization_engine.py
import io
import logging
import datetime # 日付処理用
import numpy as np
from ortools.sat.python import cp_model
from typing import TypedDict, List, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# --- グローバル定数 (ログマーカー) ---
SOLVER_LOG_START_MARKER = "--- Solver Log (Captured by app.py) ---" # app.py から移動したが、元々の名前を維持
SOLVER_LOG_END_MARKER = "--- End Solver Log (Captured by app.py) ---"   # app.py から移動したが、元々の名前を維持

# --- [ここから app.py より移動] ---
class SolverOutput(TypedDict):
    solution_status_str: str
    objective_value: Optional[float]
    assignments: List[dict]
    all_courses: List[dict]
    all_lecturers: List[dict]
    solver_raw_status_code: int
    full_application_and_solver_log: str
    pure_solver_log: str
    application_log: str

def solve_assignment(lecturers_data: List[Dict[str, Any]],
                     courses_data: List[Dict[str, Any]],
                     classrooms_data: List[Dict[str, Any]], # classrooms_data は現在未使用だが、将来のために残す
                     travel_costs_matrix: Dict[Tuple[str, str], int],
                     weight_past_assignment_recency: float,
                     weight_qualification: float,
                     weight_travel: float,
                     weight_age: float,
                     weight_frequency: float,
                     weight_assignment_shortage: float,
                     weight_lecturer_concentration: float,
                     weight_consecutive_assignment: float,
                     allow_under_assignment: bool,
                     today_date: datetime.date,
                     fixed_assignments: Optional[List[Tuple[str, str]]] = None,
                     forced_unassignments: Optional[List[Tuple[str, str]]] = None) -> SolverOutput:
    model = cp_model.CpModel()

    # --- 1. データ前処理: リストをIDをキーとする辞書に変換 ---
    lecturers_dict = {lecturer['id']: lecturer for lecturer in lecturers_data}
    courses_dict = {course['id']: course for course in courses_data}
    classrooms_dict = {classroom['id']: classroom for classroom in classrooms_data} # 教室データも辞書に変換

    # --- ログキャプチャ用の StringIO ---
    app_log_stream = io.StringIO() # アプリケーションログ用 (エンジン内の処理ログ)
    solver_capture_stream = io.StringIO() # ソルバーログキャプチャ用

    # アプリケーションログ出力関数 (標準ロガーとapp_log_streamに出力)
    def log_to_stream(message: str):
        log_line = f"[SolverEngineLog] {message}\n" # プレフィックス変更
        logger.info(log_line.strip()) # 標準ロガーには改行なしで
        app_log_stream.write(log_line) # app_log_stream には改行ありでキャプチャ

    # --- ステップ1: 連日講座ペアのリストアップ ---
    consecutive_day_pairs: List[Dict[str, Any]] = []
    log_to_stream("Starting search for consecutive general-special course pairs.")
    parsed_courses_for_pairing: List[Dict[str, Any]] = []
    for course_id_loop, course_item_loop in courses_dict.items():
        try:
            schedule_date = datetime.datetime.strptime(course_item_loop['schedule'], "%Y-%m-%d").date()
            parsed_courses_for_pairing.append({**course_item_loop, 'schedule_date_obj': schedule_date})
        except ValueError:
            log_to_stream(f"  Warning: Could not parse schedule date {course_item_loop['schedule']} for course {course_item_loop['id']} during pair finding. Skipping.")
            continue

    courses_by_classroom_for_pairing: Dict[str, List[Dict[str, Any]]] = {}
    for course_in_list in parsed_courses_for_pairing:
        cid = course_in_list['classroom_id']
        if cid not in courses_by_classroom_for_pairing:
            courses_by_classroom_for_pairing[cid] = []
        courses_by_classroom_for_pairing[cid].append(course_in_list)

    for cid_loop, classroom_courses_list in courses_by_classroom_for_pairing.items():
        classroom_courses_list.sort(key=lambda c: c['schedule_date_obj'])
        for i in range(len(classroom_courses_list) - 1):
            course1 = classroom_courses_list[i]
            course2 = classroom_courses_list[i+1]
            
            is_general_special_pair = (course1['course_type'] == 'general' and course2['course_type'] == 'special') or \
                                      (course1['course_type'] == 'special' and course2['course_type'] == 'general')
            
            if is_general_special_pair and (course2['schedule_date_obj'] - course1['schedule_date_obj']).days == 1:
                pair_c1_obj, pair_c2_obj = course1, course2 # 日付順
                consecutive_day_pairs.append({
                    "pair_id": f"CDP_{pair_c1_obj['id']}_{pair_c2_obj['id']}",
                    "course1_id": pair_c1_obj['id'], "course2_id": pair_c2_obj['id'],
                    "classroom_id": cid_loop
                })
                log_to_stream(f"  + Found consecutive day pair: {pair_c1_obj['id']} ({pair_c1_obj['schedule']}) and {pair_c2_obj['id']} ({pair_c2_obj['schedule']}) at {cid_loop}")


    # --- Main logic for model building and solving ---
    possible_assignments_temp_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
    potential_assignment_count = 0
    log_to_stream(f"Initial lecturers: {len(lecturers_data)}, Initial courses: {len(courses_data)}")

    forced_unassignments_set = set(forced_unassignments) if forced_unassignments else set()
    if forced_unassignments_set:
        log_to_stream(f"  Forced unassignments specified: {forced_unassignments_set}")

    RECENCY_COST_CONSTANT = 100000.0

    for lecturer_id_loop, lecturer in lecturers_dict.items():
        for course_id_loop, course in courses_dict.items():
            lecturer_id = lecturer["id"]
            course_id = course["id"]

            if (lecturer_id, course_id) in forced_unassignments_set:
                log_to_stream(f"  - Filtered out (forced unassignment): {lecturer_id} for {course_id}")
                continue

            course_type = course["course_type"]
            course_rank = course["rank"]
            lecturer_general_rank = lecturer["qualification_general_rank"]
            lecturer_special_rank = lecturer.get("qualification_special_rank")

            can_assign_by_qualification = False
            qualification_cost_for_this_assignment = 0

            if course_type == "general":
                if lecturer_special_rank is not None:
                    can_assign_by_qualification = True
                    qualification_cost_for_this_assignment = lecturer_general_rank
                elif lecturer_general_rank <= course_rank:
                    can_assign_by_qualification = True
                    qualification_cost_for_this_assignment = lecturer_general_rank
            elif course_type == "special":
                if lecturer_special_rank is not None and lecturer_special_rank <= course_rank:
                    can_assign_by_qualification = True
                    qualification_cost_for_this_assignment = lecturer_special_rank
            
            if not can_assign_by_qualification:
                log_to_stream(f"  - Filtered out: {lecturer_id} for {course_id} (Qualification insufficient. Course: {course_type} Rank {course_rank}. Lecturer: GenRank {lecturer_general_rank}, SpecRank {lecturer_special_rank})")
                continue

            schedule_available = course["schedule"] in lecturer["availability"]
            if not schedule_available:
                log_to_stream(f"  - Filtered out: {lecturer_id} for {course_id} (Schedule unavailable: Course_schedule={course['schedule']}, Lecturer_avail_sample={lecturer['availability'][:3]}...)")
                continue

            potential_assignment_count += 1
            log_to_stream(f"  + Potential assignment: {lecturer_id} to {course_id} on {course['schedule']}")
            var = model.NewBoolVar(f'x_{lecturer_id}_{course_id}')
            
            travel_cost = travel_costs_matrix.get((lecturer["home_classroom_id"], course["classroom_id"]), 999999) # 大きなデフォルトコスト
            age_cost = float(lecturer.get("age", 99))
            frequency_cost = float(len(lecturer.get("past_assignments", [])))
            qualification_cost_val = float(qualification_cost_for_this_assignment)

            past_assignment_recency_cost = 0.0
            days_since_last_assignment_to_classroom = -1

            if lecturer.get("past_assignments"):
                relevant_past_assignments_to_this_classroom = [
                    pa for pa in lecturer["past_assignments"]
                    if pa["classroom_id"] == course["classroom_id"]
                ]
                if relevant_past_assignments_to_this_classroom:
                    latest_assignment_date_str = relevant_past_assignments_to_this_classroom[0]["date"]
                    try:
                        latest_assignment_date = datetime.datetime.strptime(latest_assignment_date_str, "%Y-%m-%d").date()
                        days_since_last_assignment_to_classroom = (today_date - latest_assignment_date).days
                        if days_since_last_assignment_to_classroom >= 0:
                            past_assignment_recency_cost = RECENCY_COST_CONSTANT / (days_since_last_assignment_to_classroom + 1.0)
                        else:
                            past_assignment_recency_cost = 0.0
                            days_since_last_assignment_to_classroom = -2
                    except ValueError:
                        log_to_stream(f"    Warning: Could not parse date '{latest_assignment_date_str}' for {lecturer_id} and classroom {course['classroom_id']}")
                        past_assignment_recency_cost = 0.0
                        days_since_last_assignment_to_classroom = -3
            else:
                past_assignment_recency_cost = 0.0
                days_since_last_assignment_to_classroom = -1

            log_to_stream(f"    Raw costs for {lecturer_id} to {course_id}: travel={travel_cost}, age={age_cost}, freq={frequency_cost}, qual={qualification_cost_val}, recency={past_assignment_recency_cost:.2f} (days_since_last={days_since_last_assignment_to_classroom})")

            assignment_key = (lecturer_id, course_id)
            possible_assignments_temp_data[assignment_key] = {
                "lecturer_id": lecturer_id, "course_id": course_id,
                "variable": var,
                "raw_costs": {
                    "travel": travel_cost, "age": age_cost, "frequency": frequency_cost,
                    "qualification": qualification_cost_val, "recency": past_assignment_recency_cost
                },
                "debug_days_since_last_assignment": days_since_last_assignment_to_classroom
            }

    log_to_stream(f"Total potential assignments after filtering: {potential_assignment_count}")
    log_to_stream(f"Number of entries in possible_assignments_temp_data: {len(possible_assignments_temp_data)}")

    if not possible_assignments_temp_data:
        log_to_stream("No possible assignments found after filtering. Optimization will likely result in no assignments.")
        captured_app_log_early = app_log_stream.getvalue()
        return SolverOutput(
            solution_status_str="前提条件エラー (割り当て候補なし)",
            objective_value=None,
            assignments=[],
            all_courses=list(courses_dict.values()),
            all_lecturers=list(lecturers_dict.values()),
            solver_raw_status_code=cp_model.UNKNOWN, 
            full_application_and_solver_log=captured_app_log_early,
            pure_solver_log="",
            application_log=captured_app_log_early
        )

    def get_norm_factor(cost_list: List[float], name: str) -> float:
        if not cost_list: return 1.0
        avg = np.mean(cost_list)
        factor = avg if avg > 1e-9 else 1.0
        log_to_stream(f"  Normalization factor for {name}: {factor:.4f} (avg: {avg:.4f}, count: {len(cost_list)})")
        return factor

    norm_factor_travel = get_norm_factor([d["raw_costs"]["travel"] for d in possible_assignments_temp_data.values() if "travel" in d["raw_costs"]], "travel")
    norm_factor_age = get_norm_factor([d["raw_costs"]["age"] for d in possible_assignments_temp_data.values() if "age" in d["raw_costs"]], "age")
    norm_factor_frequency = get_norm_factor([d["raw_costs"]["frequency"] for d in possible_assignments_temp_data.values() if "frequency" in d["raw_costs"]], "frequency")
    norm_factor_qualification = get_norm_factor([d["raw_costs"]["qualification"] for d in possible_assignments_temp_data.values() if "qualification" in d["raw_costs"]], "qualification")
    norm_factor_recency = get_norm_factor([d["raw_costs"]["recency"] for d in possible_assignments_temp_data.values() if "recency" in d["raw_costs"]], "recency")

    possible_assignments_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, temp_data in possible_assignments_temp_data.items():
        raw = temp_data["raw_costs"]
        
        norm_travel = raw["travel"] / norm_factor_travel
        norm_age = raw["age"] / norm_factor_age
        norm_frequency = raw["frequency"] / norm_factor_frequency
        norm_qualification = raw["qualification"] / norm_factor_qualification
        norm_recency = raw["recency"] / norm_factor_recency

        total_weighted_cost_float = (
            weight_travel * norm_travel +
            weight_age * norm_age +
            weight_frequency * norm_frequency +
            weight_qualification * norm_qualification +
            weight_past_assignment_recency * norm_recency
        )
        total_weighted_cost_int = int(total_weighted_cost_float * 100)
        log_to_stream(f"    Final cost for {key[0]}-{key[1]}: total_weighted_int={total_weighted_cost_int} (norm_travel={norm_travel:.2f}, norm_age={norm_age:.2f}, norm_freq={norm_frequency:.2f}, norm_qual={norm_qualification:.2f}, norm_recency={norm_recency:.2f})")
        possible_assignments_dict[key] = {**temp_data, "cost": total_weighted_cost_int}

    assignments_by_course: Dict[str, List[Any]] = {} # Var type
    for (lecturer_id_group, course_id_group), data_group in possible_assignments_dict.items():
        variable_group = data_group["variable"]        
        if course_id_group not in assignments_by_course:
            assignments_by_course[course_id_group] = []
        assignments_by_course[course_id_group].append(variable_group)
    
    assignments_by_lecturer: Dict[str, List[Any]] = {lect_id: [] for lect_id in lecturers_dict} # Var type
    for (lecturer_id_group, course_id_group), data_group in possible_assignments_dict.items():
        assignments_by_lecturer[lecturer_id_group].append(data_group["variable"])

    TARGET_PREFECTURES_FOR_TWO_LECTURERS = ["東京都", "愛知県", "大阪府"]
    
    shortage_penalty_terms: List[Any] = [] # LinearExpr terms

    for course_item in courses_dict.values():
        course_id = course_item["id"]
        possible_assignments_for_course = assignments_by_course.get(course_id, [])
        if possible_assignments_for_course:
            course_classroom_id = course_item["classroom_id"]
            course_location = classrooms_dict[course_classroom_id]["location"]

            target_assignment_count = 1
            if course_location in TARGET_PREFECTURES_FOR_TWO_LECTURERS:
                target_assignment_count = 2
            
            if allow_under_assignment:
                model.Add(sum(possible_assignments_for_course) <= target_assignment_count)
                if weight_assignment_shortage > 0:
                    shortage_var = model.NewIntVar(0, target_assignment_count, f'shortage_var_{course_id}')
                    model.Add(shortage_var >= target_assignment_count - sum(possible_assignments_for_course))
                    
                    base_penalty_shortage_scaled = 50000 * 100
                    actual_penalty_for_shortage = int(weight_assignment_shortage * base_penalty_shortage_scaled)
                    if actual_penalty_for_shortage > 0:
                        shortage_penalty_terms.append(shortage_var * actual_penalty_for_shortage)
                        log_to_stream(f"  + Course {course_id}: Added shortage penalty term (shortage_var * {actual_penalty_for_shortage}) for target {target_assignment_count}.")
            else:
                model.Add(sum(possible_assignments_for_course) == target_assignment_count)

    objective_terms: List[Any] = [data["variable"] * data["cost"] for data in possible_assignments_dict.values()] # LinearExpr terms

    if shortage_penalty_terms:
        objective_terms.extend(shortage_penalty_terms)
        log_to_stream(f"  + Added {len(shortage_penalty_terms)} shortage penalty terms to objective.")

    if weight_lecturer_concentration > 0:
        base_penalty_concentration_scaled = 20000 * 100
        actual_penalty_concentration = int(weight_lecturer_concentration * base_penalty_concentration_scaled)

        if actual_penalty_concentration > 0:
            for lecturer_id_loop, lecturer_vars in assignments_by_lecturer.items():
                if not lecturer_vars or len(lecturer_vars) <= 1:
                    continue
                
                num_total_assignments_l = model.NewIntVar(0, len(courses_dict), f'num_total_assignments_{lecturer_id_loop}')
                model.Add(num_total_assignments_l == sum(lecturer_vars))
                
                extra_assignments_l = model.NewIntVar(0, len(courses_dict), f'extra_assign_{lecturer_id_loop}')
                model.Add(extra_assignments_l >= num_total_assignments_l - 1)
                
                objective_terms.append(extra_assignments_l * actual_penalty_concentration)
                log_to_stream(f"  + Lecturer {lecturer_id_loop}: Added concentration penalty term (extra_assign * {actual_penalty_concentration}).")
    
    consecutive_assignment_pair_vars_details: List[Dict[str, Any]] = []
    if weight_consecutive_assignment > 0 and consecutive_day_pairs:
        log_to_stream(f"Processing {len(consecutive_day_pairs)} consecutive day pairs for special assignment reward.")
        for pair_info in consecutive_day_pairs:
            pair_id = pair_info["pair_id"]
            c1_id = pair_info["course1_id"]
            c2_id = pair_info["course2_id"]

            for lecturer_id_loop_pair, lecturer_pair in lecturers_dict.items():
                if lecturer_pair.get("qualification_special_rank") is None:
                    continue

                key1 = (lecturer_id_loop_pair, c1_id)
                key2 = (lecturer_id_loop_pair, c2_id)
                if key1 not in possible_assignments_dict or key2 not in possible_assignments_dict:
                    continue

                log_to_stream(f"  + Potential consecutive pair assignment: Lecturer {lecturer_id_loop_pair} for pair {pair_id} ({c1_id}, {c2_id})")
                pair_var_name = f"y_{lecturer_id_loop_pair}_{pair_id}"
                pair_var = model.NewBoolVar(pair_var_name)
                consecutive_assignment_pair_vars_details.append({
                    "variable": pair_var, "lecturer_id": lecturer_id_loop_pair,
                    "course1_id": c1_id, "course2_id": c2_id, "pair_id": pair_id
                })

                individual_var_c1 = possible_assignments_dict[key1]["variable"]
                individual_var_c2 = possible_assignments_dict[key2]["variable"]
                model.Add(pair_var <= individual_var_c1)
                model.Add(pair_var <= individual_var_c2)

                base_reward_consecutive_scaled = 30000 * 100
                actual_reward_for_pair = int(weight_consecutive_assignment * base_reward_consecutive_scaled)
                
                if actual_reward_for_pair > 0:
                    objective_terms.append(pair_var * -actual_reward_for_pair)
                    log_to_stream(f"    Added reward {-actual_reward_for_pair} for pair_var {pair_var_name}")
    else:
        if weight_consecutive_assignment > 0: # This condition implies consecutive_day_pairs was empty
            log_to_stream("No consecutive day pairs found, or weight_consecutive_assignment is zero. Skipping reward logic.")

    if fixed_assignments:
        log_to_stream(f"Processing {len(fixed_assignments)} fixed assignments (pinning).")
        for fixed_lect_id, fixed_course_id in fixed_assignments:
            assignment_key = (fixed_lect_id, fixed_course_id)
            if assignment_key in possible_assignments_dict:
                var_to_pin = possible_assignments_dict[assignment_key]["variable"]
                model.Add(var_to_pin == 1)
                log_to_stream(f"  + Pinned assignment: {fixed_lect_id} to {fixed_course_id} (variable {var_to_pin.Name()} forced to 1).")
            else:
                log_to_stream(f"  WARNING: Attempted to pin assignment ({fixed_lect_id}, {fixed_course_id}) but it's not a possible assignment. This may lead to an INFEASIBLE solution.")
    else:
        log_to_stream("No fixed assignments specified.")

    # The redundant log for "No consecutive day pairs found..." was removed as it's covered by the earlier else block.

    if objective_terms:
        model.Minimize(sum(objective_terms))
    else:
        log_to_stream("Warning: Objective terms list was empty. Minimizing 0.")
        model.Minimize(0) 

    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    
    # num_workers = os.cpu_count()
    # if num_workers: 
    #     solver.parameters.num_search_workers = num_workers
    #     log_to_stream(f"Solver configured to use {num_workers} workers (CPU cores).")

    solver.log_callback = lambda msg: solver_capture_stream.write(msg + "\n")

    solver_capture_stream.write(f"{SOLVER_LOG_START_MARKER}\n")
    
    status_code = cp_model.UNKNOWN 
    status_code = solver.Solve(model)

    solver_capture_stream.write(f"\n{SOLVER_LOG_END_MARKER}\n")

    captured_solver_log_with_markers = solver_capture_stream.getvalue()
    captured_app_log = app_log_stream.getvalue() # This is now engine's internal processing log

    pure_solver_lines = []
    capturing_solver_log = False
    for line in captured_solver_log_with_markers.splitlines(keepends=False):
        if line == SOLVER_LOG_START_MARKER:
            capturing_solver_log = True
            continue
        if line == SOLVER_LOG_END_MARKER:
            capturing_solver_log = False
            break
        if capturing_solver_log:
            pure_solver_lines.append(line)
    
    pure_solver_log_content = "\n".join(pure_solver_lines)
    if pure_solver_lines:
        pure_solver_log_content += "\n"

    full_application_and_solver_log_content = captured_app_log + captured_solver_log_with_markers

    status_name = solver.StatusName(status_code)
    results: List[Dict[str, Any]] = []
    objective_value_solved: Optional[float] = None # Renamed to avoid conflict with outer scope if any
    solution_status_str = "解なし"

    if status_code == cp_model.OPTIMAL or status_code == cp_model.FEASIBLE:
        solution_status_str = "最適解" if status_code == cp_model.OPTIMAL else "実行可能解"
        objective_value_solved = solver.ObjectiveValue() / 100
        
        lecturer_assignment_counts_this_round: Dict[str, int] = {}
        for pa_data_count_check in possible_assignments_dict.values():
            if solver.Value(pa_data_count_check["variable"]) == 1:
                lecturer_id_for_count = pa_data_count_check["lecturer_id"]
                lecturer_assignment_counts_this_round[lecturer_id_for_count] = \
                    lecturer_assignment_counts_this_round.get(lecturer_id_for_count, 0) + 1

        solved_consecutive_assignments_map: Dict[Tuple[str, str], str] = {}
        if weight_consecutive_assignment > 0 and consecutive_assignment_pair_vars_details:
            for pair_detail in consecutive_assignment_pair_vars_details:
                if solver.Value(pair_detail["variable"]) == 1:
                    lect_id = pair_detail["lecturer_id"]
                    c1_id_res = pair_detail["course1_id"]
                    c2_id_res = pair_detail["course2_id"]
                    p_id_res = pair_detail["pair_id"]
                    solved_consecutive_assignments_map[(lect_id, c1_id_res)] = p_id_res
                    solved_consecutive_assignments_map[(lect_id, c2_id_res)] = p_id_res
                    log_to_stream(f"  Confirmed consecutive assignment for L:{lect_id} on Pair:{p_id_res} (C1:{c1_id_res}, C2:{c2_id_res})")

        for (lecturer_id_res, course_id_res), pa_data in possible_assignments_dict.items():
            if solver.Value(pa_data["variable"]) == 1:
                lecturer = lecturers_dict[lecturer_id_res]
                course = courses_dict[course_id_res]
                results.append({
                    "講師ID": lecturer["id"],
                    "講師名": lecturer["name"],
                    "講座ID": course["id"],
                    "講座名": course["name"],
                    "教室ID": course["classroom_id"],
                    "スケジュール": course['schedule'],
                    "算出コスト(x100)": pa_data["cost"],
                    "教室名": classrooms_dict.get(course["classroom_id"], {}).get("location", "不明"),
                    "移動コスト(元)": pa_data["raw_costs"]["travel"],
                    "年齢コスト(元)": pa_data["raw_costs"]["age"],
                    "頻度コスト(元)": pa_data["raw_costs"]["frequency"],
                    "資格コスト(元)": pa_data["raw_costs"]["qualification"],
                    "当該教室最終割当日からの日数": pa_data["debug_days_since_last_assignment"],
                    "講師一般ランク": lecturer.get("qualification_general_rank"),
                    "講師特別ランク": lecturer.get("qualification_special_rank", "なし"),
                    "講座タイプ": course.get("course_type"),
                    "講座ランク": course.get("rank"),
                    "今回の割り当て回数": lecturer_assignment_counts_this_round.get(lecturer["id"], 0),
                    "連続ペア割当": solved_consecutive_assignments_map.get((lecturer["id"], course["id"]), "なし")
                })
    elif status_code == cp_model.INFEASIBLE:
        solution_status_str = "実行不可能 (制約を満たす解なし)"
    else:
        solution_status_str = f"解探索失敗 (ステータス: {status_name} [{status_code}])"

    return SolverOutput(
        solution_status_str=solution_status_str,
        objective_value=objective_value_solved,
        assignments=results,
        all_courses=list(courses_dict.values()),
        all_lecturers=list(lecturers_dict.values()),
        solver_raw_status_code=status_code,
        full_application_and_solver_log=full_application_and_solver_log_content,
        pure_solver_log=pure_solver_log_content,
        application_log=captured_app_log # This is now the engine's internal processing log
    )
# --- [ここまで app.py より移動] ---