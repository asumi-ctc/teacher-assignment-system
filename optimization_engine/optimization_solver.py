import logging
import datetime
import os
import numpy as np
from ortools.sat.python import cp_model
from typing import List, Optional, Tuple, Dict, Any
# [修正] 相対インポートに変更
from .utils.types import LecturerData, CourseData, ClassroomData, SolverOutput

logger = logging.getLogger(__name__)

# --- 定数定義 ---
RECENCY_COST_CONSTANT = 100000.0
BASE_REWARD_CONSECUTIVE_SCALED = 30000 * 100
BASE_PENALTY_UNASSIGNED_SCALED = 50000 * 100

def solve_assignment(lecturers_data: List[LecturerData],
                     courses_data: List[CourseData],
                     classrooms_data: List[ClassroomData],
                     weight_past_assignment_recency: float,
                     weight_qualification: float,
                     weight_age: float,
                     weight_frequency: float,
                     weight_lecturer_concentration: float,
                     weight_consecutive_assignment: float,
                     today_date: datetime.date,
                     fixed_assignments: Optional[List[Tuple[str, str]]] = None,
                     forced_unassignments: Optional[List[Tuple[str, str]]] = None) -> SolverOutput:
    
    log_buffer = []
    def log_to_buffer(message: str):
        log_buffer.append(f"[SolverEngineLog] {message}")

    def flush_log_buffer():
        nonlocal log_buffer
        if log_buffer:
            logger.info("\n".join(log_buffer))
            log_buffer = []

    try:
        model = cp_model.CpModel()

        # --- 1. データ前処理と変数定義 ---
        lecturers_dict = {lecturer['id']: lecturer for lecturer in lecturers_data}
        courses_dict = {course['id']: course for course in courses_data}

        for lecturer in lecturers_dict.values():
            lecturer["_availability_set"] = set(lecturer.get("availability", []))
            latest_assignments = {}
            for pa in lecturer.get("past_assignments", []):
                if pa["classroom_id"] not in latest_assignments:
                    latest_assignments[pa["classroom_id"]] = pa["date"]
            lecturer["_latest_assignment_by_classroom"] = latest_assignments

        possible_assignments_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
        forced_unassignments_set = set(forced_unassignments) if forced_unassignments else set()

        for lecturer in lecturers_dict.values():
            for course in courses_dict.values():
                lecturer_id, course_id = lecturer["id"], course["id"]

                if (lecturer_id, course_id) in forced_unassignments_set:
                    continue
                
                can_assign_by_qualification = False
                if course["course_type"] == "general":
                    if lecturer.get("qualification_special_rank") is not None or \
                       (lecturer.get("qualification_general_rank") is not None and lecturer["qualification_general_rank"] <= course["rank"]):
                        can_assign_by_qualification = True
                elif course["course_type"] == "special":
                    if lecturer.get("qualification_special_rank") is not None and lecturer["qualification_special_rank"] <= course["rank"]:
                        can_assign_by_qualification = True
                
                if not can_assign_by_qualification or course["schedule"] not in lecturer["_availability_set"]:
                    continue

                var = model.NewBoolVar(f'x_{lecturer_id}_{course_id}')
                possible_assignments_dict[(lecturer_id, course_id)] = {
                    "lecturer_id": lecturer_id, "course_id": course_id, "variable": var
                }

        # --- 2. 制約の構築 ---
        assignments_by_course: Dict[str, List[Any]] = {c_id: [] for c_id in courses_dict}
        assignments_by_lecturer: Dict[str, List[Any]] = {l_id: [] for l_id in lecturers_dict}
        for (l_id, c_id), data in possible_assignments_dict.items():
            assignments_by_course[c_id].append(data["variable"])
            assignments_by_lecturer[l_id].append(data["variable"])
        
        if fixed_assignments:
            for fixed_lect_id, fixed_course_id in fixed_assignments:
                if (fixed_lect_id, fixed_course_id) in possible_assignments_dict:
                    model.Add(possible_assignments_dict[(fixed_lect_id, fixed_course_id)]["variable"] == 1)

        # --- 3. 目的関数の構築 ---
        objective_terms = []

        # [ソフト制約] 講座への割り当て
        for course_id, possible_vars in assignments_by_course.items():
            model.Add(sum(possible_vars) <= 1)
            is_assigned = model.NewBoolVar(f'is_assigned_{course_id}')
            model.Add(is_assigned == sum(possible_vars))
            objective_terms.append((1 - is_assigned) * BASE_PENALTY_UNASSIGNED_SCALED)

        # [ソフト制約] 講師の割り当て集中度ペナルティ
        if weight_lecturer_concentration > 0:
            base_penalty_concentration = 1000 * 100 
            for lecturer_id, lecturer_vars in assignments_by_lecturer.items():
                if len(lecturer_vars) > 1:
                    num_assignments = model.NewIntVar(0, len(lecturer_vars), f'num_assignments_{lecturer_id}')
                    model.Add(num_assignments == sum(lecturer_vars))
                    
                    extra_assignments = model.NewIntVar(0, len(lecturer_vars), f'extra_assignments_{lecturer_id}')
                    model.AddMaxEquality(extra_assignments, [num_assignments - 1, 0])
                    
                    penalty_amount = int(weight_lecturer_concentration * base_penalty_concentration)
                    objective_terms.append(extra_assignments * penalty_amount)

        # [コスト計算] 各種ペナルティ/報酬
        cost_keys_for_norm = ["age", "frequency", "qualification", "recency"]
        raw_cost_lists = {key: [] for key in cost_keys_for_norm}
        
        for (l_id, c_id), data in possible_assignments_dict.items():
            lecturer, course = lecturers_dict[l_id], courses_dict[c_id]
            
            qualification_cost = 0.0
            if course["course_type"] == "general":
                qualification_cost = float(lecturer["qualification_general_rank"])
            elif course["course_type"] == "special":
                qualification_cost = float(lecturer.get("qualification_special_rank", 99))

            latest_assignment_date = lecturer["_latest_assignment_by_classroom"].get(course["classroom_id"])
            recency_cost = 0.0
            days_since = -1
            if latest_assignment_date:
                days_since = (today_date - latest_assignment_date).days
                if days_since >= 0:
                    recency_cost = RECENCY_COST_CONSTANT / (days_since + 1.0)
            
            data["raw_costs"] = {
                "age": float(lecturer.get("age", 99)),
                "frequency": float(len(lecturer.get("past_assignments", []))),
                "qualification": qualification_cost,
                "recency": recency_cost
            }
            data["debug_days_since_last_assignment"] = days_since
            for key in cost_keys_for_norm:
                raw_cost_lists[key].append(data["raw_costs"][key])

        def get_norm_factor(cost_list: List[float]) -> float:
            if not cost_list: return 1.0
            avg = np.mean(cost_list)
            return avg if avg > 1e-9 else 1.0
        
        norm_factors = {key: get_norm_factor(raw_cost_lists[key]) for key in cost_keys_for_norm}
        
        for data in possible_assignments_dict.values():
            raw = data["raw_costs"]
            total_weighted_cost_float = (
                weight_age * (raw["age"] / norm_factors["age"]) +
                weight_frequency * (raw["frequency"] / norm_factors["frequency"]) +
                weight_qualification * (raw["qualification"] / norm_factors["qualification"]) +
                weight_past_assignment_recency * (raw["recency"] / norm_factors["recency"])
            )
            data["cost"] = int(round(total_weighted_cost_float * 100))
            objective_terms.append(data["variable"] * data["cost"])
        
        model.Minimize(sum(objective_terms))
        
        # --- 4. 求解 ---
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = os.cpu_count() or 8
        solver.log_callback = lambda msg: log_to_buffer(f"[OR-Tools Solver] {msg.strip()}")
        
        flush_log_buffer()
        status = solver.Solve(model)
        flush_log_buffer()

        # --- 5. 結果の整形 ---
        solution_status_str = solver.StatusName(status)
        objective_value_solved = solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None
        
        assignments = []
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            lecturer_assignment_counts_this_round: Dict[str, int] = {}
            for (l_id, c_id), data in possible_assignments_dict.items():
                if solver.Value(data["variable"]) == 1:
                    lecturer_assignment_counts_this_round[l_id] = lecturer_assignment_counts_this_round.get(l_id, 0) + 1
            
            for (l_id, c_id), data in possible_assignments_dict.items():
                if solver.Value(data["variable"]) == 1:
                    assignments.append({
                        "講師ID": l_id, 
                        "講座ID": c_id,
                        "算出コスト(x100)": data.get("cost", 0),
                        "年齢コスト(元)": data["raw_costs"]["age"],
                        "頻度コスト(元)": data["raw_costs"]["frequency"],
                        "資格コスト(元)": data["raw_costs"]["qualification"],
                        "当該教室最終割当日からの日数": data["debug_days_since_last_assignment"],
                        "今回の割り当て回数": lecturer_assignment_counts_this_round.get(l_id, 0),
                        "連続ペア割当": "なし"
                    })
        
        return SolverOutput(
            solution_status_str=solution_status_str,
            objective_value=objective_value_solved,
            assignments=assignments,
            all_courses=courses_data,
            all_lecturers=lecturers_data,
            solver_raw_status_code=status
        )
    finally:
        flush_log_buffer()
