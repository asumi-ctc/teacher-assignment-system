# ==============================================================================
# 2. optimization_solver.py (最適化モデルの構築と求解)
# ==============================================================================
import logging
import datetime
import os
import numpy as np
from ortools.sat.python import cp_model
from typing import List, Optional, Tuple, Dict, Any
from .utils.types import LecturerData, CourseData, ClassroomData, SolverOutput

logger = logging.getLogger(__name__)

# --- 定数定義 ---
RECENCY_COST_CONSTANT = 100000.0
BASE_REWARD_CONSECUTIVE_SCALED = 30000 * 100

def _build_base_model_and_vars(
    model: cp_model.CpModel,
    lecturers_data: List[LecturerData],
    courses_data: List[CourseData],
    forced_unassignments: Optional[List[Tuple[str, str]]],
    fixed_assignments: Optional[List[Tuple[str, str]]]
) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """基本的な変数とハード制約をモデルに追加し、関連する辞書を返す"""
    lecturers_dict = {l['id']: l for l in lecturers_data}
    courses_dict = {c['id']: c for c in courses_data}

    for lecturer in lecturers_dict.values():
        lecturer["_availability_set"] = set(lecturer.get("availability", []))

    possible_assignments_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
    forced_unassignments_set = set(forced_unassignments) if forced_unassignments else set()

    for lecturer in lecturers_dict.values():
        for course in courses_dict.values():
            lecturer_id = lecturer["id"]
            course_id = course["id"]

            if (lecturer_id, course_id) in forced_unassignments_set:
                continue
            
            can_assign_by_qualification = False
            if course["course_type"] == "general":
                if lecturer.get("qualification_special_rank") is not None or \
                   lecturer["qualification_general_rank"] <= course["rank"]:
                    can_assign_by_qualification = True
            elif course["course_type"] == "special":
                if lecturer.get("qualification_special_rank") is not None and \
                   lecturer["qualification_special_rank"] <= course["rank"]:
                    can_assign_by_qualification = True
            
            if not can_assign_by_qualification or course["schedule"] not in lecturer["_availability_set"]:
                continue

            var = model.NewBoolVar(f'x_{lecturer_id}_{course_id}')
            possible_assignments_dict[(lecturer_id, course_id)] = {
                "lecturer_id": lecturer_id, "course_id": course_id, "variable": var
            }

    assignments_by_course: Dict[str, List[Any]] = {c_id: [] for c_id in courses_dict}
    assignments_by_lecturer: Dict[str, List[Any]] = {l_id: [] for l_id in lecturers_dict}
    for (l_id, c_id), data in possible_assignments_dict.items():
        assignments_by_course[c_id].append(data["variable"])
        assignments_by_lecturer[l_id].append(data["variable"])
    
    for course_id, possible_vars in assignments_by_course.items():
        if possible_vars:
            model.Add(sum(possible_vars) <= 1)

    if fixed_assignments:
        for fixed_lect_id, fixed_course_id in fixed_assignments:
            if (fixed_lect_id, fixed_course_id) in possible_assignments_dict:
                model.Add(possible_assignments_dict[(fixed_lect_id, fixed_course_id)]["variable"] == 1)

    return lecturers_dict, courses_dict, possible_assignments_dict, assignments_by_lecturer, assignments_by_course

def solve_assignment_lexicographically(
    lecturers_data: List[LecturerData],
    courses_data: List[CourseData],
    classrooms_data: List[ClassroomData],
    **kwargs: Any
) -> SolverOutput:
    
    today_date = kwargs["today_date"]
    log_buffer = []
    def log_to_buffer(message: str):
        log_buffer.append(f"[SolverEngineLog] {message}")
    
    def flush_log_buffer():
        nonlocal log_buffer
        if log_buffer:
            logger.info("\n".join(log_buffer))
            log_buffer = []

    # --- フェーズ1: 総割り当て数の最大化 ---
    log_to_buffer("--- Starting Lexicographical Phase 1: Maximize Total Assignments ---")
    model_phase1 = cp_model.CpModel()
    _, _, possible_assignments_dict, _, _ = _build_base_model_and_vars(
        model_phase1, lecturers_data, courses_data, 
        kwargs.get("forced_unassignments"), kwargs.get("fixed_assignments")
    )
    
    all_assignment_vars = [d["variable"] for d in possible_assignments_dict.values()]
    total_assignments = model_phase1.NewIntVar(0, len(all_assignment_vars), "total_assignments")
    model_phase1.Add(total_assignments == sum(all_assignment_vars))
    model_phase1.Maximize(total_assignments)
    
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = os.cpu_count() or 8
    solver.log_callback = lambda msg: log_to_buffer(f"[OR-Tools Solver] {msg.strip()}")
    
    flush_log_buffer()
    status_phase1 = solver.Solve(model_phase1)
    flush_log_buffer()
    
    if status_phase1 not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return SolverOutput(solution_status_str="実行不可能 (フェーズ1)", objective_value=None, assignments=[], all_courses=courses_data, all_lecturers=lecturers_data, solver_raw_status_code=status_phase1)
        
    max_assignments_val = solver.Value(total_assignments)
    log_to_buffer(f"Phase 1 complete. Max possible assignments: {max_assignments_val}")

    # --- フェーズ2: 割り当ての平準化 ---
    log_to_buffer("--- Starting Lexicographical Phase 2: Minimize Max Assignments per Lecturer ---")
    model_phase2 = cp_model.CpModel()
    _, _, possible_assignments_dict, assignments_by_lecturer, _ = _build_base_model_and_vars(
        model_phase2, lecturers_data, courses_data,
        kwargs.get("forced_unassignments"), kwargs.get("fixed_assignments")
    )
    
    all_assignment_vars_p2 = [d["variable"] for d in possible_assignments_dict.values()]
    total_assignments_p2 = model_phase2.NewIntVar(0, len(all_assignment_vars_p2), "total_assignments_p2")
    model_phase2.Add(total_assignments_p2 == sum(all_assignment_vars_p2))
    model_phase2.Add(total_assignments_p2 == max_assignments_val)

    num_assignments_per_lecturer = []
    for lect_id, lect_vars in assignments_by_lecturer.items():
        if lect_vars:
            num_assignments = model_phase2.NewIntVar(0, len(lect_vars), f"num_assignments_{lect_id}")
            model_phase2.Add(num_assignments == sum(lect_vars))
            num_assignments_per_lecturer.append(num_assignments)

    max_assignments_var = model_phase2.NewIntVar(0, len(courses_data), "max_assignments_var")
    if num_assignments_per_lecturer:
        model_phase2.AddMaxEquality(max_assignments_var, num_assignments_per_lecturer)
    else:
        model_phase2.Add(max_assignments_var == 0)
    model_phase2.Minimize(max_assignments_var)
    
    flush_log_buffer()
    status_phase2 = solver.Solve(model_phase2)
    flush_log_buffer()
    
    if status_phase2 not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return SolverOutput(solution_status_str="実行不可能 (フェーズ2)", objective_value=None, assignments=[], all_courses=courses_data, all_lecturers=lecturers_data, solver_raw_status_code=status_phase2)
        
    min_max_assignments_val = solver.Value(max_assignments_var)
    log_to_buffer(f"Phase 2 complete. Min-max assignments per lecturer: {min_max_assignments_val}")

    # --- フェーズ3: ペナルティ（コスト）の最小化 ---
    log_to_buffer("--- Starting Lexicographical Phase 3: Minimize Penalty Costs ---")
    model_phase3 = cp_model.CpModel()
    lecturers_dict, courses_dict, possible_assignments_dict, assignments_by_lecturer, _ = _build_base_model_and_vars(
        model_phase3, lecturers_data, courses_data,
        kwargs.get("forced_unassignments"), kwargs.get("fixed_assignments")
    )
    
    all_assignment_vars_p3 = [d["variable"] for d in possible_assignments_dict.values()]
    total_assignments_p3 = model_phase3.NewIntVar(0, len(all_assignment_vars_p3), "total_assignments_p3")
    model_phase3.Add(total_assignments_p3 == sum(all_assignment_vars_p3))
    model_phase3.Add(total_assignments_p3 == max_assignments_val)

    for lect_id, lect_vars in assignments_by_lecturer.items():
        if lect_vars:
            num_assignments_p3 = model_phase3.NewIntVar(0, len(lect_vars), f"num_assignments_p3_{lect_id}")
            model_phase3.Add(num_assignments_p3 == sum(lect_vars))
            model_phase3.Add(num_assignments_p3 <= min_max_assignments_val)
    
    for lecturer in lecturers_dict.values():
        latest_assignments = {}
        for pa in lecturer.get("past_assignments", []):
            if pa["classroom_id"] not in latest_assignments:
                latest_assignments[pa["classroom_id"]] = pa["date"]
        lecturer["_latest_assignment_by_classroom"] = latest_assignments
    
    cost_keys_for_norm = ["age", "frequency", "qualification", "recency"]
    raw_cost_lists = {key: [] for key in cost_keys_for_norm}
    
    for (l_id, c_id), data in possible_assignments_dict.items():
        lecturer = lecturers_dict[l_id]
        course = courses_dict[c_id]
        
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
            kwargs.get("weight_age", 0.0) * (raw["age"] / norm_factors["age"]) +
            kwargs.get("weight_frequency", 0.0) * (raw["frequency"] / norm_factors["frequency"]) +
            kwargs.get("weight_qualification", 0.0) * (raw["qualification"] / norm_factors["qualification"]) +
            kwargs.get("weight_past_assignment_recency", 0.0) * (raw["recency"] / norm_factors["recency"])
        )
        data["cost"] = int(round(total_weighted_cost_float * 100))

    objective_terms = [data["variable"] * data["cost"] for data in possible_assignments_dict.values()]

    model_phase3.Minimize(sum(objective_terms))
    
    flush_log_buffer()
    status_phase3 = solver.Solve(model_phase3)
    flush_log_buffer()

    final_status = status_phase3
    final_solver = solver

    solution_status_str = final_solver.StatusName(final_status)
    objective_value_solved = final_solver.ObjectiveValue() / 100 if final_status in [cp_model.OPTIMAL, cp_model.FEASIBLE] and final_solver.ObjectiveValue() is not None else None
    
    assignments = []
    if final_status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        lecturer_assignment_counts_this_round: Dict[str, int] = {}
        for (l_id, c_id), data in possible_assignments_dict.items():
            if final_solver.Value(data["variable"]) == 1:
                lecturer_assignment_counts_this_round[l_id] = lecturer_assignment_counts_this_round.get(l_id, 0) + 1
        
        for (l_id, c_id), data in possible_assignments_dict.items():
            if final_solver.Value(data["variable"]) == 1:
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
        solver_raw_status_code=final_status
    )
