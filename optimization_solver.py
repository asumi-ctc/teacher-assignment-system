# optimization_solver.py (完全・修正版)

import logging
import datetime
import os
import numpy as np
from ortools.sat.python import cp_model
from typing import List, Optional, Tuple, Dict, Any

from utils.types import SolverParameters, SolverOutput

logger = logging.getLogger(__name__)

# --- 定数定義 ---
RECENCY_COST_CONSTANT = 100000.0
BASE_PENALTY_SHORTAGE_SCALED = 50000 * 100
BASE_PENALTY_CONCENTRATION_SCALED = 20000 * 100
BASE_REWARD_CONSECUTIVE_SCALED = 30000 * 100

def solve_assignment(
    lecturers_map: Dict[str, Any],
    courses_map: Dict[str, Any],
    classrooms_map: Dict[str, Any],
    travel_costs_matrix: Dict[Tuple[str, str], int],
    solver_params: SolverParameters,
    today_date: datetime.date,
    fixed_assignments: Optional[List[Tuple[str, str]]] = None,
    forced_unassignments: Optional[List[Tuple[str, str]]] = None
) -> SolverOutput:
    
    log_buffer = []
    def log_to_buffer(message: str):
        log_buffer.append(f"[SolverEngineLog] {message}")
    def flush_log_buffer():
        nonlocal log_buffer
        if log_buffer: logger.info("\n".join(log_buffer))
        log_buffer = []

    try:
        model = cp_model.CpModel()
        weights = solver_params.weights
        allow_under_assignment = solver_params.allow_under_assignment

        # --- 1. 割り当て候補とコストの計算 ---
        possible_assignments_temp_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
        forced_unassignments_set = set(forced_unassignments) if forced_unassignments else set()

        for lecturer_id, lecturer in lecturers_map.items():
            for course_id, course in courses_map.items():
                if (lecturer_id, course_id) in forced_unassignments_set: continue

                can_assign, qual_cost = False, 0
                if course["course_type"] == "general":
                    if lecturer.get("qualification_special_rank") is not None:
                        can_assign, qual_cost = True, lecturer["qualification_general_rank"]
                    elif lecturer["qualification_general_rank"] <= course["rank"]:
                        can_assign, qual_cost = True, lecturer["qualification_general_rank"]
                elif course["course_type"] == "special":
                    if lecturer.get("qualification_special_rank") is not None and lecturer["qualification_special_rank"] <= course["rank"]:
                        can_assign, qual_cost = True, lecturer["qualification_special_rank"]
                
                if not can_assign or course["schedule"] not in lecturer["availability"]: continue

                var = model.NewBoolVar(f'x_{lecturer_id}_{course_id}')
                travel_cost = travel_costs_matrix.get((lecturer["home_classroom_id"], course["classroom_id"]), 999999)
                age_cost = float(lecturer.get("age", 99))
                frequency_cost = float(len(lecturer.get("past_assignments", [])))
                
                recency_cost, days_since = 0.0, -1
                past_assignments_at_classroom = [pa for pa in lecturer.get("past_assignments", []) if pa["classroom_id"] == course["classroom_id"]]
                if past_assignments_at_classroom:
                    try:
                        latest_date = datetime.datetime.strptime(past_assignments_at_classroom[0]["date"], "%Y-%m-%d").date()
                        days_since = (today_date - latest_date).days
                        if days_since >= 0: recency_cost = RECENCY_COST_CONSTANT / (days_since + 1.0)
                    except (ValueError, IndexError): pass

                possible_assignments_temp_data[(lecturer_id, course_id)] = {
                    "variable": var, "lecturer_id": lecturer_id, "course_id": course_id,
                    "raw_costs": {"travel": travel_cost, "age": age_cost, "frequency": frequency_cost, "qualification": qual_cost, "recency": recency_cost},
                    "debug_days_since_last_assignment": days_since
                }

        if not possible_assignments_temp_data:
            return SolverOutput(solution_status_str="前提条件エラー", objective_value=None, assignments=[], all_courses=list(courses_map.values()), all_lecturers=list(lecturers_map.values()), solver_raw_status_code=cp_model.UNKNOWN)

        # --- 2. コスト正規化と最終コスト計算 ---
        cost_keys = ["travel", "age", "frequency", "qualification", "recency"]
        norm_factors = {key: np.mean([d["raw_costs"][key] for d in possible_assignments_temp_data.values() if d["raw_costs"][key] > 0]) for key in cost_keys}
        cost_weights_map = {key: getattr(weights, key.replace("_recency", "")) for key in cost_keys} # A bit tricky mapping for recency
        cost_weights_map["recency"] = weights.past_assignment_recency

        possible_assignments_dict = {}
        for key, temp_data in possible_assignments_temp_data.items():
            total_cost = 0.0
            for cost_key, weight in cost_weights_map.items():
                if weight > 0:
                    factor = norm_factors.get(cost_key)
                    if factor and factor > 1e-9:
                        total_cost += weight * (temp_data["raw_costs"][cost_key] / factor)
            temp_data["cost"] = int(round(total_cost * 100))
            possible_assignments_dict[key] = temp_data

        # --- 3. 制約と目的関数の構築 ---
        objective_terms = [d["variable"] * d["cost"] for d in possible_assignments_dict.values()]
        
        assignments_by_course = {cid: [] for cid in courses_map}
        for (lid, cid), data in possible_assignments_dict.items():
            assignments_by_course[cid].append(data["variable"])
        
        for course_id, variables in assignments_by_course.items():
            if not variables: continue
            target_count = 2 if classrooms_map[courses_map[course_id]['classroom_id']]['location'] in ["東京都", "愛知県", "大阪府"] else 1
            if allow_under_assignment:
                model.Add(sum(variables) <= target_count)
            else:
                model.Add(sum(variables) == target_count)
        
        model.Minimize(sum(objective_terms))
        flush_log_buffer()

        # --- 4. 求解 ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = solver_params.max_search_seconds
        solver.parameters.num_search_workers = os.cpu_count() or 1
        status_code = solver.Solve(model)

        # --- 5. 結果の抽出 ---
        results: List[Dict] = []
        if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for (lecturer_id, course_id), data in possible_assignments_dict.items():
                if solver.Value(data["variable"]) == 1:
                    results.append({"講師ID": lecturer_id, "講座ID": course_id, **data})
        
        status_map = {cp_model.OPTIMAL: "最適解", cp_model.FEASIBLE: "実行可能解", cp_model.INFEASIBLE: "実行不可能", cp_model.UNKNOWN: "解探索失敗"}
        return SolverOutput(
            solution_status_str=status_map.get(status_code, f"その他({status_code})"),
            objective_value=solver.ObjectiveValue() / 100 if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
            assignments=results,
            all_courses=list(courses_map.values()),
            all_lecturers=list(lecturers_map.values()),
            solver_raw_status_code=status_code
        )
    finally:
        flush_log_buffer()