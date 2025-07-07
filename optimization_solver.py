# optimization_solver.py

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
    
    model = cp_model.CpModel()
    weights = solver_params.weights
    allow_under_assignment = solver_params.allow_under_assignment

    # --- 1. 割り当て候補とコストの計算 ---
    possible_assignments_temp_data = {}
    forced_unassignments_set = set(forced_unassignments) if forced_unassignments else set()

    for lecturer_id, lecturer in lecturers_map.items():
        for course_id, course in courses_map.items():
            if (lecturer_id, course_id) in forced_unassignments_set:
                continue

            # 資格・スケジュールチェック
            course_type = course["course_type"]
            course_rank = course["rank"]
            lecturer_general_rank = lecturer["qualification_general_rank"]
            lecturer_special_rank = lecturer.get("qualification_special_rank")
            can_assign = False
            qual_cost = 0
            if course_type == "general":
                if lecturer_special_rank is not None:
                    can_assign, qual_cost = True, lecturer_general_rank
                elif lecturer_general_rank <= course_rank:
                    can_assign, qual_cost = True, lecturer_general_rank
            elif course_type == "special":
                if lecturer_special_rank is not None and lecturer_special_rank <= course_rank:
                    can_assign, qual_cost = True, lecturer_special_rank
            
            if not can_assign or course["schedule"] not in lecturer["availability"]:
                continue

            var = model.NewBoolVar(f'x_{lecturer_id}_{course_id}')
            # (コスト計算ロジックをここに移植)
            # ...

            possible_assignments_temp_data[(lecturer_id, course_id)] = {
                "variable": var,
                # ... (他のコスト情報)
            }
            
    if not possible_assignments_temp_data:
        return SolverOutput(solution_status_str="前提条件エラー", objective_value=None, assignments=[], all_courses=list(courses_map.values()), all_lecturers=list(lecturers_map.values()), solver_raw_status_code=cp_model.UNKNOWN)

    # --- 2. コスト正規化と最終コスト計算 ---
    # (コスト正規化と最終コスト計算ロジックをここに移植)
    possible_assignments_dict = {} # 正規化後の最終データ
    
    # --- 3. 制約と目的関数の構築 ---
    objective_terms = [d["variable"] * d["cost"] for d in possible_assignments_dict.values()]
    # (制約と目的関数項の追加ロジックをここに移植)
    
    model.Minimize(sum(objective_terms))

    # --- 4. 求解 ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_params.max_search_seconds
    status_code = solver.Solve(model)

    # --- 5. 結果の抽出 ---
    results: List[Dict] = []
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (lecturer_id, course_id), data in possible_assignments_dict.items():
            if solver.Value(data["variable"]) == 1:
                results.append({"講師ID": lecturer_id, "講座ID": course_id, **data}) # コスト情報なども含める

    status_name_map = {cp_model.OPTIMAL: "最適解", cp_model.FEASIBLE: "実行可能解", cp_model.INFEASIBLE: "実行不可能", cp_model.UNKNOWN: "解探索失敗"}
    
    return SolverOutput(
        solution_status_str=status_name_map.get(status_code, f"その他({status_code})"),
        objective_value=solver.ObjectiveValue() / 100 if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        assignments=results,
        all_courses=list(courses_map.values()),
        all_lecturers=list(lecturers_map.values()),
        solver_raw_status_code=status_code
    )