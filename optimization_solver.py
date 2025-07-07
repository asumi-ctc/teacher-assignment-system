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
    
    possible_assignments_dict: Dict[Tuple[str, str], Dict[str, Any]] = {}
    
    # --- 1. 割り当て候補とコストの計算 ---
    for lecturer_id, lecturer in lecturers_map.items():
        for course_id, course in courses_map.items():
            # (ここに、割り当て可能かどうかのチェックロジックを実装)
            # ...
            
            # (ここに、各コスト要素の計算ロジックを実装)
            # ...
            pass # 仮

    # (もし割り当て候補がなければ、早期にリターン)
    if not possible_assignments_dict:
        return SolverOutput(solution_status_str="前提条件エラー", objective_value=None, assignments=[], all_courses=list(courses_map.values()), all_lecturers=list(lecturers_map.values()), solver_raw_status_code=cp_model.UNKNOWN)

    # --- 2. 制約と目的関数の構築 ---
    objective_terms = [d["variable"] * d["cost"] for d in possible_assignments_dict.values()]
    
    # (ここに、各制約・目的関数項を追加するロジックを実装)
    # 例: 講座ごとの割り当て人数制約
    assignments_by_course = {cid: [] for cid in courses_map}
    for (lid, cid), data in possible_assignments_dict.items():
        assignments_by_course[cid].append(data["variable"])
    
    for course_id, variables in assignments_by_course.items():
        if variables:
            model.Add(sum(variables) <= 2 if classrooms_map[courses_map[course_id]['classroom_id']]['location'] in ["東京都", "愛知県", "大阪府"] else 1)

    model.Minimize(sum(objective_terms))

    # --- 3. 求解 ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solver_params.max_search_seconds
    status_code = solver.Solve(model)

    # --- 4. 結果の抽出 ---
    results: List[Dict] = []
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (lecturer_id, course_id), data in possible_assignments_dict.items():
            if solver.Value(data["variable"]) == 1:
                results.append({"講師ID": lecturer_id, "講座ID": course_id})

    status_name_map = {cp_model.OPTIMAL: "最適解", cp_model.FEASIBLE: "実行可能解", cp_model.INFEASIBLE: "実行不可能", cp_model.UNKNOWN: "解探索失敗"}
    
    return SolverOutput(
        solution_status_str=status_name_map.get(status_code, f"その他({status_code})"),
        objective_value=solver.ObjectiveValue() / 100 if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        assignments=results,
        all_courses=list(courses_map.values()),
        all_lecturers=list(lecturers_map.values()),
        solver_raw_status_code=status_code
    )