# optimization_solver.py

import logging
import datetime
import os
import numpy as np
from ortools.sat.python import cp_model
from typing import TypedDict, List, Optional, Tuple, Dict, Any

# gatewayから型定義をインポート
from optimization_gateway import SolverParameters

logger = logging.getLogger(__name__)

# --- 定数定義 ---
RECENCY_COST_CONSTANT = 100000.0
BASE_PENALTY_SHORTAGE_SCALED = 50000 * 100
BASE_PENALTY_CONCENTRATION_SCALED = 20000 * 100
BASE_REWARD_CONSECUTIVE_SCALED = 30000 * 100

class SolverOutput(TypedDict):
    solution_status_str: str
    objective_value: Optional[float]
    assignments: List[dict]
    all_courses: List[dict]
    all_lecturers: List[dict]
    solver_raw_status_code: int

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
        if log_buffer:
            logger.info("\n".join(log_buffer))
            log_buffer = []

    try:
        model = cp_model.CpModel()
        weights = solver_params.weights
        allow_under_assignment = solver_params.allow_under_assignment

        # --- 連日講座ペアのリストアップ ---
        consecutive_day_pairs = [] # ... (このロジックは前回から変更なし) ...

        # --- モデル構築 ---
        possible_assignments_temp_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
        forced_unassignments_set = set(forced_unassignments) if forced_unassignments else set()

        for lecturer_id, lecturer in lecturers_map.items():
            for course_id, course in courses_map.items():
                if (lecturer_id, course_id) in forced_unassignments_set:
                    continue
                
                # 資格・スケジュールチェック
                can_assign_by_qualification = False
                qualification_cost_for_this_assignment = 0
                if course["course_type"] == "general":
                    if lecturer.get("qualification_special_rank") is not None or lecturer["qualification_general_rank"] <= course["rank"]:
                        can_assign_by_qualification = True
                        qualification_cost_for_this_assignment = lecturer["qualification_general_rank"]
                elif course["course_type"] == "special":
                    if lecturer.get("qualification_special_rank") is not None and lecturer["qualification_special_rank"] <= course["rank"]:
                        can_assign_by_qualification = True
                        qualification_cost_for_this_assignment = lecturer["qualification_special_rank"]

                if not can_assign_by_qualification or course["schedule"] not in lecturer["availability"]:
                    continue

                # コスト計算
                travel_cost = travel_costs_matrix.get((lecturer["home_classroom_id"], course["classroom_id"]), 999999)
                age_cost = float(lecturer.get("age", 99))
                frequency_cost = float(len(lecturer.get("past_assignments", [])))
                
                # ... (他のコスト計算ロジック) ...

                possible_assignments_temp_data[(lecturer_id, course_id)] = {
                    "lecturer_id": lecturer_id, "course_id": course_id, "variable": model.NewBoolVar(f'x_{lecturer_id}_{course_id}'),
                    "raw_costs": {"travel": travel_cost, "age": age_cost, "frequency": frequency_cost, "qualification": qualification_cost_for_this_assignment, "recency": 0.0},
                    "debug_days_since_last_assignment": -1
                }
        
        if not possible_assignments_temp_data:
            return SolverOutput(solution_status_str="前提条件エラー (割り当て候補なし)", objective_value=None, assignments=[], all_courses=list(courses_map.values()), all_lecturers=list(lecturers_map.values()), solver_raw_status_code=cp_model.UNKNOWN)

        # --- コスト正規化と最終コスト計算 ---
        norm_factors = {key: np.mean([d["raw_costs"][key] for d in possible_assignments_temp_data.values()]) for key in ["travel", "age", "frequency", "qualification", "recency"]}
        cost_weights_map = {
            "travel": weights.travel, "age": weights.age, "frequency": weights.frequency,
            "qualification": weights.qualification, "recency": weights.past_assignment_recency
        }
        possible_assignments_dict = {}
        for key, temp_data in possible_assignments_temp_data.items():
            total_weighted_cost_float = sum(weight * (temp_data["raw_costs"][cost_key] / (norm_factors.get(cost_key) or 1.0)) for cost_key, weight in cost_weights_map.items() if weight > 0)
            possible_assignments_dict[key] = {**temp_data, "cost": int(round(total_weighted_cost_float * 100))}

        # --- 制約と目的関数の構築 ---
        assignments_by_course = {cid: [] for cid in courses_map}
        for (lid, cid), data in possible_assignments_dict.items():
            assignments_by_course[cid].append(data["variable"])
        
        # ... (他の制約・目的関数項の構築ロジック) ...
        objective_terms = [d["variable"] * d["cost"] for d in possible_assignments_dict.values()]
        model.Minimize(sum(objective_terms))
        
        flush_log_buffer()

        # --- 求解 ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = solver_params.max_search_seconds
        status_code = solver.Solve(model)

        # --- 結果の抽出 ---
        results = []
        # ... (結果抽出ロジック) ...

        status_name_map = {cp_model.OPTIMAL: "最適解", cp_model.FEASIBLE: "実行可能解", cp_model.INFEASIBLE: "実行不可能", cp_model.UNKNOWN: "解探索失敗"}
        return SolverOutput(
            solution_status_str=status_name_map.get(status_code, f"その他({status_code})"),
            objective_value=solver.ObjectiveValue() / 100 if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
            assignments=results,
            all_courses=list(courses_map.values()),
            all_lecturers=list(lecturers_map.values()),
            solver_raw_status_code=status_code
        )
    finally:
        flush_log_buffer()