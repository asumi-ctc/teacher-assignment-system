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
    all_courses: List[dict] # 元のリストを返すために残す
    all_lecturers: List[dict] # 元のリストを返すために残す
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
    """
    前処理済みのデータと設定オブジェクトに基づき、最適化計算を実行する。
    """
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

        # --- パラメータの展開 ---
        weights = solver_params.weights
        allow_under_assignment = solver_params.allow_under_assignment

        # --- [削除] データ前処理（リストから辞書へ）のコードは不要に ---

        # --- ステップ1: 連日講座ペアのリストアップ (courses_map を使用) ---
        # ... (ロジックは同じだが、courses_dict を courses_map に置換) ...
        
        # ... (以降のモデル構築ロジックも、_dict を _map に置換し、
        #     個別の weight_* 変数を weights.* 経由で参照するように変更) ...

        # --- Main logic for model building and solving ---
        # (ここでは主要な変更箇所のみ示します)
        
        # コスト計算部分の変更例
        cost_weights = {
            "travel": weights.travel,
            "age": weights.age,
            "frequency": weights.frequency,
            "qualification": weights.qualification,
            "recency": weights.past_assignment_recency,
        }
        
        # 制約構築部分の変更例 (allow_under_assignment を直接使用)
        if allow_under_assignment:
            # ...
            if weights.assignment_shortage > 0:
                # ...
        
        # 目的関数構築部分の変更例
        if weights.lecturer_concentration > 0:
            # ...

        # --- 求解 ---
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        solver.parameters.num_search_workers = os.cpu_count() or 1
        solver.parameters.max_time_in_seconds = solver_params.max_search_seconds

        solver.log_callback = lambda msg: logger.info(f"[OR-Tools Solver] {msg.strip()}")
        status_code = solver.Solve(model)
        
        # ... (結果の抽出と返却ロジックはほぼ同じ) ...
        # ただし、all_courses, all_lecturers を返すために、元のリストを再構築するか、
        # map.values() をリスト化して返す
        
        return SolverOutput(
            # ...
            all_courses=list(courses_map.values()),
            all_lecturers=list(lecturers_map.values()),
            # ...
        )

    finally:
        flush_log_buffer()

# (注意: 上記は主要な変更点を示すためのスケルトンコードです。
#  完全な実装には、全ての `_dict` を `_map` に、`weight_*` を `weights.*` に置換する必要があります)