# utils/types.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, TypedDict, Optional
import datetime

# --- 1. 設定オブジェクト関連の型定義 ---

@dataclass
class OptimizationWeights:
    """目的関数の重み"""
    past_assignment_recency: float = 0.5
    qualification: float = 0.5
    travel: float = 0.5
    age: float = 0.5
    frequency: float = 0.5
    assignment_shortage: float = 0.5
    lecturer_concentration: float = 0.5
    consecutive_assignment: float = 0.5

@dataclass
class SolverParameters:
    """ソルバーの振る舞いを制御する全パラメータ"""
    weights: OptimizationWeights = field(default_factory=OptimizationWeights)
    allow_under_assignment: bool = True
    max_search_seconds: int = 90

@dataclass
class OptimizationInput:
    """最適化エンジンへの全入力データ"""
    lecturers_data: List[Dict[str, Any]]
    courses_data: List[Dict[str, Any]]
    classrooms_data: List[Dict[str, Any]]
    travel_costs_matrix: Dict[Tuple[str, str], int]
    solver_params: SolverParameters
    today_date: datetime.date
    fixed_assignments: Optional[List[Tuple[str, str]]] = None
    forced_unassignments: Optional[List[Tuple[str, str]]] = None

# --- 2. 出力関連の型定義 ---

class SolverOutput(TypedDict):
    """ソルバーからの直接の出力"""
    solution_status_str: str
    objective_value: Optional[float]
    assignments: List[dict]
    all_courses: List[dict]
    all_lecturers: List[dict]
    solver_raw_status_code: int

class OptimizationResult(TypedDict):
    """ゲートウェイからの最終的な実行結果"""
    status: str
    message: str
    solution_status: str
    objective_value: Optional[float]
    assignments_df: List[Dict[str, Any]]
    lecturer_course_counts: Dict[str, int]
    course_assignment_counts: Dict[str, int]
    course_remaining_capacity: Dict[str, int]
    raw_solver_status_code: int