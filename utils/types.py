# utils/types.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, TypedDict, Optional, Union
import datetime

# --- 0. データ構造の型定義 ---
# データソース（DBやCSVなど）から読み込まれるデータの型を定義します。
# これにより、前処理や最適化エンジンへの入力の型安全性が向上します。
# 注: 日付は前処理段階でdatetime.dateオブジェクトに変換されることを想定しています。

class PastAssignment(TypedDict):
    """過去の担当履歴"""
    classroom_id: str
    date: datetime.date

class LecturerData(TypedDict):
    """講師1人分のデータ構造"""
    id: str
    age: int
    home_classroom_id: str
    qualification_general_rank: int
    qualification_special_rank: Optional[int]
    availability: List[datetime.date]
    past_assignments: List[PastAssignment]

class CourseData(TypedDict):
    """コース1つ分のデータ構造"""
    id: str
    classroom_id: str
    course_type: str
    rank: int
    schedule: datetime.date

class ClassroomData(TypedDict):
    """教室1つ分のデータ構造"""
    id: str
    location: str

# --- 1. 設定オブジェクト関連の型定義 ---

@dataclass
class OptimizationWeights:
    """目的関数の重み"""
    past_assignment_recency: float = 0.5
    qualification: float = 0.5
    travel: float = 0.5
    age: float = 0.5
    frequency: float = 0.5
    lecturer_concentration: float = 0.5
    consecutive_assignment: float = 0.5

@dataclass
class SolverParameters:
    """ソルバーの振る舞いを制御する全パラメータ"""
    weights: OptimizationWeights = field(default_factory=OptimizationWeights)
    max_search_seconds: int = 90

@dataclass
class OptimizationInput:
    """最適化エンジンへの全入力データ"""
    lecturers_data: List[LecturerData]
    courses_data: List[CourseData]
    classrooms_data: List[ClassroomData]
    travel_costs_matrix: Dict[Tuple[str, str], int]
    solver_params: SolverParameters
    today_date: datetime.date
    fixed_assignments: Optional[List[Tuple[str, str]]] = None
    forced_unassignments: Optional[List[Tuple[str, str]]] = None

# --- 2. 出力関連の型定義 ---

# ソルバーからの割り当て結果1件分のデータ
SolverAssignment = TypedDict(
    "SolverAssignment",
    {
        "講師ID": str,
        "講座ID": str,
        "算出コスト(x100)": int,
        "移動コスト(元)": int,
        "年齢コスト(元)": float,
        "頻度コスト(元)": float,
        "資格コスト(元)": float,
        "当該教室最終割当日からの日数": int,
        "今回の割り当て回数": int,
        "連続ペア割当": str,
    },
)

class SolverOutput(TypedDict):
    """ソルバーからの直接の出力"""
    solution_status_str: str
    objective_value: Optional[float]
    assignments: List[SolverAssignment]
    # all_coursesとall_lecturersはソルバー内で加工される可能性があるため、汎用的な型を使用します。
    all_courses: List[Dict[str, Any]]
    all_lecturers: List[Dict[str, Any]]
    solver_raw_status_code: int

# 最終的な割り当て結果の1行（DataFrame化される前のデータ）。
# ソルバーからの情報とゲートウェイで追加される情報をすべて含む。
AssignmentResultRow = TypedDict(
    "AssignmentResultRow",
    {
        # --- From SolverAssignment ---
        "講師ID": str,
        "講座ID": str,
        "算出コスト(x100)": int,
        "移動コスト(元)": int,
        "年齢コスト(元)": float,
        "頻度コスト(元)": float,
        "資格コスト(元)": float,
        "当該教室最終割当日からの日数": int,
        "今回の割り当て回数": int,
        "連続ペア割当": str,
        # --- Added in Gateway ---
        "講師名": str,
        "講座名": str,
        "教室ID": str,
        "スケジュール": datetime.date,
        "教室名": str,
        "講師一般ランク": Optional[int],
        "講師特別ランク": Optional[Union[int, str]],
        "講座タイプ": Optional[str],
        "講座ランク": Optional[int],
    },
)

class OptimizationResult(TypedDict):
    """ゲートウェイからの最終的な実行結果"""
    status: str
    message: str
    solution_status: str
    objective_value: Optional[float]
    assignments_df: List[AssignmentResultRow]
    lecturer_course_counts: Dict[str, int]
    course_assignment_counts: Dict[str, int]
    course_remaining_capacity: Dict[str, int]
    raw_solver_status_code: int