# ==============================================================================
# 3. utils/types.py (型定義)
# ==============================================================================
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, TypedDict, Optional, Union
import datetime

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

SolverAssignment = TypedDict(
    "SolverAssignment",
    {
        "講師ID": str,
        "講座ID": str,
        "算出コスト(x100)": int,
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
    all_courses: List[Dict[str, Any]]
    all_lecturers: List[Dict[str, Any]]
    solver_raw_status_code: int

AssignmentResultRow = TypedDict(
    "AssignmentResultRow",
    {
        "講師ID": str,
        "講座ID": str,
        "算出コスト(x100)": int,
        "年齢コスト(元)": float,
        "頻度コスト(元)": float,
        "資格コスト(元)": float,
        "当該教室最終割当日からの日数": int,
        "今回の割り当て回数": int,
        "連続ペア割当": str,
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
