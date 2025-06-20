# data_adapter.py
from typing import List, Dict, Any, Tuple

def adapt_data_for_engine(
    lecturers_data: List[Dict[str, Any]],
    courses_data: List[Dict[str, Any]],
    classrooms_data: List[Dict[str, Any]],
    travel_costs_matrix: Dict[Tuple[str, str], int],
    **kwargs # 将来的な拡張のために残す
) -> Dict[str, Any]:
    """
    最適化エンジンへの入力データを準備する。
    今回はデータをそのまま辞書に格納して返す。
    """
    return {
        "lecturers_data": lecturers_data,
        "courses_data": courses_data,
        "classrooms_data": classrooms_data,
        "travel_costs_matrix": travel_costs_matrix,
    }