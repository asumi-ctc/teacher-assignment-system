# optimization_gateway.py

import logging
import datetime
import os
import multiprocessing
from multiprocessing.connection import Connection
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set, TypedDict, Optional

from utils.logging_config import setup_logging
from utils.error_definitions import InvalidInputError
import optimization_solver

logger = logging.getLogger(__name__)

# --- 1. 型定義 (入力と出力、設定オブジェクト) ---

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

class OptimizationResult(TypedDict):
    """最適化の実行結果"""
    status: str
    message: str
    solution_status: str
    objective_value: Optional[float]
    assignments_df: List[Dict[str, Any]]
    lecturer_course_counts: Dict[str, int]
    course_assignment_counts: Dict[str, int]
    course_remaining_capacity: Dict[str, int]
    raw_solver_status_code: int

# --- 2. データ検証・前処理ヘルパー関数 ---

def _validate_date_string(date_str: Any, context: str) -> None:
    if not isinstance(date_str, str):
        raise InvalidInputError(f"{context}: 日付は文字列である必要がありますが、型 '{type(date_str).__name__}' を受け取りました。値: {date_str}")
    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise InvalidInputError(f"{context}: 日付形式が無効です。'{date_str}' は 'YYYY-MM-DD' 形式ではありません。")

def _validate_classrooms(classrooms_data: List[Dict[str, Any]]) -> Set[str]:
    if not isinstance(classrooms_data, list):
        raise InvalidInputError(f"教室データはリストである必要がありますが、型 '{type(classrooms_data).__name__}' を受け取りました。")
    if not classrooms_data:
        raise InvalidInputError("教室データが空です。")
    validated_ids = set()
    for i, classroom in enumerate(classrooms_data):
        context = f"教室データ[インデックス:{i}]"
        if not isinstance(classroom, dict):
            raise InvalidInputError(f"{context}: 各教室は辞書形式でなければなりません。")
        for key in ["id", "location"]:
            if key not in classroom:
                raise InvalidInputError(f"{context}: 必須フィールド '{key}' がありません。")
        classroom_id = classroom.get("id")
        if not isinstance(classroom_id, str) or not classroom_id:
            raise InvalidInputError(f"{context}: 'id' は空でない文字列である必要があります。")
        if classroom_id in validated_ids:
            raise InvalidInputError(f"{context}: 教室ID '{classroom_id}' が重複しています。")
        validated_ids.add(classroom_id)
    return validated_ids

def _validate_lecturers(lecturers_data: List[Dict[str, Any]], valid_classroom_ids: Set[str]) -> None:
    if not isinstance(lecturers_data, list):
        raise InvalidInputError(f"講師データはリストである必要がありますが、型 '{type(lecturers_data).__name__}' を受け取りました。")
    if not lecturers_data:
        raise InvalidInputError("講師データが空です。")
    validated_ids = set()
    for i, lecturer in enumerate(lecturers_data):
        context = f"講師データ[インデックス:{i}]"
        if not isinstance(lecturer, dict):
            raise InvalidInputError(f"{context}: 各講師は辞書形式でなければなりません。")
        required_keys = ["id", "name", "age", "home_classroom_id", "qualification_general_rank", "availability", "past_assignments"]
        for key in required_keys:
            if key not in lecturer:
                raise InvalidInputError(f"{context} (ID:{lecturer.get('id', '不明')}): 必須フィールド '{key}' がありません。")
        lecturer_id = lecturer.get("id")
        if not isinstance(lecturer_id, str) or not lecturer_id:
            raise InvalidInputError(f"{context}: 'id' は空でない文字列である必要があります。")
        if lecturer_id in validated_ids:
            raise InvalidInputError(f"{context}: 講師ID '{lecturer_id}' が重複しています。")
        validated_ids.add(lecturer_id)
        if lecturer["home_classroom_id"] not in valid_classroom_ids:
            raise InvalidInputError(f"{context} (ID:{lecturer_id}): 'home_classroom_id' ('{lecturer['home_classroom_id']}') が教室マスタに存在しません。")

def _validate_courses(courses_data: List[Dict[str, Any]], valid_classroom_ids: Set[str]) -> None:
    if not isinstance(courses_data, list):
        raise InvalidInputError(f"講座データはリストである必要がありますが、型 '{type(courses_data).__name__}' を受け取りました。")
    if not courses_data:
        raise InvalidInputError("講座データが空です。")
    validated_ids = set()
    for i, course in enumerate(courses_data):
        context = f"講座データ[インデックス:{i}]"
        if not isinstance(course, dict):
            raise InvalidInputError(f"{context}: 各講座は辞書形式でなければなりません。")
        required_keys = ["id", "name", "classroom_id", "course_type", "rank", "schedule"]
        for key in required_keys:
            if key not in course:
                raise InvalidInputError(f"{context} (ID:{course.get('id', '不明')}): 必須フィールド '{key}' がありません。")
        course_id = course.get("id")
        if not isinstance(course_id, str) or not course_id:
            raise InvalidInputError(f"{context}: 'id' は空でない文字列である必要があります。")
        if course_id in validated_ids:
            raise InvalidInputError(f"{context}: 講座ID '{course_id}' が重複しています。")
        validated_ids.add(course_id)
        if course["classroom_id"] not in valid_classroom_ids:
            raise InvalidInputError(f"{context} (ID:{course_id}): 'classroom_id' ('{course['classroom_id']}') が教室マスタに存在しません。")

def _validate_travel_costs(travel_costs_matrix: Dict[Tuple[str, str], int], valid_classroom_ids: Set[str]) -> None:
    if not isinstance(travel_costs_matrix, dict):
        raise InvalidInputError(f"移動コストデータは辞書である必要がありますが、型 '{type(travel_costs_matrix).__name__}' を受け取りました。")
    for key, value in travel_costs_matrix.items():
        if not (isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], str) and isinstance(key[1], str)):
            raise InvalidInputError(f"移動コストのキーが不正です: {key}")
        if key[0] not in valid_classroom_ids or key[1] not in valid_classroom_ids:
            raise InvalidInputError(f"移動コストのキーに未知の教室IDが含まれています: {key}")
        if not isinstance(value, (int, float)) or value < 0:
            raise InvalidInputError(f"移動コストの値が不正です (キー: {key}, 値: {value})")

def _validate_and_preprocess_data(input_data: OptimizationInput) -> Dict[str, Any]:
    logger.info("入力データの検証と前処理を開始します...")
    valid_classroom_ids = _validate_classrooms(input_data.classrooms_data)
    _validate_lecturers(input_data.lecturers_data, valid_classroom_ids)
    _validate_courses(input_data.courses_data, valid_classroom_ids)
    _validate_travel_costs(input_data.travel_costs_matrix, valid_classroom_ids)
    logger.info("データバリデーションが正常に完了しました。")

    logger.info("データ前処理（マップ形式へ変換）を実行中...")
    preprocessed_data = {
        "lecturers_map": {d['id']: d for d in input_data.lecturers_data},
        "courses_map": {d['id']: d for d in input_data.courses_data},
        "classrooms_map": {d['id']: d for d in input_data.classrooms_data},
        "travel_costs_matrix": input_data.travel_costs_matrix,
        "solver_params": input_data.solver_params,
        "today_date": input_data.today_date,
        "fixed_assignments": input_data.fixed_assignments,
        "forced_unassignments": input_data.forced_unassignments,
    }
    logger.info("データ前処理が正常に完了しました。")
    return preprocessed_data

# --- 3. 最適化実行プロセス ---

def _run_solver_process(conn: Connection, solver_args: Dict[str, Any]):
    try:
        setup_logging(target_loggers=['optimization_solver'])
        result = optimization_solver.solve_assignment(**solver_args)
        conn.send(result)
    except Exception as e:
        child_logger = logging.getLogger('optimization_solver')
        child_logger.error(f"最適化子プロセスで致命的なエラーが発生: {e}", exc_info=True)
        conn.send(e)
    finally:
        conn.close()

def _run_optimization_with_monitoring(solver_args: Dict[str, Any], timeout_seconds: int) -> optimization_solver.SolverOutput:
    logger.info(f"最適化プロセスを開始します。(タイムアウト: {timeout_seconds}秒)")
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(target=_run_solver_process, args=(child_conn, solver_args))
    solver_output = None
    try:
        process.start()
        if parent_conn.poll(timeout_seconds):
            result = parent_conn.recv()
            if isinstance(result, Exception):
                raise InvalidInputError(f"最適化プロセスでエラーが発生しました: {result}") from result
            solver_output = result
            logger.info("最適化プロセスから結果を正常に受信しました。")
        else:
            raise TimeoutError(f"最適化処理が設定時間（{timeout_seconds}秒）内に完了しませんでした。")
    finally:
        if process.is_alive():
            logger.warning("最適化プロセスを強制終了します。")
            process.terminate()
            process.join(timeout=5)
        parent_conn.close()
        child_conn.close()
    if solver_output is None:
        raise Exception("最適化プロセスから予期せぬ空の結果が返されました。")
    return solver_output

# --- 4. 結果の整形 ---

def _format_result(solver_output: optimization_solver.SolverOutput) -> OptimizationResult:
    logger.info("最適化結果を整形します...")
    
    # 元のデータは solver_output 内に含まれるようになったので、引数から削除
    all_lecturers_dict = {l['id']: l for l in solver_output['all_lecturers']}
    all_courses_dict = {c['id']: c for c in solver_output['all_courses']}
    # classrooms_map は solver に渡しているが、結果整形では使わないので不要なら削除可
    
    processed_assignments = []
    for assignment in solver_output['assignments']:
        lecturer_id = assignment['講師ID']
        course_id = assignment['講座ID']
        lecturer = all_lecturers_dict.get(lecturer_id, {})
        course = all_courses_dict.get(course_id, {})

        processed_assignments.append({
            **assignment,
            "講師名": lecturer.get("name", "不明"),
            "講座名": course.get("name", "不明"),
            "教室ID": course.get("classroom_id", "不明"),
            "スケジュール": course.get("schedule", "不明"),
            # location は course や classroom から取得する必要があるが、ここでは省略
            "教室名": "不明", 
            "講師一般ランク": lecturer.get("qualification_general_rank"),
            "講師特別ランク": lecturer.get("qualification_special_rank", "なし"),
            "講座タイプ": course.get("course_type"),
            "講座ランク": course.get("rank"),
        })

    # (集計情報の計算ロジック)
    lecturer_counts = {lect_id: 0 for lect_id in all_lecturers_dict}
    for assign in processed_assignments:
        lecturer_counts[assign['講師ID']] += 1

    final_result: OptimizationResult = {
        "status": "成功",
        "message": "最適化処理が正常に完了しました。",
        "solution_status": solver_output["solution_status_str"],
        "objective_value": solver_output["objective_value"],
        "assignments_df": processed_assignments,
        "lecturer_course_counts": lecturer_counts,
        "course_assignment_counts": {}, # TODO: 実装
        "course_remaining_capacity": {}, # TODO: 実装
        "raw_solver_status_code": solver_output["raw_solver_status_code"]
    }
    logger.info("最適化結果の整形が完了しました。")
    return final_result

# --- 5. 公開インターフェース (エントリーポイント) ---

def execute_optimization(input_data: OptimizationInput) -> OptimizationResult:
    try:
        solver_args = _validate_and_preprocess_data(input_data)
        timeout = input_data.solver_params.max_search_seconds
        solver_output = _run_optimization_with_monitoring(solver_args, timeout)
        return _format_result(solver_output)
    except (InvalidInputError, TimeoutError) as e:
        logger.error(f"最適化の事前処理または実行中にエラーが発生: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)
        raise InvalidInputError(f"予期せぬ内部エラー: {e}") from e