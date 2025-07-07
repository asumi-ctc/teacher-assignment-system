# optimization_gateway.py

import logging
import datetime
import os
import multiprocessing
from multiprocessing.connection import Connection
from dataclasses import dataclass, field # dataclasses を使用
from typing import List, Dict, Any, Tuple, Set, TypedDict, Optional, Union

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
    # 将来の制約フラグなどもここに追加

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


# --- 2. データ検証・前処理 (バリデーションと前処理) ---

def _validate_and_preprocess_data(input_data: OptimizationInput) -> Dict[str, Any]:
    """
    入力データを検証し、ソルバーが計算しやすい形式に前処理する。
    """
    logger.info("入力データの検証と前処理を開始します...")

    # --- 検証フェーズ ---
    logger.info("ステップ1/2: データバリデーションを実行中...")
    valid_classroom_ids = _validate_classrooms(input_data.classrooms_data)
    _validate_lecturers(input_data.lecturers_data, valid_classroom_ids)
    _validate_courses(input_data.courses_data, valid_classroom_ids)
    _validate_travel_costs(input_data.travel_costs_matrix, valid_classroom_ids)
    logger.info("データバリデーションが正常に完了しました。")

    # --- 前処理フェーズ ---
    logger.info("ステップ2/2: データ前処理を実行中...")
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

# (ここに _validate_classrooms, _validate_lecturers などのヘルパー関数群を配置)
# ... (前回のコードから変更なしのため、ここでは省略) ...


# --- 3. 最適化実行 (プロセス管理) ---

def _run_solver_process(conn: Connection, solver_args: Dict[str, Any]):
    """子プロセスで実行されるソルバー呼び出しラッパー"""
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

def _run_optimization_with_monitoring(
    solver_args: Dict[str, Any],
    timeout_seconds: int
) -> optimization_solver.SolverOutput:
    """
    最適化エンジンを監視付きの別プロセスで実行し、タイムアウトを管理する。
    """
    logger.info(f"最適化プロセスを開始します。(タイムアウト: {timeout_seconds}秒)")
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_run_solver_process,
        args=(child_conn, solver_args)
    )

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
            logger.error(f"最適化プロセスがタイムアウトしました ({timeout_seconds}秒)。")
            # タイムアウトした場合もエラーとして処理
            raise TimeoutError(f"最適化処理が設定時間（{timeout_seconds}秒）内に完了しませんでした。")
    except (IOError, EOFError) as e:
        logger.error(f"プロセス間通信中にエラーが発生しました: {e}", exc_info=True)
        raise
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


# --- 4. 結果の整形 (出力整形) ---

def _format_result(
    solver_output: optimization_solver.SolverOutput,
    all_courses: List[Dict[str, Any]],
    all_classrooms: List[Dict[str, Any]]
) -> OptimizationResult:
    """ソルバーの生の結果を、最終的なAPIレスポンス形式に整形する"""
    logger.info("最適化結果を整形します...")

    all_lecturers_dict = {l['id']: l for l in solver_output['all_lecturers']}
    all_courses_dict = {c['id']: c for c in all_courses}
    all_classrooms_dict = {c['id']: c for c in all_classrooms}

    processed_assignments = []
    for assignment in solver_output['assignments']:
        lecturer_id = assignment['講師ID']
        course_id = assignment['講座ID']
        lecturer = all_lecturers_dict.get(lecturer_id, {})
        course = all_courses_dict.get(course_id, {})
        classroom = all_classrooms_dict.get(course.get('classroom_id'), {})

        processed_assignments.append({
            **assignment,
            "講師名": lecturer.get("name", "不明"),
            "講座名": course.get("name", "不明"),
            "教室ID": course.get("classroom_id", "不明"),
            "スケジュール": course.get("schedule", "不明"),
            "教室名": classroom.get("location", "不明"),
            "講師一般ランク": lecturer.get("qualification_general_rank"),
            "講師特別ランク": lecturer.get("qualification_special_rank", "なし"),
            "講座タイプ": course.get("course_type"),
            "講座ランク": course.get("rank"),
        })
    
    # ... (集計情報の計算ロジックは変更なしのため省略) ...

    final_result: OptimizationResult = {
        "status": "成功",
        "message": "最適化処理が正常に完了しました。",
        "solution_status": solver_output["solution_status_str"],
        "objective_value": solver_output["objective_value"],
        "assignments_df": processed_assignments,
        "lecturer_course_counts": {}, # 省略
        "course_assignment_counts": {}, # 省略
        "course_remaining_capacity": {}, # 省略
        "raw_solver_status_code": solver_output["solver_raw_status_code"]
    }
    logger.info("最適化結果の整形が完了しました。")
    return final_result


# --- 5. 公開インターフェース (エントリーポイント) ---

def execute_optimization(input_data: OptimizationInput) -> OptimizationResult:
    """
    最適化を実行するための唯一の公開関数（エントリーポイント）。
    """
    try:
        # 1. 検証と前処理
        solver_args = _validate_and_preprocess_data(input_data)
        
        # 2. 監視付きで最適化を実行
        timeout = input_data.solver_params.max_search_seconds
        solver_output = _run_optimization_with_monitoring(solver_args, timeout)
        
        # 3. 結果を整形して返す
        return _format_result(
            solver_output,
            input_data.courses_data,
            input_data.classrooms_data
        )
    except (InvalidInputError, TimeoutError) as e:
        logger.error(f"最適化の事前処理または実行中にエラーが発生: {e}", exc_info=True)
        raise  # エラーをそのまま呼び出し元に再スロー
    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)
        # 予期せぬエラーも、捕捉可能なようにInvalidInputErrorでラップして再スロー
        raise InvalidInputError(f"予期せぬ内部エラー: {e}") from e

# --- ここに _validate_* ヘルパー関数群をペーストしてください ---
# ... (上記のコードスニペットからコピー) ...