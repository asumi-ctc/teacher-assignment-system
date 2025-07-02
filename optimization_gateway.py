# optimization_gateway.py

import logging
import datetime
import os
import multiprocessing
from multiprocessing.connection import Connection
from utils.logging_config import setup_logging
from utils.error_definitions import InvalidInputError  # <<< 変更: 新しいエラー定義をインポート
import optimization_solver
from typing import List, Dict, Any, Tuple, Set, TypedDict, Optional, Union

# --- [削除] ここにあった InvalidInputError クラス定義を削除 ---

logger = logging.getLogger(__name__)

class OptimizationResult(TypedDict):
    status: str
    message: str
    solution_status: str
    objective_value: Optional[float]
    assignments_df: List[Dict[str, Union[str, int, float, None]]]
    lecturer_course_counts: Dict[str, int]
    course_assignment_counts: Dict[str, int]
    course_remaining_capacity: Dict[str, int]
    raw_solver_status_code: int


# --- バリデーションヘルパー関数 ---
def _validate_date_string(date_str: Any, context: str) -> None:
    """日付文字列が 'YYYY-MM-DD' 形式か検証する"""
    if not isinstance(date_str, str):
        raise InvalidInputError(f"{context}: 日付は文字列である必要がありますが、型 '{type(date_str).__name__}' を受け取りました。値: {date_str}")
    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise InvalidInputError(f"{context}: 日付形式が無効です。'{date_str}' は 'YYYY-MM-DD' 形式ではありません。")

# --- 各データセットのバリデーション関数 ---
# (以下、関数の内部ロジックは変更なし)

def _validate_classrooms(classrooms_data: List[Dict[str, Any]]) -> Set[str]:
    """教室データのバリデーションを行い、有効な教室IDのセットを返す"""
    if not isinstance(classrooms_data, list):
        raise InvalidInputError(f"教室データはリストである必要がありますが、型 '{type(classrooms_data).__name__}' を受け取りました。")
    if not classrooms_data:
        raise InvalidInputError("教室データが空です。少なくとも1つの教室が必要です。")

    validated_ids = set()
    for i, classroom in enumerate(classrooms_data):
        context = f"教室データ[インデックス:{i}]"
        if not isinstance(classroom, dict):
            raise InvalidInputError(f"{context}: 各教室は辞書である必要がありますが、型 '{type(classroom).__name__}' を受け取りました。")

        for key in ["id", "location"]:
            if key not in classroom:
                raise InvalidInputError(f"{context}: 必須フィールド '{key}' がありません。")

        classroom_id = classroom.get("id", f"不明(インデックス:{i})")
        if not isinstance(classroom_id, str) or not classroom_id:
            raise InvalidInputError(f"{context}: 'id' は空でない文字列である必要がありますが、'{classroom_id}' を受け取りました。")
        if classroom_id in validated_ids:
            raise InvalidInputError(f"{context}: 教室ID '{classroom_id}' が重複しています。IDは一意である必要があります。")
        
        if not isinstance(classroom["location"], str) or not classroom["location"]:
            raise InvalidInputError(f"{context} (ID:{classroom_id}): 'location' は空でない文字列である必要があります。")

        validated_ids.add(classroom_id)
    
    return validated_ids

def _validate_lecturers(lecturers_data: List[Dict[str, Any]], valid_classroom_ids: Set[str]) -> None:
    """講師データのバリデーションを行う"""
    if not isinstance(lecturers_data, list):
        raise InvalidInputError(f"講師データはリストである必要がありますが、型 '{type(lecturers_data).__name__}' を受け取りました。")
    if not lecturers_data:
        raise InvalidInputError("講師データが空です。少なくとも1人の講師が必要です。")

    validated_ids = set()
    for i, lecturer in enumerate(lecturers_data):
        context = f"講師データ[インデックス:{i}]"
        if not isinstance(lecturer, dict):
            raise InvalidInputError(f"{context}: 各講師は辞書である必要がありますが、型 '{type(lecturer).__name__}' を受け取りました。")

        required_keys = ["id", "name", "age", "home_classroom_id", "qualification_general_rank", "availability", "past_assignments"]
        for key in required_keys:
            if key not in lecturer:
                raise InvalidInputError(f"{context}: 必須フィールド '{key}' がありません。")

        lecturer_id = lecturer.get("id", f"不明(インデックス:{i})")
        context = f"講師データ (ID:{lecturer_id})"

        if not isinstance(lecturer_id, str) or not lecturer_id:
            raise InvalidInputError(f"{context}: 'id' は空でない文字列である必要があります。")
        if lecturer_id in validated_ids:
            raise InvalidInputError(f"{context}: 講師ID '{lecturer_id}' が重複しています。")
        validated_ids.add(lecturer_id)

        if not isinstance(lecturer["name"], str) or not lecturer["name"]:
            raise InvalidInputError(f"{context}: 'name' は空でない文字列である必要があります。")
        
        age = lecturer["age"]
        if not isinstance(age, (int, float)) or not (18 <= age <= 100):
            raise InvalidInputError(f"{context}: 'age' は18から100の範囲の数値である必要がありますが、'{age}' を受け取りました。")

        home_classroom_id = lecturer["home_classroom_id"]
        if home_classroom_id not in valid_classroom_ids:
            raise InvalidInputError(f"{context}: 'home_classroom_id' ('{home_classroom_id}') が既存の教室IDに存在しません。")

        q_general_rank = lecturer["qualification_general_rank"]
        if not isinstance(q_general_rank, int) or not (1 <= q_general_rank <= 5):
            raise InvalidInputError(f"{context}: 'qualification_general_rank' は1から5の範囲の整数である必要がありますが、'{q_general_rank}' を受け取りました。")

        if "qualification_special_rank" in lecturer and lecturer["qualification_special_rank"] is not None:
            q_special_rank = lecturer["qualification_special_rank"]
            if not isinstance(q_special_rank, int) or not (1 <= q_special_rank <= 5):
                raise InvalidInputError(f"{context}: 'qualification_special_rank' は1から5の範囲の整数である必要がありますが、'{q_special_rank}' を受け取りました。")

        if not isinstance(lecturer["availability"], list):
            raise InvalidInputError(f"{context}: 'availability' はリストである必要があります。")
        for date_str in lecturer["availability"]:
            _validate_date_string(date_str, f"{context} の 'availability' 内の日付")

        if not isinstance(lecturer["past_assignments"], list):
            raise InvalidInputError(f"{context}: 'past_assignments' はリストである必要があります。")
        for pa_idx, pa in enumerate(lecturer["past_assignments"]):
            pa_context = f"{context} の 'past_assignments' [インデックス:{pa_idx}]"
            if not isinstance(pa, dict):
                raise InvalidInputError(f"{pa_context}: 各過去の割り当ては辞書である必要があります。")
            for key in ["classroom_id", "date"]:
                if key not in pa:
                    raise InvalidInputError(f"{pa_context}: 必須フィールド '{key}' がありません。")
            if pa["classroom_id"] not in valid_classroom_ids:
                raise InvalidInputError(f"{pa_context}: 'classroom_id' ('{pa['classroom_id']}') が既存の教室IDに存在しません。")
            _validate_date_string(pa["date"], f"{pa_context} の 'date'")

def _validate_courses(courses_data: List[Dict[str, Any]], valid_classroom_ids: Set[str]) -> None:
    """講座データのバリデーションを行う"""
    if not isinstance(courses_data, list):
        raise InvalidInputError(f"講座データはリストである必要がありますが、型 '{type(courses_data).__name__}' を受け取りました。")
    if not courses_data:
        raise InvalidInputError("講座データが空です。少なくとも1つの講座が必要です。")

    validated_ids = set()
    for i, course in enumerate(courses_data):
        context = f"講座データ[インデックス:{i}]"
        if not isinstance(course, dict):
            raise InvalidInputError(f"{context}: 各講座は辞書である必要がありますが、型 '{type(course).__name__}' を受け取りました。")

        required_keys = ["id", "name", "classroom_id", "course_type", "rank", "schedule"]
        for key in required_keys:
            if key not in course:
                raise InvalidInputError(f"{context}: 必須フィールド '{key}' がありません。")

        course_id = course.get("id", f"不明(インデックス:{i})")
        context = f"講座データ (ID:{course_id})"

        if not isinstance(course_id, str) or not course_id:
            raise InvalidInputError(f"{context}: 'id' は空でない文字列である必要があります。")
        if course_id in validated_ids:
            raise InvalidInputError(f"{context}: 講座ID '{course_id}' が重複しています。")
        validated_ids.add(course_id)

        if not isinstance(course["name"], str) or not course["name"]:
            raise InvalidInputError(f"{context}: 'name' は空でない文字列である必要があります。")
        
        if course["classroom_id"] not in valid_classroom_ids:
            raise InvalidInputError(f"{context}: 'classroom_id' ('{course['classroom_id']}') が既存の教室IDに存在しません。")

        course_type = course["course_type"]
        if course_type not in ["general", "special"]:
            raise InvalidInputError(f"{context}: 'course_type' は 'general' または 'special' である必要がありますが、'{course_type}' を受け取りました。")

        rank = course["rank"]
        if not isinstance(rank, int) or not (1 <= rank <= 5):
            raise InvalidInputError(f"{context}: 'rank' は1から5の範囲の整数である必要がありますが、'{rank}' を受け取りました。")

        _validate_date_string(course["schedule"], f"{context} の 'schedule'")

def _validate_travel_costs(travel_costs_matrix: Dict[Tuple[str, str], int], valid_classroom_ids: Set[str]) -> None:
    """移動コストデータのバリデーションを行う"""
    if not isinstance(travel_costs_matrix, dict):
        raise InvalidInputError(f"移動コストデータは辞書である必要がありますが、型 '{type(travel_costs_matrix).__name__}' を受け取りました。")

    if not travel_costs_matrix:
        logger.warning("移動コストデータが空です。コスト計算はデフォルト値にフォールバックする可能性があります。")
        return

    for key, value in travel_costs_matrix.items():
        context = f"移動コストデータ (キー:{key})"
        if not isinstance(key, tuple) or len(key) != 2:
            raise InvalidInputError(f"{context}: キーは2つの要素を持つタプルである必要があります。")
        
        from_id, to_id = key
        if not isinstance(from_id, str) or not isinstance(to_id, str):
            raise InvalidInputError(f"{context}: キーの要素は両方とも文字列である必要があります。")
        
        if from_id not in valid_classroom_ids:
            raise InvalidInputError(f"{context}: 出発教室ID '{from_id}' が既存の教室IDに存在しません。")
        if to_id not in valid_classroom_ids:
            raise InvalidInputError(f"{context}: 到着教室ID '{to_id}' が既存の教室IDに存在しません。")

        if not isinstance(value, (int, float)) or value < 0:
            raise InvalidInputError(f"{context}: 値は0以上の数値である必要がありますが、'{value}' を受け取りました。")


# --- メインアダプター関数 ---

def adapt_data_for_engine(
    lecturers_data: List[Dict[str, Any]],
    courses_data: List[Dict[str, Any]],
    classrooms_data: List[Dict[str, Any]],
    travel_costs_matrix: Dict[Tuple[str, str], int],
    **kwargs
) -> Dict[str, Any]:
    """
    最適化エンジンへの入力データを準備し、その過程で厳格なバリデーションを実行する。
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info("ステップ1/4: 教室データを検証中...")
        valid_classroom_ids = _validate_classrooms(classrooms_data)
        logger.info(f"教室データの検証が完了しました。{len(valid_classroom_ids)}件の有効な教室を確認しました。")

        logger.info("ステップ2/4: 講師データを検証中...")
        _validate_lecturers(lecturers_data, valid_classroom_ids)
        logger.info(f"講師データの検証が完了しました。{len(lecturers_data)}件の講師データは有効です。")

        logger.info("ステップ3/4: 講座データを検証中...")
        _validate_courses(courses_data, valid_classroom_ids)
        logger.info(f"講座データの検証が完了しました。{len(courses_data)}件の講座データは有効です。")

        logger.info("ステップ4/4: 移動コストデータを検証中...")
        _validate_travel_costs(travel_costs_matrix, valid_classroom_ids)
        logger.info(f"移動コストデータの検証が完了しました。{len(travel_costs_matrix)}件のエントリは有効です。")

        logger.info("全ての入力データのバリデーションが正常に完了しました。")

        return {
            "lecturers_data": lecturers_data,
            "courses_data": courses_data,
            "classrooms_data": classrooms_data,
            "travel_costs_matrix": travel_costs_matrix,
        }
    except InvalidInputError as e:
        logger.error(f"入力データのバリデーションに失敗しました: {e}", exc_info=True)
        raise

# --- 最適化実行ラッパー ---
# (以下、変更なし)
RETRY_LIMIT = 2
PROCESS_TIMEOUT_SECONDS = 90

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

def run_optimization_with_monitoring(
    lecturers_data: List[Dict[str, Any]],
    courses_data: List[Dict[str, Any]],
    classrooms_data: List[Dict[str, Any]],
    travel_costs_matrix: Dict[Tuple[str, str], int],
    **kwargs: Any
) -> OptimizationResult:
    """
    最適化エンジンを監視付きの別プロセスで実行し、タイムアウトや再試行を管理する。
    """
    logger = logging.getLogger(__name__)

    solver_args = {
        "lecturers_data": lecturers_data,
        "courses_data": courses_data,
        "classrooms_data": classrooms_data,
        "travel_costs_matrix": travel_costs_matrix,
        **kwargs
    }

    for attempt in range(RETRY_LIMIT):
        logger.info(f"最適化プロセスを開始します。(試行 {attempt + 1}/{RETRY_LIMIT})")
        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)

        process = multiprocessing.Process(
            target=_run_solver_process,
            args=(child_conn, solver_args)
        )

        solver_output = None
        try:
            process.start()
            if parent_conn.poll(PROCESS_TIMEOUT_SECONDS):
                result = parent_conn.recv()
                if isinstance(result, Exception):
                    raise InvalidInputError(f"最適化プロセスでエラーが発生しました: {result}") from result
                solver_output = result
                logger.info("最適化プロセスから結果を正常に受信しました。")
            else:
                logger.error(f"最適化プロセスがタイムアウトしました ({PROCESS_TIMEOUT_SECONDS}秒)。")

        except (IOError, EOFError) as e:
            logger.error(f"プロセス間通信中にエラーが発生しました: {e}", exc_info=True)
        finally:
            process.join(timeout=5) 
            if process.is_alive():
                logger.error("最適化プロセスが応答しません。強制終了します。")
                process.terminate()
                process.join()
            parent_conn.close()

        if solver_output is not None:
            logger.info("最適化プロセスが正常に完了しました。")
            logger.info("最適化結果を整形します...")

            all_lecturers_dict = {l['id']: l for l in solver_output['all_lecturers']}
            all_courses_dict = {c['id']: c for c in solver_output['all_courses']}
            all_classrooms_dict = {c['id']: c for c in solver_args['classrooms_data']}

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

            lecturer_course_counts: Dict[str, int] = {}
            for assignment in processed_assignments:
                lecturer_id = assignment['講師ID']
                lecturer_course_counts[lecturer_id] = lecturer_course_counts.get(lecturer_id, 0) + 1

            course_assignment_counts: Dict[str, int] = {}
            course_remaining_capacity: Dict[str, int] = {}
            TARGET_PREFECTURES_FOR_TWO_LECTURERS = ["東京都", "愛知県", "大阪府"]
            for course_item in solver_output['all_courses']:
                course_id = course_item['id']
                assigned_count = sum(1 for a in processed_assignments if a['講座ID'] == course_id)
                course_assignment_counts[course_id] = assigned_count
                
                classroom_id = course_item.get('classroom_id')
                location = all_classrooms_dict.get(classroom_id, {}).get('location')
                capacity = 2 if location in TARGET_PREFECTURES_FOR_TWO_LECTURERS else 1
                course_remaining_capacity[course_id] = capacity - assigned_count

            final_result: OptimizationResult = {
                "status": "成功",
                "message": "最適化処理が正常に完了しました。",
                "solution_status": solver_output["solution_status_str"],
                "objective_value": solver_output["objective_value"],
                "assignments_df": processed_assignments,
                "lecturer_course_counts": lecturer_course_counts,
                "course_assignment_counts": course_assignment_counts,
                "course_remaining_capacity": course_remaining_capacity,
                "raw_solver_status_code": solver_output["solver_raw_status_code"]
            }
            logger.info("最適化結果の整形が完了しました。")
            return final_result

        if attempt < RETRY_LIMIT - 1:
            logger.info("再試行します...")
            continue

    raise InvalidInputError(
        "最適化処理が複数回の試行でも設定時間内に完了しませんでした。"
        "問題が解決しない場合は、入力データを簡素化するか、モデル設定を見直してください。"
    )