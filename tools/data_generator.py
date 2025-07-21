import datetime
import random
import json
import os
import argparse
from pathlib import Path
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Any, Tuple, Optional, Callable

# --- Constants (from original app.py) ---
PREFECTURES = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県",
    "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県",
    "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県",
    "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"
]
PREFECTURE_CLASSROOM_IDS = [f"P{i+1}" for i in range(len(PREFECTURES))]

GENERAL_COURSE_LEVELS = [
    {"name_suffix": "初心", "rank": 5}, {"name_suffix": "初級", "rank": 4},
    {"name_suffix": "中級", "rank": 3}, {"name_suffix": "上級", "rank": 2},
    {"name_suffix": "プロ", "rank": 1}
]
SPECIAL_COURSE_LEVELS = [
    {"name_suffix": "初心", "rank": 5}, {"name_suffix": "初級", "rank": 4},
    {"name_suffix": "中級", "rank": 3}, {"name_suffix": "上級", "rank": 2},
    {"name_suffix": "プロ", "rank": 1}
]

# --- Data Generation Functions (extracted from app.py) ---

def generate_prefectures_data() -> Tuple[List[str], List[str]]:
    """都道府県名と対応する教室IDを生成する。"""
    return PREFECTURES, PREFECTURE_CLASSROOM_IDS

def generate_classrooms_data(prefectures: List[str], prefecture_classroom_ids: List[str]) -> List[Dict[str, str]]:
    """教室データを生成する。"""
    classrooms_data = []
    for i, pref_name in enumerate(prefectures):
        classrooms_data.append({
            "id": prefecture_classroom_ids[i],
            "location": pref_name
        })
    return classrooms_data

def generate_lecturers_data(
    prefecture_classroom_ids: List[str],
    today_date: datetime.date,
    assignment_target_month_start: datetime.date,
    assignment_target_month_end: datetime.date,
    num_lecturers: int
) -> List[Dict[str, Any]]:
    """
    講師データを生成する。
    Args:
        prefecture_classroom_ids: 教室IDのリスト。
        today_date: 現在の日付。
        assignment_target_month_start: 割り当て対象月の開始日。
        assignment_target_month_end: 割り当て対象月の終了日。
        num_lecturers: 生成する講師の数。
    Returns:
        生成された講師データのリスト。
    """
    lecturers_data = []
    # 講師の空き日を生成する期間 (対象月の前後1ヶ月)
    availability_period_start = assignment_target_month_start - relativedelta(months=1)
    availability_period_end = assignment_target_month_end + relativedelta(months=1)

    all_possible_dates_for_availability = []
    current_date_iter = availability_period_start
    while current_date_iter <= availability_period_end:
        all_possible_dates_for_availability.append(current_date_iter)
        current_date_iter += datetime.timedelta(days=1)

    for i in range(num_lecturers):
        num_available_days = random.randint(15, 45) # 対象月±1ヶ月の期間内で15～45日空いている
        if len(all_possible_dates_for_availability) >= num_available_days:
            availability = random.sample(all_possible_dates_for_availability, num_available_days)
            availability.sort()
        else:
            availability = all_possible_dates_for_availability[:]
        
        # 過去の割り当て履歴を生成 (約10件)
        num_past_assignments = random.randint(8, 12)
        past_assignments = []
        for _ in range(num_past_assignments):
            days_ago = random.randint(1, 730) # 過去2年以内のランダムな日付
            assignment_date = today_date - datetime.timedelta(days=days_ago)
            past_assignments.append({
                "classroom_id": random.choice(prefecture_classroom_ids),
                "date": assignment_date
            })
        past_assignments.sort(key=lambda x: x["date"], reverse=True)

        # 新しい資格ランク生成ロジック
        has_special_qualification = random.choice([True, False, False]) # 約1/3が特別資格持ち
        special_rank = None
        if has_special_qualification:
            special_rank = random.randint(1, 5)
            general_rank = 1 # 特別資格持ちは一般資格ランク1
        else:
            general_rank = random.randint(1, 5)

        lecturers_data.append({
            "id": f"L{i:03d}",
            "name": f"講師{i:03d}",
            "age": random.randint(22, 65),
            "home_classroom_id": "P13", # 全講師のホーム教室を東京に固定
            "qualification_general_rank": general_rank,
            "qualification_special_rank": special_rank,
            "availability": availability,
            "past_assignments": past_assignments
        })
    return lecturers_data

def generate_courses_data(
    prefectures: List[str],
    prefecture_classroom_ids: List[str],
    assignment_target_month_start: datetime.date,
    assignment_target_month_end: datetime.date,
    num_general_courses_per_sunday: int,
    generate_special_courses: bool
) -> List[Dict[str, Any]]:
    """
    講座データを生成する。
    Args:
        prefectures: 都道府県名のリスト。
        prefecture_classroom_ids: 教室IDのリスト。
        assignment_target_month_start: 割り当て対象月の開始日。
        assignment_target_month_end: 割り当て対象月の終了日。
        num_general_courses_per_sunday: 各日曜日に生成する一般講座の数。
        generate_special_courses: 特別講座を生成するかどうかのフラグ。
    Returns:
        生成された講座データのリスト。
    """
    courses_data = []
    course_counter = 1

    # 対象月の日曜日リストを作成
    sundays_in_target_month = []
    current_date_iter = assignment_target_month_start
    while current_date_iter <= assignment_target_month_end:
        if current_date_iter.weekday() == 6: # 0:月曜日, 6:日曜日
            sundays_in_target_month.append(current_date_iter)
        current_date_iter += datetime.timedelta(days=1)

    # 対象月の土曜日と平日リストを作成 (特別講座用)
    saturdays_in_target_month_for_special = []
    weekdays_in_target_month_for_special = []
    all_days_in_target_month_for_special_obj = []
    current_date_iter = assignment_target_month_start
    while current_date_iter <= assignment_target_month_end:
        all_days_in_target_month_for_special_obj.append(current_date_iter)
        if current_date_iter.weekday() == 5: # 土曜日
            saturdays_in_target_month_for_special.append(current_date_iter)
        elif current_date_iter.weekday() < 5: # 平日
            weekdays_in_target_month_for_special.append(current_date_iter)
        current_date_iter += datetime.timedelta(days=1)

    for i, pref_classroom_id in enumerate(prefecture_classroom_ids):
        pref_name = prefectures[i]

        # 一般講座の生成 (対象月の各日曜日に、ランダムに指定数開催)
        for sunday_date in sundays_in_target_month:
            selected_levels_for_general = random.sample(GENERAL_COURSE_LEVELS, min(num_general_courses_per_sunday, len(GENERAL_COURSE_LEVELS)))
            for level_info in selected_levels_for_general:
                courses_data.append({
                    "id": f"{pref_classroom_id}-GC{course_counter}",
                    "name": f"{pref_name} 一般講座 {level_info['name_suffix']} ({sunday_date.strftime('%m-%d')})",
                    "classroom_id": pref_classroom_id,
                    "course_type": "general",
                    "rank": level_info['rank'],
                    "schedule": sunday_date
                })
                course_counter += 1

        # 特別講座の生成 (対象月内に1回、土曜優先)
        if generate_special_courses:
            chosen_date_for_special_course = None
            if saturdays_in_target_month_for_special:
                chosen_date_for_special_course = random.choice(saturdays_in_target_month_for_special)
            elif weekdays_in_target_month_for_special:
                chosen_date_for_special_course = random.choice(weekdays_in_target_month_for_special)
            elif all_days_in_target_month_for_special_obj:
                chosen_date_for_special_course = random.choice(all_days_in_target_month_for_special_obj)
            
            if chosen_date_for_special_course:
                level_info_special = random.choice(SPECIAL_COURSE_LEVELS)
                courses_data.append({
                    "id": f"{pref_classroom_id}-SC{course_counter}",
                    "name": f"{pref_name} 特別講座 {level_info_special['name_suffix']} ({chosen_date_for_special_course.strftime('%m-%d')})",
                    "classroom_id": pref_classroom_id,
                    "course_type": "special",
                    "rank": level_info_special['rank'],
                    "schedule": chosen_date_for_special_course
                })
                course_counter += 1
    return courses_data

# --- Data Corruption Functions (extracted from app.py) ---
# これらの関数は、生成されたデータリストを直接変更します。

def _corrupt_duplicate_classroom_id(classrooms_data: List[Dict[str, Any]]) -> str:
    """教室データのIDを重複させる"""
    if len(classrooms_data) > 1:
        classrooms_data[1]['id'] = classrooms_data[0]['id']
        return "教室データのIDを重複させました (classrooms[1]['id'] = classrooms[0]['id'])。"
    return "教室データが少なく、IDを重複させられませんでした。"

def _corrupt_missing_classroom_location(classrooms_data: List[Dict[str, Any]]) -> str:
    """教室データのlocationを欠落させる"""
    if classrooms_data:
        if 'location' in classrooms_data[0]:
            del classrooms_data[0]['location']
            return "教室データの必須項目 'location' を欠落させました。"
        return "教室データに'location'キーが元々ありませんでした。"
    return "教室データが空で、不正化できませんでした。"

def _corrupt_lecturer_bad_age(lecturers_data: List[Dict[str, Any]]) -> str:
    """講師データのageを範囲外にする"""
    if lecturers_data:
        lecturers_data[0]['age'] = 101
        return "講師データの 'age' を範囲外の値 (101) にしました。"
    return "講師データが空で、不正化できませんでした。"

def _corrupt_lecturer_bad_availability_date(lecturers_data: List[Dict[str, Any]]) -> str:
    """講師データのavailabilityに不正な日付形式を含める"""
    if lecturers_data and lecturers_data[0].get('availability'):
        # availabilityリストの最初の要素を不正な形式に変換
        lecturers_data[0]['availability'][0] = "2025/01/01"
        return "講師データの 'availability' に不正な日付形式 ('YYYY/MM/DD') を含めました。"
    return "講師データまたはavailabilityが空で、不正化できませんでした。"

def _corrupt_course_bad_rank(courses_data: List[Dict[str, Any]]) -> str:
    """講座データのrankを非整数にする"""
    if courses_data:
        courses_data[0]['rank'] = "A"
        return "講座データの 'rank' を非整数 ('A') にしました。"
    return "講座データが空で、不正化できませんでした。"

def _corrupt_course_with_nonexistent_classroom(courses_data: List[Dict[str, Any]]) -> str:
    """講座データが、存在しない教室IDを参照するようにする"""
    if courses_data:
        courses_data[0]['classroom_id'] = "C_NON_EXISTENT_ID"
        return "講座データの 'classroom_id' を存在しないIDにしました。"
    return "講座データが空で、不正化できませんでした。"

# --- Main Data Generation Logic ---

def generate_sample_data(
    output_dir: str,
    num_lecturers: int,
    num_general_courses_per_sunday: int,
    generate_special_courses: bool,
    corrupt_data: bool,
    corruption_type: Optional[str]
) -> Dict[str, Any]:
    """
    指定された条件でサンプルデータを生成し、不正化（オプション）を行う。
    Args:
        output_dir: 生成されたデータを保存するディレクトリ。
        num_lecturers: 生成する講師の数。
        num_general_courses_per_sunday: 各日曜日に生成する一般講座の数。
        generate_special_courses: 特別講座を生成するかどうかのフラグ。
        corrupt_data: データを不正化するかどうかのフラグ。
        corruption_type: 適用する不正化の種類 (corrupt_dataがTrueの場合のみ)。
    Returns:
        生成された全データを含む辞書。
    """
    print(f"--- データ生成開始: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"講師数: {num_lecturers}")
    print(f"一般講座/日曜(各教室): {num_general_courses_per_sunday}")
    print(f"特別講座生成: {generate_special_courses}")
    print(f"不正データ生成: {corrupt_data} (タイプ: {corruption_type if corrupt_data else 'なし'})")

    # 割り当て対象月の設定 (現在の4ヶ月後)
    today_date = datetime.date.today()
    assignment_target_month_start = (today_date + relativedelta(months=4)).replace(day=1)
    assignment_target_month_end = (assignment_target_month_start + relativedelta(months=1)) - datetime.timedelta(days=1)
    print(f"割り当て対象月: {assignment_target_month_start} ~ {assignment_target_month_end}")

    prefectures, prefecture_classroom_ids = generate_prefectures_data()
    classrooms_data = generate_classrooms_data(prefectures, prefecture_classroom_ids)
    lecturers_data = generate_lecturers_data(prefecture_classroom_ids, today_date, assignment_target_month_start, assignment_target_month_end, num_lecturers)
    courses_data = generate_courses_data(prefectures, prefecture_classroom_ids, assignment_target_month_start, assignment_target_month_end, num_general_courses_per_sunday, generate_special_courses)

    print(f"生成データ数: 講師={len(lecturers_data)}, 講座={len(courses_data)}, 教室={len(classrooms_data)}")

    corruption_description = "なし"
    if corrupt_data:
        corruption_functions = {
            "duplicate_classroom_id": lambda: _corrupt_duplicate_classroom_id(classrooms_data),
            "missing_classroom_location": lambda: _corrupt_missing_classroom_location(classrooms_data),
            "lecturer_bad_age": lambda: _corrupt_lecturer_bad_age(lecturers_data),
            "lecturer_bad_availability_date": lambda: _corrupt_lecturer_bad_availability_date(lecturers_data),
            "course_bad_rank": lambda: _corrupt_course_bad_rank(courses_data),
            "course_nonexistent_classroom": lambda: _corrupt_course_with_nonexistent_classroom(courses_data),
        }

        if corruption_type and corruption_type in corruption_functions:
            corruption_description = corruption_functions[corruption_type]()
        else: # ランダムに選択 (元のapp.pyのgenerate_invalid_sample_dataの挙動)
            chosen_corruption_func = random.choice(list(corruption_functions.values()))
            corruption_description = chosen_corruption_func()
        print(f"不正データ適用: {corruption_description}")

    # 日付オブジェクトをISOフォーマット文字列に変換してJSONシリアライズ可能にする
    def convert_dates_to_iso_str(obj):
        if isinstance(obj, datetime.date):
            return obj.isoformat()
        if isinstance(obj, list):
            return [convert_dates_to_iso_str(elem) for elem in obj]
        if isinstance(obj, dict):
            return {k: convert_dates_to_iso_str(v) for k, v in obj.items()}
        return obj

    final_lecturers_data = convert_dates_to_iso_str(lecturers_data)
    final_courses_data = convert_dates_to_iso_str(courses_data)
    final_classrooms_data = convert_dates_to_iso_str(classrooms_data)

    generated_data = {
        "lecturers": final_lecturers_data,
        "courses": final_courses_data,
        "classrooms": final_classrooms_data
    }
    
    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSONファイルとして保存
    with open(output_path / "lecturers.json", "w", encoding="utf-8") as f:
        json.dump(final_lecturers_data, f, ensure_ascii=False, indent=2)
    with open(output_path / "courses.json", "w", encoding="utf-8") as f:
        json.dump(final_courses_data, f, ensure_ascii=False, indent=2)
    with open(output_path / "classrooms.json", "w", encoding="utf-8") as f:
        json.dump(final_classrooms_data, f, ensure_ascii=False, indent=2)
    
    print(f"データ生成完了。ファイルは '{output_dir}' に保存されました。")
    return generated_data

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="講師割り当てシステム用のサンプルデータを生成します。",
        formatter_class=argparse.RawTextHelpFormatter # ヘルプメッセージの整形用
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="生成されたJSONファイルを保存するディレクトリ。デフォルト: data"
    )
    parser.add_argument(
        "--num_lecturers",
        type=int,
        default=300, # 旧 app.py と同じデフォルト値
        help="生成する講師の数。デフォルト: 300"
    )
    parser.add_argument(
        "--num_general_courses_per_sunday",
        type=int,
        default=2, # 旧 app.py と同じデフォルト値
        help="各教室の各日曜日に生成する一般講座の数。デフォルト: 2"
    )
    parser.add_argument(
        "--no_special_courses",
        action="store_true", # このフラグが存在すればTrue
        help="特別講座を生成しない場合は指定します。"
    )
    parser.add_argument(
        "--corrupt_data",
        action="store_true", # このフラグが存在すればTrue
        help="生成されたデータに意図的に不正な値を適用します（ランダムな不正化）。"
    )
    parser.add_argument(
        "--corruption_type",
        type=str,
        choices=[
            "duplicate_classroom_id",
            "missing_classroom_location",
            "lecturer_bad_age",
            "lecturer_bad_availability_date",
            "course_bad_rank",
            "course_nonexistent_classroom"
        ],
        help="適用する不正化の種類を具体的に指定します。--corrupt_data と併用。"
    )

    args = parser.parse_args()

    # 引数に基づきデータ生成を実行
    generate_sample_data(
        output_dir=args.output_dir,
        num_lecturers=args.num_lecturers,
        num_general_courses_per_sunday=args.num_general_courses_per_sunday,
        generate_special_courses=not args.no_special_courses, # no_special_coursesがTrueならFalse
        corrupt_data=args.corrupt_data,
        corruption_type=args.corruption_type
    )
