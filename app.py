import streamlit as st
# import streamlit.components.v1 as components # 削除
import pandas as pd
import datetime # 日付処理用に追加
import time # 処理時間測定用に追加
import google.generativeai as genai # Gemini API 用
# from streamlit_oauth import OAuth2Component # OIDC認証用 # 削除
# from google.oauth2 import id_token # IDトークン検証用 # 削除
# from google.auth.transport import requests as google_requests # IDトークン検証用 # 削除
import multiprocessing
import random # データ生成用
import os # CPUコア数を取得するために追加
import numpy as np # データ型変換のために追加
# dateutil.relativedelta を使用するため、インストールが必要な場合があります。
# pip install python-dateutil
from dateutil.relativedelta import relativedelta
import logging # logging モジュールをインポート
from typing import List, Optional, Any, Tuple # TypedDict は optimization_engine に移動

# --- [修正点1] 分離したモジュールをインポート ---
import optimization_gateway
import optimization_solver
from ortools.sat.python import cp_model # solver_raw_status_code の比較等で使用
# ---------------------------------------------

# --- [修正点3] ログ設定を別ファイルに分離し、定数をインポート ---
from utils.logging_config import setup_logging, APP_LOG_FILE, GATEWAY_LOG_FILE, SOLVER_LOG_FILE
from utils.error_definitions import InvalidInputError
# ---


# --- 1. データ定義 (LOG_EXPLANATIONS と _get_log_explanation は削除) --- 
# SolverOutput は optimization_engine.py に移動

# --- Gemini API送信用ログのフィルタリング関数 (グローバルスコープに移動) ---
def filter_log_for_gemini(log_content: str) -> str:
    """
    ログ全体から OR-Tools ソルバーに関連するログ行のみを抽出する。
    [OR-Tools Solver] プレフィックスを持つ行をフィルタリングします。
    """
    lines = log_content.splitlines()
    solver_log_prefix = "[OR-Tools Solver]"
    
    solver_log_lines = [line for line in lines if solver_log_prefix in line]
    
    if not solver_log_lines:
        return "ソルバーのログが見つかりませんでした。最適化が実行されなかったか、ログの形式が変更された可能性があります。"
        
    return "\n".join(solver_log_lines)

# --- 大規模データ生成 ---
# (変更なし)

# --- Gemini API 連携 ---
# (get_gemini_explanation 関数は変更なしのため省略)
# ... (get_gemini_explanation の元のコード) ...

# --- データ生成関数群 ---
# これらの関数は initialize_app_data 内で呼び出され、結果を st.session_state に格納する
def get_gemini_explanation(log_text: str,
                           api_key: str,
                           solver_status: str,
                           objective_value: Optional[float],
                           assignments_summary: Optional[pd.DataFrame]) -> Tuple[str, Optional[str]]:
    """
    指定されたログテキストと最適化結果を Gemini API に送信し、解説を取得します。
    戻り値: (解説テキストまたはエラーメッセージ, 送信したプロンプト全体またはNone)
    """
    if not api_key:
        return "エラー: Gemini API キーが設定されていません。", None

    readme_content = ""
    readme_path = "README.md" # app.py と同じ階層にあると仮定
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
    except FileNotFoundError:
        readme_content = "システム仕様書(README.md)が見つかりませんでした。\n"
    except Exception as e:
        # READMEの読み込みエラーもプロンプトに含めてGeminiに送るため、ここではエラーを返さない
        # ただし、ログには残す
        logging.error(f"Error reading README.md for Gemini prompt: {e}", exc_info=True)
        readme_content = f"システム仕様書(README.md)の読み込み中にエラーが発生しました: {str(e)}\n" # このエラーメッセージがプロンプトに含まれる

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro') 
        prompt = f"""以下のシステム仕様とログについて、IT専門家でない人にも分かりやすく解説してください。
## システム仕様
{readme_content}

## ログ解説のリクエスト
上記のシステム仕様を踏まえ、以下のログの各部分が何を示しているのか、全体としてどのような処理が行われているのかを説明してください。
特に重要な情報、警告、エラーがあれば指摘し、考えられる原因や対処法についても言及してください。
最適化結果とログの内容を関連付けて解説してください。

## 最適化結果のサマリー
- 求解ステータス: {solver_status}
- 目的値: {objective_value if objective_value is not None else 'N/A'}
"""
        if assignments_summary is not None and not assignments_summary.empty:
            prompt += f"- 割り当て件数: {len(assignments_summary)} 件\n"
            # 必要に応じて、assignments_summary からさらに情報を抜粋してプロンプトに追加できます。
            # 例: prompt += f"- 主な割り当て講師（上位3名）: ... \n"
        else:
            prompt += "- 割り当て: なし\n"

        prompt += f"""
ログ本文:
```text
{log_text}
```

解説:
"""
        response = model.generate_content(prompt)
        return response.text, prompt
    except Exception as e:
        # st.error を直接呼ばず、エラーメッセージ文字列を返す
        return f"Gemini APIエラー: {str(e)[:500]}...", prompt # エラー時も構築されたプロンプトを返す

def generate_prefectures_data():
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
    return PREFECTURES, PREFECTURE_CLASSROOM_IDS

def generate_classrooms_data(prefectures, prefecture_classroom_ids):
    classrooms_data = []
    for i, pref_name in enumerate(prefectures):
        classrooms_data.append({"id": prefecture_classroom_ids[i], "location": pref_name})
    return classrooms_data

def generate_lecturers_data(prefecture_classroom_ids, today_date, assignment_target_month_start, assignment_target_month_end):
    lecturers_data = []
    # 講師の空き日を生成する期間 (対象月の前後1ヶ月)
    availability_period_start = assignment_target_month_start - relativedelta(months=1)
    availability_period_end = assignment_target_month_end + relativedelta(months=1)

    all_possible_dates_for_availability = []
    current_date_iter = availability_period_start
    while current_date_iter <= availability_period_end:
        all_possible_dates_for_availability.append(current_date_iter.strftime("%Y-%m-%d"))
        current_date_iter += datetime.timedelta(days=1)

    for i in range(1, 301): # 講師数を300人に変更
        num_available_days = random.randint(15, 45) # 対象月±1ヶ月の期間内で15～45日空いている
        if len(all_possible_dates_for_availability) >= num_available_days:
            availability = random.sample(all_possible_dates_for_availability, num_available_days)
            availability.sort()
        else: # 万が一、候補日が少ない場合（通常ありえない）
            availability = all_possible_dates_for_availability[:]
        # 過去の割り当て履歴を生成 (約10件)
        num_past_assignments = random.randint(8, 12) # 8から12件の間でランダム
        past_assignments = []

        # 新しい資格ランク生成ロジック
        has_special_qualification = random.choice([True, False, False]) # 約1/3が特別資格持ち
        special_rank = None
        if has_special_qualification:
            special_rank = random.randint(1, 5)
            general_rank = 1 # 特別資格持ちは一般資格ランク1
        else:
            general_rank = random.randint(1, 5)

        for _ in range(num_past_assignments):
            days_ago = random.randint(1, 730) # 過去2年以内のランダムな日付
            assignment_date = today_date - datetime.timedelta(days=days_ago)
            past_assignments.append({
                "classroom_id": random.choice(prefecture_classroom_ids),
                "date": assignment_date.strftime("%Y-%m-%d")
            })
        past_assignments.sort(key=lambda x: x["date"], reverse=True)
        lecturers_data.append({
            "id": f"L{i}",
            "name": f"講師{i:03d}",
            "age": random.randint(22, 65),
            "home_classroom_id": random.choice(prefecture_classroom_ids),
            "qualification_general_rank": general_rank,
            "qualification_special_rank": special_rank,
            "availability": availability,
            "past_assignments": past_assignments
        })
    return lecturers_data

def generate_courses_data(prefectures, prefecture_classroom_ids, assignment_target_month_start, assignment_target_month_end):
    # 新しい講座定義
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

    # 対象月の日曜日リストを作成
    sundays_in_target_month = []
    current_date_iter = assignment_target_month_start
    while current_date_iter <= assignment_target_month_end:
        if current_date_iter.weekday() == 6: # 0:月曜日, 6:日曜日
            sundays_in_target_month.append(current_date_iter.strftime("%Y-%m-%d"))
        current_date_iter += datetime.timedelta(days=1)

    # 対象月の土曜日と平日リストを作成 (特別講座用)
    saturdays_in_target_month_for_special = []
    weekdays_in_target_month_for_special = []
    all_days_in_target_month_for_special_obj = [] # 日付オブジェクトを保持
    current_date_iter = assignment_target_month_start
    while current_date_iter <= assignment_target_month_end:
        all_days_in_target_month_for_special_obj.append(current_date_iter)
        if current_date_iter.weekday() == 5: # 土曜日
            saturdays_in_target_month_for_special.append(current_date_iter.strftime("%Y-%m-%d"))
        elif current_date_iter.weekday() < 5: # 平日
            weekdays_in_target_month_for_special.append(current_date_iter.strftime("%Y-%m-%d"))
        current_date_iter += datetime.timedelta(days=1)

    courses_data = []
    course_counter = 1
    for i, pref_classroom_id in enumerate(prefecture_classroom_ids):
        pref_name = prefectures[i]

        # 一般講座の生成 (対象月の各日曜日に、ランダムに2つのレベルで開催)
        for sunday_str in sundays_in_target_month:
            selected_levels_for_general = random.sample(GENERAL_COURSE_LEVELS, min(2, len(GENERAL_COURSE_LEVELS)))
            for level_info in selected_levels_for_general:
                courses_data.append({
                    "id": f"{pref_classroom_id}-GC{course_counter}",
                    "name": f"{pref_name} 一般講座 {level_info['name_suffix']} ({sunday_str[-5:]})",
                    "classroom_id": pref_classroom_id,
                    "course_type": "general",
                    "rank": level_info['rank'],
                    "schedule": sunday_str
                })
                course_counter += 1

        # 特別講座の生成 (対象月内に1回、土曜優先)
        chosen_date_for_special_course = None
        if saturdays_in_target_month_for_special:
            chosen_date_for_special_course = random.choice(saturdays_in_target_month_for_special)
        elif weekdays_in_target_month_for_special:
            chosen_date_for_special_course = random.choice(weekdays_in_target_month_for_special)
        elif all_days_in_target_month_for_special_obj:
            chosen_date_for_special_course = random.choice(all_days_in_target_month_for_special_obj).strftime("%Y-%m-%d")
        
        if chosen_date_for_special_course:
            level_info_special = random.choice(SPECIAL_COURSE_LEVELS)
            courses_data.append({
                "id": f"{pref_classroom_id}-SC{course_counter}",
                "name": f"{pref_name} 特別講座 {level_info_special['name_suffix']} ({chosen_date_for_special_course[-5:]})",
                "classroom_id": pref_classroom_id,
                "course_type": "special",
                "rank": level_info_special['rank'],
                "schedule": chosen_date_for_special_course
            })
            course_counter += 1
    return courses_data

def get_region_hops(region1, region2, graph):
    if region1 == region2: return 0
    queue = [(region1, 0)]; visited = {region1}
    while queue:
        current_region, dist = queue.pop(0)
        if current_region == region2: return dist
        for neighbor in graph.get(current_region, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float('inf')

def generate_travel_costs_matrix(all_classroom_ids_combined, classroom_id_to_pref_name, prefecture_to_region, region_graph):
    travel_costs_matrix = {}
    for c_from in all_classroom_ids_combined:
        for c_to in all_classroom_ids_combined:
            if c_from == c_to: base_cost = 0
            else:
                pref_from, pref_to = classroom_id_to_pref_name[c_from], classroom_id_to_pref_name[c_to]
                region_from, region_to = prefecture_to_region[pref_from], prefecture_to_region[pref_to]
                is_okinawa_involved = ("沖縄県" in (pref_from, pref_to)) and (pref_from != pref_to)
                if is_okinawa_involved: base_cost = random.randint(80000, 120000)
                elif region_from == region_to: base_cost = random.randint(5000, 15000)
                else:
                    hops = get_region_hops(region_from, region_to, region_graph)
                    if hops == 1: base_cost = random.randint(15000, 30000)
                    elif hops == 2: base_cost = random.randint(35000, 60000)
                    else: base_cost = random.randint(70000, 100000)
            travel_costs_matrix[(c_from, c_to)] = base_cost
    return travel_costs_matrix

def initialize_app_data(force_regenerate: bool = False):
    logger = logging.getLogger('app')
    if force_regenerate or not st.session_state.get("app_data_initialized"):
        logger.info("Initializing application data...")
        st.session_state.TODAY = datetime.date.today()
        start = (st.session_state.TODAY + relativedelta(months=4)).replace(day=1)
        end = (start + relativedelta(months=1)) - datetime.timedelta(days=1)
        st.session_state.ASSIGNMENT_TARGET_MONTH_START, st.session_state.ASSIGNMENT_TARGET_MONTH_END = start, end
        prefs, pref_ids = generate_prefectures_data()
        st.session_state.PREFECTURES, st.session_state.PREFECTURE_CLASSROOM_IDS = prefs, pref_ids
        st.session_state.DEFAULT_CLASSROOMS_DATA = generate_classrooms_data(prefs, pref_ids)
        st.session_state.ALL_CLASSROOM_IDS_COMBINED = pref_ids
        st.session_state.DEFAULT_LECTURERS_DATA = generate_lecturers_data(pref_ids, st.session_state.TODAY, start, end)
        st.session_state.DEFAULT_COURSES_DATA = generate_courses_data(prefs, pref_ids, start, end)
        REGIONS = {"Hokkaido": ["北海道"], "Tohoku": ["青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県"], "Kanto": ["茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県"], "Chubu": ["新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県", "静岡県", "愛知県"], "Kinki": ["三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県"], "Chugoku": ["鳥取県", "島根県", "岡山県", "広島県", "山口県"], "Shikoku": ["徳島県", "香川県", "愛媛県", "高知県"], "Kyushu_Okinawa": ["福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"]}
        st.session_state.PREFECTURE_TO_REGION = {p: r for r, pl in REGIONS.items() for p in pl}
        st.session_state.REGION_GRAPH = {"Hokkaido": {"Tohoku"}, "Tohoku": {"Hokkaido", "Kanto", "Chubu"}, "Kanto": {"Tohoku", "Chubu"}, "Chubu": {"Tohoku", "Kanto", "Kinki"}, "Kinki": {"Chubu", "Chugoku", "Shikoku"}, "Chugoku": {"Kinki", "Shikoku", "Kyushu_Okinawa"}, "Shikoku": {"Kinki", "Chugoku", "Kyushu_Okinawa"}, "Kyushu_Okinawa": {"Chugoku", "Shikoku"}}
        st.session_state.CLASSROOM_ID_TO_PREF_NAME = {item["id"]: item["location"] for item in st.session_state.DEFAULT_CLASSROOMS_DATA}
        st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX = generate_travel_costs_matrix(pref_ids, st.session_state.CLASSROOM_ID_TO_PREF_NAME, st.session_state.PREFECTURE_TO_REGION, st.session_state.REGION_GRAPH)
        st.session_state.app_data_initialized = True
        logger.info("Application data initialized.")

def _corrupt_duplicate_classroom_id():
    classrooms = st.session_state.DEFAULT_CLASSROOMS_DATA
    if len(classrooms) > 1:
        classrooms[1]['id'] = classrooms[0]['id']
        st.session_state.DEFAULT_CLASSROOMS_DATA = classrooms
        return "教室データのIDを重複させました (classrooms[1]['id'] = classrooms[0]['id'])。"
    return "教室データが少なく、IDを重複させられませんでした。"

def _corrupt_missing_classroom_location():
    classrooms = st.session_state.DEFAULT_CLASSROOMS_DATA
    if classrooms:
        del classrooms[0]['location']
        st.session_state.DEFAULT_CLASSROOMS_DATA = classrooms
        return "教室データの必須項目 'location' を欠落させました。"
    return "教室データが空で、不正化できませんでした。"

def _corrupt_lecturer_bad_age():
    lecturers = st.session_state.DEFAULT_LECTURERS_DATA
    if lecturers:
        lecturers[0]['age'] = 101
        st.session_state.DEFAULT_LECTURERS_DATA = lecturers
        return "講師データの 'age' を範囲外の値 (101) にしました。"
    return "講師データが空で、不正化できませんでした。"

def _corrupt_lecturer_bad_availability_date():
    lecturers = st.session_state.DEFAULT_LECTURERS_DATA
    if lecturers and lecturers[0]['availability']:
        lecturers[0]['availability'][0] = "2025/01/01" # 不正な形式
        st.session_state.DEFAULT_LECTURERS_DATA = lecturers
        return "講師データの 'availability' に不正な日付形式 ('YYYY/MM/DD') を含めました。"
    return "講師データまたはavailabilityが空で、不正化できませんでした。"

def _corrupt_course_bad_rank():
    courses = st.session_state.DEFAULT_COURSES_DATA
    if courses:
        courses[0]['rank'] = "A"
        st.session_state.DEFAULT_COURSES_DATA = courses
        return "講座データの 'rank' を非整数 ('A') にしました。"
    return "講座データが空で、不正化できませんでした。"

def _corrupt_course_with_nonexistent_classroom():
    courses = st.session_state.DEFAULT_COURSES_DATA
    if courses:
        courses[0]['classroom_id'] = "C_NON_EXISTENT_ID"
        st.session_state.DEFAULT_COURSES_DATA = courses
        return "講座データの 'classroom_id' を存在しないIDにしました。"
    return "講座データが空で、不正化できませんでした。"

def _corrupt_travel_costs_negative_value():
    costs = st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX
    if costs:
        first_key = next(iter(costs))
        costs[first_key] = -100
        st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX = costs
        return "移動コスト行列に負の値を含めました。"
    return "移動コスト行列が空で、不正化できませんでした。"

def generate_invalid_sample_data():
    initialize_app_data(force_regenerate=True)
    logger = logging.getLogger('app')
    logger.info("Generated a fresh set of valid data to be corrupted for testing.")
    corruption_functions = [_corrupt_duplicate_classroom_id, _corrupt_missing_classroom_location, _corrupt_lecturer_bad_age, _corrupt_lecturer_bad_availability_date, _corrupt_course_bad_rank, _corrupt_course_with_nonexistent_classroom, _corrupt_travel_costs_negative_value,]
    chosen_corruption = random.choice(corruption_functions)
    description = chosen_corruption()
    logger.info(f"Data corruption applied: {description}")
    return description

def handle_regenerate_sample_data():
    logger = logging.getLogger('app')
    logger.info("Regenerate sample data button clicked, callback triggered.")
    initialize_app_data(force_regenerate=True)
    st.session_state.show_regenerate_success_message = True

def handle_generate_invalid_data():
    logger = logging.getLogger('app')
    logger.info("Generate invalid data button clicked, callback triggered.")
    description = generate_invalid_sample_data()
    st.session_state.show_invalid_data_message = description

def handle_execute_changes_callback():
    logger = logging.getLogger('app')
    logger.info(f"Callback: Executing changes for {len(st.session_state.get('assignments_to_change_list', []))} selected assignments.")
    current_forced = st.session_state.get("forced_unassignments_for_solver", [])
    if not isinstance(current_forced, list):
        current_forced = []
        logger.warning("forced_unassignments_for_solver was not a list or None, re-initialized to empty list.")
    if not st.session_state.get("assignments_to_change_list"):
        st.warning("交代する割り当てが選択されていません。")
        logger.warning("handle_execute_changes_callback called with empty assignments_to_change_list.")
        return
    st.session_state.pending_change_summary_info = [{"lecturer_id": item[0], "course_id": item[1], "lecturer_name": item[2], "course_name": item[3], "classroom_name": item[4]} for item in st.session_state.assignments_to_change_list]
    logger.info(f"Pending change summary info: {st.session_state.pending_change_summary_info}")
    newly_forced_unassignments = [(item[0], item[1]) for item in st.session_state.assignments_to_change_list]
    for pair in newly_forced_unassignments:
        if pair not in current_forced:
            current_forced.append(pair)
    st.session_state.forced_unassignments_for_solver = current_forced
    logger.info(f"forced_unassignments_for_solver updated to: {st.session_state.forced_unassignments_for_solver}")
    st.session_state.assignments_to_change_list = []
    run_optimization()

def read_log_file(log_path: str) -> str:
    try:
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logging.error(f"Failed to read log file {log_path}: {e}")
    return ""

def run_optimization():
    logger = logging.getLogger('app')
    keys_to_clear = ["solver_result_cache", "solver_log_for_download", "optimization_error_message", "optimization_gateway_log_for_download", "app_log_for_download", "gemini_explanation", "gemini_api_requested", "gemini_api_error", "last_full_prompt_for_gemini", "optimization_duration"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    logger.info("Cleared previous optimization results from session_state.")
    
    try:
        with st.spinner("最適化計算を実行中..."):
            start_time = time.time()
            
            # --- 変更箇所: adapt_data_for_engineの呼び出しを削除 ---
            logger.info("Starting optimization calculation (optimization_gateway.run_optimization_with_monitoring).")
            solver_output = optimization_gateway.run_optimization_with_monitoring(
                lecturers_data=st.session_state.DEFAULT_LECTURERS_DATA,
                courses_data=st.session_state.DEFAULT_COURSES_DATA,
                classrooms_data=st.session_state.DEFAULT_CLASSROOMS_DATA,
                travel_costs_matrix=st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX,
                weight_past_assignment_recency=st.session_state.get("weight_past_assignment_exp", 0.5),
                weight_qualification=st.session_state.get("weight_qualification_exp", 0.5),
                weight_travel=st.session_state.get("weight_travel_exp", 0.5),
                weight_age=st.session_state.get("weight_age_exp", 0.5),
                weight_frequency=st.session_state.get("weight_frequency_exp", 0.5),
                weight_assignment_shortage=st.session_state.get("weight_assignment_shortage_exp", 0.5),
                weight_lecturer_concentration=st.session_state.get("weight_lecturer_concentration_exp", 0.5),
                weight_consecutive_assignment=st.session_state.get("weight_consecutive_assignment_exp", 0.5),
                allow_under_assignment=st.session_state.allow_under_assignment_cb,
                today_date=st.session_state.TODAY,
                fixed_assignments=st.session_state.get("fixed_assignments_for_solver"),
                forced_unassignments=st.session_state.get("forced_unassignments_for_solver")
            )
            
            st.session_state.optimization_duration = time.time() - start_time
            st.session_state.solver_result_cache = solver_output
            if "fixed_assignments_for_solver" in st.session_state: del st.session_state.fixed_assignments_for_solver
            if "forced_unassignments_for_solver" in st.session_state: del st.session_state.forced_unassignments_for_solver
            st.session_state.solution_executed = True
            st.session_state.view_mode = "optimization_result"

    except InvalidInputError as e:
        logger.error(f"データバリデーションエラーが発生しました: {e}", exc_info=True)
        st.session_state.optimization_error_message = f"入力データの検証中にエラーが発生しました:\n\n{e}"
        st.session_state.solution_executed = True
        st.session_state.view_mode = "optimization_result"
        st.rerun()
    except Exception as e:
        logger.error(f"Unexpected error during optimization process: {e}", exc_info=True)
        st.session_state.optimization_error_message = f"最適化処理中にエラーが発生しました:\n\n{e}"
        st.session_state.solution_executed = True
        st.session_state.view_mode = "optimization_result"
    finally:
        st.session_state.optimization_gateway_log_for_download = read_log_file(GATEWAY_LOG_FILE)
        engine_log_content = read_log_file(SOLVER_LOG_FILE)
        st.session_state.optimization_engine_log_for_download_from_file = engine_log_content
        st.session_state.app_log_for_download = read_log_file(APP_LOG_FILE)
        solver_log_lines = [line for line in engine_log_content.splitlines() if "[OR-Tools Solver]" in line]
        st.session_state.solver_log_for_download = "\n".join(solver_log_lines)

def display_sample_data_view():
    logger = logging.getLogger('app')
    st.header("入力データ")
    if st.session_state.get("show_regenerate_success_message"):
        st.success("サンプルデータを再生成しました。")
        del st.session_state.show_regenerate_success_message
    if st.session_state.get("show_invalid_data_message"):
        st.warning(f"テスト用の不正データを生成しました: {st.session_state.show_invalid_data_message}")
        del st.session_state.show_invalid_data_message
    col1, col2 = st.columns(2)
    with col1:
        st.button("サンプルデータ再生成", key="regenerate_sample_data_button", on_click=handle_regenerate_sample_data, type="primary")
    with col2:
        st.button("テスト用不正データ生成", key="generate_invalid_data_button", on_click=handle_generate_invalid_data)
    st.markdown(f"**現在の割り当て対象月:** {st.session_state.ASSIGNMENT_TARGET_MONTH_START.strftime('%Y年%m月%d日')} ～ {st.session_state.ASSIGNMENT_TARGET_MONTH_END.strftime('%Y年%m月%d日')}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("講師データ (サンプル)")
        df = pd.DataFrame(st.session_state.DEFAULT_LECTURERS_DATA)
        if 'qualification_special_rank' in df.columns:
            df['qualification_special_rank'] = df['qualification_special_rank'].apply(lambda x: "なし" if x is None else x)
        if 'past_assignments' in df.columns:
            df['past_assignments_display'] = df['past_assignments'].apply(lambda a: ", ".join([f"{i['classroom_id']} ({i['date']})" for i in a]) if isinstance(a, list) and a else "履歴なし")
        if 'availability' in df.columns:
            df['availability_display'] = df['availability'].apply(lambda d: ", ".join(d) if isinstance(d, list) else "")
        st.dataframe(df[["id", "name", "age", "home_classroom_id", "qualification_general_rank", "qualification_special_rank", "availability_display", "past_assignments_display"]], height=200)
    with col2:
        st.subheader("講座データ (サンプル)")
        st.dataframe(pd.DataFrame(st.session_state.DEFAULT_COURSES_DATA)[["id", "name", "classroom_id", "course_type", "rank", "schedule"]], height=200)
    st.subheader("教室データと移動コスト (サンプル)")
    col3, col4 = st.columns(2)
    with col3:
        st.dataframe(pd.DataFrame(st.session_state.DEFAULT_CLASSROOMS_DATA))
    with col4:
        df_costs = pd.DataFrame([{"出発教室": k[0], "到着教室": k[1], "コスト": v} for k, v in st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX.items()])
        st.dataframe(df_costs)

def display_objective_function_view():
    # (元のUI表示ロジックをここに完全実装)
    pass

def display_optimization_result_view(gemini_api_key: Optional[str]):
    # (元のUI表示ロジックをここに完全実装)
    pass

def display_change_assignment_view():
    # (元のUI表示ロジックをここに完全実装)
    pass

def main():
    setup_logging()
    logger = logging.getLogger('app')
    st.set_page_config(page_title="講師割り当てシステムデモ", layout="wide")
    initialize_app_data()
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

    if "view_mode" not in st.session_state: st.session_state.view_mode = "sample_data"
    if "assignments_to_change_list" not in st.session_state: st.session_state.assignments_to_change_list = []
    if "solution_executed" not in st.session_state: st.session_state.solution_executed = False
    if "allow_under_assignment_cb" not in st.session_state: st.session_state.allow_under_assignment_cb = True

    st.title("講師割り当てシステム(OR-Tools)-プロトタイプ")

    nav_cols = st.columns([2, 2, 2, 1])
    with nav_cols[0]:
        if st.button("サンプルデータ", use_container_width=True, type="primary" if st.session_state.view_mode == "sample_data" else "secondary"):
            st.session_state.view_mode = "sample_data"; st.rerun()
    with nav_cols[1]:
        if st.button("ソルバーとmodelオブジェクト", use_container_width=True, type="primary" if st.session_state.view_mode == "objective_function" else "secondary"):
            st.session_state.view_mode = "objective_function"; st.rerun()
    with nav_cols[2]:
        if st.session_state.get("solution_executed", False):
            if st.button("最適化結果", use_container_width=True, type="primary" if st.session_state.view_mode == "optimization_result" else "secondary"):
                st.session_state.view_mode = "optimization_result"; st.rerun()

    st.sidebar.button("最適割り当てを実行", type="primary", on_click=run_optimization)
    st.sidebar.markdown("---")
    
    if st.session_state.get("solution_executed") and st.session_state.get("solver_result_cache") and st.session_state.solver_result_cache.get('assignments_df'):
        if st.sidebar.button("割り当て結果を変更", key="change_assignment_view_button", type="secondary" if st.session_state.view_mode != "change_assignment_view" else "primary"):
            st.session_state.view_mode = "change_assignment_view"; st.rerun()
    st.sidebar.markdown("---")

    with st.sidebar.expander("【制約】", expanded=False):
        st.markdown("- 1.講師は、資格ランクを超える講座への割り当てはできない")
        st.markdown("- 2.講師は、個人スケジュールに適合しない講座への割り当てはできない。")
        st.markdown("- 3.講師は、東京、名古屋、大阪の教室には2名を割り当て、それ以外には1名を割り当てる。")

    with st.sidebar.expander("【許容条件】", expanded=False):
        st.checkbox("上記ハード制約3に対し、割り当て不足を許容する", key="allow_under_assignment_cb", value=st.session_state.allow_under_assignment_cb)

    with st.sidebar.expander("【最適化目標】", expanded=False):
        st.slider("移動コストが低い人を優先", 0.0, 1.0, 0.5, 0.1, format="%.1f", key="weight_travel_exp")
        st.slider("年齢の若い人を優先", 0.0, 1.0, 0.5, 0.1, format="%.1f", key="weight_age_exp")
        st.slider("割り当て頻度の少ない人を優先", 0.0, 1.0, 0.5, 0.1, format="%.1f", key="weight_frequency_exp")
        st.slider("講師資格が高い人を優先", 0.0, 1.0, 0.5, 0.1, format="%.1f", key="weight_qualification_exp")
        st.slider("同教室への割り当て実績が無い人を優先", 0.0, 1.0, 0.5, 0.1, format="%.1f", key="weight_past_assignment_exp")
        st.slider("割り当て不足を最小化", 0.0, 1.0, 0.5, 0.1, format="%.1f", key="weight_assignment_shortage_exp")
        st.slider("講師の割り当て集中度を低くする", 0.0, 1.0, 0.5, 0.1, format="%.1f", key="weight_lecturer_concentration_exp")
        st.slider("連日講座への連続割り当てを優先", 0.0, 1.0, 0.5, 0.1, format="%.1f", key="weight_consecutive_assignment_exp")

    if st.session_state.view_mode == "sample_data":
        display_sample_data_view()
    elif st.session_state.view_mode == "objective_function":
        display_objective_function_view()
    elif st.session_state.view_mode == "optimization_result":
        display_optimization_result_view(GEMINI_API_KEY)
    elif st.session_state.view_mode == "change_assignment_view":
        display_change_assignment_view()
    else:
        st.info("サイドバーから表示するデータを選択してください。")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
