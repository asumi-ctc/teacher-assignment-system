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
from utils.error_definitions import InvalidInputError, ProcessExecutionError, ProcessTimeoutError
from ortools.sat.python import cp_model # solver_raw_status_code の比較等で使用
# ---------------------------------------------

# --- [修正点3] ログ設定を別ファイルに分離し、定数をインポート ---
from utils.logging_config import setup_logging, APP_LOG_FILE, GATEWAY_LOG_FILE, SOLVER_LOG_FILE
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
        # num_available_slots = random.randint(3, 7) # 以前のAM/PMスロット数
        # availability = random.sample(ALL_SLOTS, num_available_slots) # 以前の形式

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
            # 各日曜日に異なるレベルの一般講座を例えば2つ生成
            selected_levels_for_general = random.sample(GENERAL_COURSE_LEVELS, min(2, len(GENERAL_COURSE_LEVELS)))
            for level_info in selected_levels_for_general:
                courses_data.append({
                    "id": f"{pref_classroom_id}-GC{course_counter}",
                    "name": f"{pref_name} 一般講座 {level_info['name_suffix']} ({sunday_str[-5:]})", # 日付の月日部分を名前に含める
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
        elif all_days_in_target_month_for_special_obj: # 万が一の場合
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

def generate_travel_costs_matrix(all_classroom_ids_combined, classroom_id_to_pref_name, prefecture_to_region, region_graph):
    travel_costs_matrix = {}
    for c_from in all_classroom_ids_combined:
        for c_to in all_classroom_ids_combined:
            if c_from == c_to:
                base_cost = 0
            else:
                pref_from = classroom_id_to_pref_name[c_from]
                pref_to = classroom_id_to_pref_name[c_to]
                region_from = prefecture_to_region[pref_from]
                region_to = prefecture_to_region[pref_to]
                is_okinawa_involved = (pref_from == "沖縄県" and pref_to != "沖縄県") or \
                                      (pref_to == "沖縄県" and pref_from != "沖縄県")
                if is_okinawa_involved:
                    base_cost = random.randint(80000, 120000)
                elif region_from == region_to:
                    base_cost = random.randint(5000, 15000)
                else:
                    hops = get_region_hops(region_from, region_to, region_graph)
                    if hops == 1:
                        base_cost = random.randint(15000, 30000)
                    elif hops == 2:
                        base_cost = random.randint(35000, 60000)
                    else:
                        base_cost = random.randint(70000, 100000)
            travel_costs_matrix[(c_from, c_to)] = base_cost
    return travel_costs_matrix

# get_region_hops 関数は変更なしのため省略
# ... (get_region_hops の元のコード) ...
def get_region_hops(region1, region2, graph):
    """地域間のホップ数（隣接度）を計算する"""
    if region1 == region2:
        return 0
    
    queue = [(region1, 0)]
    visited = {region1}
    
    while queue:
        current_region, dist = queue.pop(0)
        if current_region == region2:
            return dist
        
        for neighbor in graph.get(current_region, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float('inf') # 到達不能 (通常は発生しない)

# --- グローバル定数 (データ生成後に設定されるもの) ---
# これらは initialize_app_data 内で設定され、st.session_state に格納される
# PREFECTURES, PREFECTURE_CLASSROOM_IDS, DEFAULT_CLASSROOMS_DATA, ALL_CLASSROOM_IDS_COMBINED,
# DEFAULT_LECTURERS_DATA, DEFAULT_COURSES_DATA, REGIONS, PREFECTURE_TO_REGION, REGION_GRAPH,
# CLASSROOM_ID_TO_PREF_NAME, DEFAULT_TRAVEL_COSTS_MATRIX, TODAY,
# DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT
# は initialize_app_data で st.session_state に格納される

# 過去の割り当てがない、または日付パース不能な場合に設定するデフォルトの経過日数 (ペナルティ計算上、十分に大きい値)
# これは initialize_app_data で st.session_state.DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT に設定される

# --- 2. OR-Tools 最適化ロジック ---
# --- [修正点2] solve_assignment関数とSolverOutputクラスはここから削除されている ---

# --- 3. Streamlit UI ---
def initialize_app_data(force_regenerate: bool = False):
    """
    アプリケーションの初期データを生成し、セッション状態に保存する。
    force_regenerate=True の場合、既存のデータがあっても強制的に再生成する。
    """
    logger = logging.getLogger('app')
    logger.info(f"Entering initialize_app_data(force_regenerate={force_regenerate})")
    logger.info(f"  Initial 'app_data_initialized': {st.session_state.get('app_data_initialized')}")

    # データ生成の条件:
    # 1. force_regenerate が True
    # 2. st.session_state.get("app_data_initialized") が True でない (存在しない、None、False など)
    if force_regenerate or not st.session_state.get("app_data_initialized"):
        if force_regenerate:
            logger.info("  force_regenerate is True. Regenerating data.")
        elif "app_data_initialized" not in st.session_state:
            logger.info("  'app_data_initialized' not in session_state. Generating data for the first time or after session loss.")
        else: # app_data_initialized が存在するが、True ではない場合
            logger.info(f"  'app_data_initialized' is {st.session_state.get('app_data_initialized')}. Regenerating data.")

        # --- データ生成処理 ---
        st.session_state.TODAY = datetime.date.today()
        logger.info("  TODAY set.")
        # 割り当て対象月の設定 (現在の4ヶ月後)
        assignment_target_month_start_val = (st.session_state.TODAY + relativedelta(months=4)).replace(day=1)
        st.session_state.ASSIGNMENT_TARGET_MONTH_START = assignment_target_month_start_val
        next_month_val = assignment_target_month_start_val + relativedelta(months=1)
        st.session_state.ASSIGNMENT_TARGET_MONTH_END = next_month_val - datetime.timedelta(days=1)
        logger.info(f"  Assignment target month set: {st.session_state.ASSIGNMENT_TARGET_MONTH_START} to {st.session_state.ASSIGNMENT_TARGET_MONTH_END}")

        PREFECTURES_val, PREFECTURE_CLASSROOM_IDS_val = generate_prefectures_data()
        st.session_state.PREFECTURES = PREFECTURES_val
        st.session_state.PREFECTURE_CLASSROOM_IDS = PREFECTURE_CLASSROOM_IDS_val
        logger.info(f"  generate_prefectures_data() completed. {len(PREFECTURES_val)} prefectures.")

        st.session_state.DEFAULT_CLASSROOMS_DATA = generate_classrooms_data(
            st.session_state.PREFECTURES, st.session_state.PREFECTURE_CLASSROOM_IDS
        )
        logger.info(f"  generate_classrooms_data() completed. {len(st.session_state.DEFAULT_CLASSROOMS_DATA)} classrooms.")
        st.session_state.ALL_CLASSROOM_IDS_COMBINED = st.session_state.PREFECTURE_CLASSROOM_IDS

        st.session_state.DEFAULT_LECTURERS_DATA = generate_lecturers_data(
            st.session_state.PREFECTURE_CLASSROOM_IDS, st.session_state.TODAY,
            st.session_state.ASSIGNMENT_TARGET_MONTH_START, # 追加
            st.session_state.ASSIGNMENT_TARGET_MONTH_END    # 追加
        )
        logger.info(f"  generate_lecturers_data() completed. {len(st.session_state.DEFAULT_LECTURERS_DATA)} lecturers.")
        st.session_state.DEFAULT_COURSES_DATA = generate_courses_data(
            st.session_state.PREFECTURES, st.session_state.PREFECTURE_CLASSROOM_IDS,
            st.session_state.ASSIGNMENT_TARGET_MONTH_START, # 追加
            st.session_state.ASSIGNMENT_TARGET_MONTH_END    # 追加
        )
        logger.info(f"  generate_courses_data() completed. {len(st.session_state.DEFAULT_COURSES_DATA)} courses.")

        REGIONS = {
            "Hokkaido": ["北海道"], "Tohoku": ["青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県"],
            "Kanto": ["茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県"],
            "Chubu": ["新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県", "静岡県", "愛知県"],
            "Kinki": ["三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県"],
            "Chugoku": ["鳥取県", "島根県", "岡山県", "広島県", "山口県"],
            "Shikoku": ["徳島県", "香川県", "愛媛県", "高知県"],
            "Kyushu_Okinawa": ["福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"]
        }
        # logger.info("REGIONS defined.") # ログレベル調整
        st.session_state.PREFECTURE_TO_REGION = {pref: region for region, prefs in REGIONS.items() for pref in prefs}
        st.session_state.REGION_GRAPH = {
            "Hokkaido": {"Tohoku"}, "Tohoku": {"Hokkaido", "Kanto", "Chubu"},
            "Kanto": {"Tohoku", "Chubu"}, "Chubu": {"Tohoku", "Kanto", "Kinki"},
            "Kinki": {"Chubu", "Chugoku", "Shikoku"}, "Chugoku": {"Kinki", "Shikoku", "Kyushu_Okinawa"},
            "Shikoku": {"Kinki", "Chugoku", "Kyushu_Okinawa"}, "Kyushu_Okinawa": {"Chugoku", "Shikoku"}
        }
        # logger.info("PREFECTURE_TO_REGION and REGION_GRAPH defined.") # ログレベル調整
        st.session_state.CLASSROOM_ID_TO_PREF_NAME = {
            item["id"]: item["location"] for item in st.session_state.DEFAULT_CLASSROOMS_DATA
        }
        st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX = generate_travel_costs_matrix(
            st.session_state.ALL_CLASSROOM_IDS_COMBINED,
            st.session_state.CLASSROOM_ID_TO_PREF_NAME,
            st.session_state.PREFECTURE_TO_REGION,
            st.session_state.REGION_GRAPH
        )
        logger.info(f"  generate_travel_costs_matrix() completed. {len(st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX)} entries.")
        # --- データ生成処理終了 ---
        st.session_state.app_data_initialized = True
        logger.info("  Data generation logic executed. 'app_data_initialized' set to True.")
    else:
        # st.session_state.get("app_data_initialized") is True and force_regenerate is False
        logger.info("  'app_data_initialized' is True and force_regenerate is False. Skipping data generation.")

    logger.info(f"  Exiting initialize_app_data. 'app_data_initialized': {st.session_state.get('app_data_initialized')}")

def _corrupt_duplicate_classroom_id():
    """教室データのIDを重複させる"""
    classrooms = st.session_state.DEFAULT_CLASSROOMS_DATA
    if len(classrooms) > 1:
        classrooms[1]['id'] = classrooms[0]['id']
        st.session_state.DEFAULT_CLASSROOMS_DATA = classrooms
        return "教室データのIDを重複させました (classrooms[1]['id'] = classrooms[0]['id'])。"
    return "教室データが少なく、IDを重複させられませんでした。"

def _corrupt_missing_classroom_location():
    """教室データのlocationを欠落させる"""
    classrooms = st.session_state.DEFAULT_CLASSROOMS_DATA
    if classrooms:
        del classrooms[0]['location']
        st.session_state.DEFAULT_CLASSROOMS_DATA = classrooms
        return "教室データの必須項目 'location' を欠落させました。"
    return "教室データが空で、不正化できませんでした。"

def _corrupt_lecturer_bad_age():
    """講師データのageを範囲外にする"""
    lecturers = st.session_state.DEFAULT_LECTURERS_DATA
    if lecturers:
        lecturers[0]['age'] = 101
        st.session_state.DEFAULT_LECTURERS_DATA = lecturers
        return "講師データの 'age' を範囲外の値 (101) にしました。"
    return "講師データが空で、不正化できませんでした。"

def _corrupt_lecturer_bad_availability_date():
    """講師データのavailabilityに不正な日付形式を含める"""
    lecturers = st.session_state.DEFAULT_LECTURERS_DATA
    if lecturers and lecturers[0]['availability']:
        lecturers[0]['availability'][0] = "2025/01/01" # 不正な形式
        st.session_state.DEFAULT_LECTURERS_DATA = lecturers
        return "講師データの 'availability' に不正な日付形式 ('YYYY/MM/DD') を含めました。"
    return "講師データまたはavailabilityが空で、不正化できませんでした。"

def _corrupt_course_bad_rank():
    """講座データのrankを非整数にする"""
    courses = st.session_state.DEFAULT_COURSES_DATA
    if courses:
        courses[0]['rank'] = "A"
        st.session_state.DEFAULT_COURSES_DATA = courses
        return "講座データの 'rank' を非整数 ('A') にしました。"
    return "講座データが空で、不正化できませんでした。"

def _corrupt_course_with_nonexistent_classroom():
    """講座データが、存在しない教室IDを参照するようにする"""
    courses = st.session_state.DEFAULT_COURSES_DATA
    if courses:
        courses[0]['classroom_id'] = "C_NON_EXISTENT_ID"
        st.session_state.DEFAULT_COURSES_DATA = courses
        return "講座データの 'classroom_id' を存在しないIDにしました。"
    return "講座データが空で、不正化できませんでした。"

def _corrupt_travel_costs_negative_value():
    """移動コストに負の値を含める"""
    costs = st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX
    if costs:
        first_key = next(iter(costs))
        costs[first_key] = -100
        st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX = costs
        return "移動コスト行列に負の値を含めました。"
    return "移動コスト行列が空で、不正化できませんでした。"

def generate_invalid_sample_data():
    """意図的に不正なデータを生成し、st.session_state を上書きする。"""
    initialize_app_data(force_regenerate=True)
    logger = logging.getLogger('app')
    logger.info("Generated a fresh set of valid data to be corrupted for testing.")

    corruption_functions = [
        _corrupt_duplicate_classroom_id, _corrupt_missing_classroom_location,
        _corrupt_lecturer_bad_age, _corrupt_lecturer_bad_availability_date,
        _corrupt_course_bad_rank, _corrupt_course_with_nonexistent_classroom,
        _corrupt_travel_costs_negative_value,
    ]
    chosen_corruption = random.choice(corruption_functions)
    description = chosen_corruption()
    logger.info(f"Data corruption applied: {description}")
    return description

# --- コールバック関数の定義 (グローバルスコープに移動) ---
def handle_regenerate_sample_data():
    logger = logging.getLogger('app') # Get logger inside function
    logger.info("Regenerate sample data button clicked, callback triggered.")
    initialize_app_data(force_regenerate=True)
    st.session_state.show_regenerate_success_message = True # メッセージ表示用フラグ

def handle_generate_invalid_data():
    logger = logging.getLogger('app') # Get logger inside function
    logger.info("Generate invalid data button clicked, callback triggered.")
    description = generate_invalid_sample_data()
    st.session_state.show_invalid_data_message = description # メッセージを保存

def run_optimization():
    """最適化を実行し、結果をセッション状態に保存するコールバック関数"""
    logger = logging.getLogger('app') # Get logger inside function
    keys_to_clear_on_execute = [
        "solver_result_cache",
        "solver_log_for_download", "optimization_error_message",
        "optimization_gateway_log_for_download",
        "app_log_for_download", "gemini_explanation", "gemini_api_requested",
        "gemini_api_error", "last_full_prompt_for_gemini", "optimization_duration" # 処理時間もクリア
    ]
    for key in keys_to_clear_on_execute:
        if key in st.session_state:
            del st.session_state[key]
    logger.info("Cleared previous optimization results from session_state.")

    def read_log_file(log_path: str) -> str:
        """ログファイルを読み込んで内容を返す。存在しない場合は空文字列を返す。"""
        try:
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to read log file {log_path}: {e}")
        return ""

    try:
        with st.spinner("最適化計算を実行中..."):
            start_time = time.time() # 処理時間測定開始
            
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
        
        end_time = time.time() # 処理時間測定終了
        elapsed_time = end_time - start_time
        logger.info(f"Optimization process took {elapsed_time:.2f} seconds.")
        st.session_state.optimization_duration = elapsed_time # 結果をセッションに保存

        logger.info("solve_assignment completed.")

        if not isinstance(solver_output, dict):
            raise TypeError(f"最適化関数の戻り値が不正です。型: {type(solver_output).__name__}")

        required_keys = ["status", "message", "solution_status", "objective_value", "assignments_df", "lecturer_course_counts", "course_assignment_counts", "course_remaining_capacity", "raw_solver_status_code"]
        missing_keys = [key for key in required_keys if key not in solver_output]
        if missing_keys:
            raise KeyError(f"最適化関数の戻り値に必要なキーが不足しています。不足キー: {missing_keys}")

        st.session_state.solver_result_cache = solver_output

        if "fixed_assignments_for_solver" in st.session_state: del st.session_state.fixed_assignments_for_solver
        if "forced_unassignments_for_solver" in st.session_state: del st.session_state.forced_unassignments_for_solver

        st.session_state.solution_executed = True
        st.session_state.view_mode = "optimization_result"

    except (InvalidInputError, ProcessExecutionError, ProcessTimeoutError) as e:
        logger.error(f"最適化ゲートウェイでエラーが発生しました: {e}", exc_info=True)
        # エラーの種類に応じてユーザーへのメッセージを調整
        if isinstance(e, InvalidInputError):
            error_message = f"入力データの検証中にエラーが発生しました:\n\n{e}"
        elif isinstance(e, ProcessTimeoutError):
            error_message = f"最適化処理がタイムアウトしました:\n\n{e}"
        elif isinstance(e, ProcessExecutionError):
            error_message = f"最適化プロセスの実行中にエラーが発生しました:\n\n{e}"
        else:
            error_message = f"予期せぬ最適化エラーが発生しました:\n\n{e}"

        st.session_state.optimization_error_message = error_message
        # ログダウンロード用に空の文字列を設定
        st.session_state.solver_log_for_download = ""
        st.session_state.app_log_for_download = ""
        # UIにエラーを表示するための設定
        st.session_state.solution_executed = True
        st.session_state.view_mode = "optimization_result"
        st.rerun() # UIを即時更新してエラーを表示

    except Exception as e:
        logger.error(f"予期せぬエラーが発生しました: {e}", exc_info=True)
        import traceback
        error_trace = traceback.format_exc()
        st.session_state.optimization_error_message = f"最適化処理中にエラーが発生しました:\n\n{error_trace}"
        st.session_state.solver_log_for_download = ""
        st.session_state.app_log_for_download = ""
        st.session_state.solution_executed = True
        st.session_state.view_mode = "optimization_result"

    finally:
        # 処理の最後にログファイルを読み込む
        logger.info("Reading log files to store in session state.")
        st.session_state.optimization_gateway_log_for_download = read_log_file(GATEWAY_LOG_FILE)
        # optimization_engine のログは直接ファイルから読み込む
        engine_log_content = read_log_file(SOLVER_LOG_FILE)
        st.session_state.optimization_engine_log_for_download_from_file = engine_log_content
        st.session_state.app_log_for_download = read_log_file(APP_LOG_FILE)

        # OR-Toolsソルバーログを抽出してダウンロード用に設定
        solver_log_lines = []
        if engine_log_content:
            solver_log_prefix = "[OR-Tools Solver]"
            for line in engine_log_content.splitlines():
                if solver_log_prefix in line:
                    solver_log_lines.append(line)
        st.session_state.solver_log_for_download = "\n".join(solver_log_lines)
        logger.info(f"Extracted {len(solver_log_lines)} lines of OR-Tools solver log for download.")

        logger.info("Finished reading log files.")

def handle_execute_changes_callback():
    logger = logging.getLogger('app') # Get logger inside function
    logger.info(
        f"Callback: Executing changes for {len(st.session_state.get('assignments_to_change_list', []))} selected assignments."
    )
    
    current_forced = st.session_state.get("forced_unassignments_for_solver", [])
    # Ensure current_forced is a list, even if it was something else or None
    if not isinstance(current_forced, list):
        current_forced = []
        logger.warning("  forced_unassignments_for_solver was not a list or None, re-initialized to empty list.")

    # --- 新しいロジック ---
    if not st.session_state.get("assignments_to_change_list"):
        st.warning("交代する割り当てが選択されていません。")
        logger.warning("handle_execute_changes_callback called with empty assignments_to_change_list.")
        return

    # Store info for summary display later
    st.session_state.pending_change_summary_info = [
        {
            "lecturer_id": item[0], "course_id": item[1],
            "lecturer_name": item[2], "course_name": item[3],
            "classroom_name": item[4] # 教室名も追加
        }
        for item in st.session_state.assignments_to_change_list
    ]
    logger.info(f"  Pending change summary info: {st.session_state.pending_change_summary_info}")

    # Prepare forced_unassignments for the solver
    newly_forced_unassignments = [
        (item[0], item[1]) for item in st.session_state.assignments_to_change_list
    ]
    
    # Combine with any existing forced_unassignments (e.g., from previous individual changes if that feature were kept)
    # For now, it primarily uses the current list from assignments_to_change_list.
    # Ensure current_forced is a list (already done above)
    
    # Add new ones, avoid duplicates
    for pair in newly_forced_unassignments:
        if pair not in current_forced: # current_forced is from st.session_state.get("forced_unassignments_for_solver", [])
            current_forced.append(pair)
            
    st.session_state.forced_unassignments_for_solver = current_forced # Update session state
    logger.info(f"  forced_unassignments_for_solver updated to: {st.session_state.forced_unassignments_for_solver}")
    
    # Clear the selection list for the "Change Assignment" view as they are now being processed
    st.session_state.assignments_to_change_list = []
    # --- ここまで新しいロジック ---

    # Trigger the main optimization logic
    run_optimization()
# --- コールバック関数の定義ここまで ---

def display_sample_data_view():
    """「サンプルデータ」ビューを描画する"""
    logger = logging.getLogger('app')
    st.header("入力データ")

    if st.session_state.get("show_regenerate_success_message"):
        st.success("サンプルデータを再生成しました。")
        del st.session_state.show_regenerate_success_message # メッセージ表示後にフラグを削除

    if st.session_state.get("show_invalid_data_message"):
        st.warning(f"テスト用の不正データを生成しました: {st.session_state.show_invalid_data_message}")
        del st.session_state.show_invalid_data_message

    # ボタンを横並びに配置し、配色を変更
    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "サンプルデータ再生成",
            key="regenerate_sample_data_button",
            on_click=handle_regenerate_sample_data,
            type="primary", # 重要な操作のため primary (青系) に変更
            help="現在の入力データを破棄し、新しいサンプルデータを生成します。注意：未保存の変更は失われます。"
        )
    with col2:
        st.button(
            "テスト用不正データ生成",
            key="generate_invalid_data_button",
            on_click=handle_generate_invalid_data,
            help="データバリデーションのテスト用に、意図的に不正なデータを生成します。"
        )

    logger.info("Displaying sample data.")
    st.markdown(
        f"**現在の割り当て対象月:** {st.session_state.ASSIGNMENT_TARGET_MONTH_START.strftime('%Y年%m月%d日')} "
        f"～ {st.session_state.ASSIGNMENT_TARGET_MONTH_END.strftime('%Y年%m月%d日')}"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("講師データ (サンプル)")
        # st.session_state からデータを取得
        df_lecturers_display = pd.DataFrame(st.session_state.DEFAULT_LECTURERS_DATA)
        # 表示用に 'qualification_special_rank' が None の場合は "なし" に変換
        if 'qualification_special_rank' in df_lecturers_display.columns:
            df_lecturers_display['qualification_special_rank'] = df_lecturers_display['qualification_special_rank'].apply(lambda x: "なし" if x is None else x)
        if 'past_assignments' in df_lecturers_display.columns:
            df_lecturers_display['past_assignments_display'] = df_lecturers_display['past_assignments'].apply( # 新しい列名
                lambda assignments: ", ".join([f"{a['classroom_id']} ({a['date']})" for a in assignments]) if isinstance(assignments, list) and assignments else "履歴なし"
            )
        if 'availability' in df_lecturers_display.columns:
            df_lecturers_display['availability_display'] = df_lecturers_display['availability'].apply(lambda dates: ", ".join(dates) if isinstance(dates, list) else "") # 新しい列名
        # 表示するカラムを調整
        lecturer_display_columns = ["id", "name", "age", "home_classroom_id", "qualification_general_rank", "qualification_special_rank", "availability_display", "past_assignments_display"]
        st.dataframe(df_lecturers_display[lecturer_display_columns], height=200)
    with col2:
        st.subheader("講座データ (サンプル)")
        df_courses_display = pd.DataFrame(st.session_state.DEFAULT_COURSES_DATA)
        course_display_columns = ["id", "name", "classroom_id", "course_type", "rank", "schedule"]
        st.dataframe(df_courses_display[course_display_columns], height=200)

    st.subheader("教室データと移動コスト (サンプル)")
    col3, col4 = st.columns(2)
    with col3:
        st.dataframe(pd.DataFrame(st.session_state.DEFAULT_CLASSROOMS_DATA)) # st.session_state から取得
    with col4:
        # travel_costs_matrix を表示用に整形
        df_travel_costs = pd.DataFrame([
            {"出発教室": k[0], "到着教室": k[1], "コスト": v}
            for k, v in st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX.items() # st.session_state から取得
        ])
        st.dataframe(df_travel_costs)
    logger.info("Sample data display complete.")

def display_objective_function_view():
    """「ソルバーとmodelオブジェクト」ビューを描画する"""
    logger = logging.getLogger('app')
    logger.info("Displaying objective function explanation.")
    st.header("ソルバーとmodelオブジェクト") # ヘッダー名を変更

    st.markdown(
        """
        このシステムでは、数理最適化問題を解くために特定のソルバーと、そのソルバーが理解できる形式で問題を記述した「model オブジェクト」を使用します。
        """
    )

    st.subheader("選択されたソルバー: CP-SAT")
    st.markdown(
        r"""
        この講師割り当て問題の解決には、Google OR-Toolsに含まれる**CP-SATソルバー**が選択されています。
        CP-SATは「Constraint Programming - Satisfiability」の略で、制約プログラミングと充足可能性問題解決の技術を組み合わせた強力なソルバーです。
        """
    )
    with st.expander("CP-SATソルバーを選択した理由"):
        st.markdown(r"""CP-SATソルバーがこの講師割り当て問題に適している主な理由は以下の通りです。

1.  **問題の性質との適合性**:
    講師をどの講座に割り当てるかという多数の組み合わせの中から最適なものを見つけ出す問題であり、CP-SATはこのような問題を得意としています。

2.  **制約の表現力と処理能力**:
    「各講座には1人の講師」「各講師は最大1講座」「資格ランクの適合」「スケジュールの適合（またはペナルティ）」といった複雑な制約条件を効率的に扱うことができます。特に論理的な制約の記述に優れています。

3.  **離散変数への対応**:
    「講師Lを講座Cに割り当てるか否か」を0か1で表現する決定変数が中心となります。CP-SATはこのような離散変数、特にブール変数を効率的に処理するSAT技術を基盤としています。

4.  **目的関数の最適化**:
    割り当ての総コスト（移動、年齢、頻度、資格、近接性、スケジュール違反ペナルティなどの重み付き合計）を最小化するという明確な目的のもと、制約を満たしつつ最適な解を探索できます。

**他のソルバーとの比較の観点:**
-   **線形計画法（LP）ソルバー**では、決定変数が連続値であることを前提とするため、本問題のような0/1のバイナリ変数を直接扱うことや、複雑な論理制約を表現することが困難です。
-   **混合整数計画法（MIP）ソルバー**は、連続変数と整数変数（バイナリ変数を含む）を扱うことができ、本問題もMIPとして定式化可能です。実際に、CP-SATソルバーはMIPソルバーの能力も内包しています。
-   しかし、CP-SATは特に**制約充足の側面が強い問題**や、**組み合わせ的な構造が顕著な問題**において、MIPソルバーよりも効率的に解を見つけられることがあります。本問題は、多くの「割り当てるか否か」の判断と、それらにかかる制約条件が複雑に絡み合っているため、CP-SATの特性が活きやすいと言えます。

これらの特性と他のソルバーとの比較から、CP-SATソルバーは本問題に対して効率的かつ効果的な解を提供するための強力な選択肢となります。""")
    st.markdown("---")
    st.subheader("model オブジェクトの構成要素")

    st.subheader("1. 決定変数 (Decision Variables)")
    st.markdown(
        r"""
        決定変数は、ソルバーが最適解を見つけるために値を決定する要素です。このモデルでは主に以下の変数が使用されます。

        **基本決定変数:**
        - 各講師 $l$ が各講座 $c$ に割り当てられるかどうかを示すバイナリ変数 $x_{l,c}$。
        $$
        x_{l,c} \in \{0, 1\} \quad (\forall l \in L, \forall c \in C)
        $$
        ここで、
        - $L$ は講師の集合
        - $C$ は講座の集合            
        - $x_{l,c} = 1$ ならば、講師 $l$ は講座 $c$ に割り当てられます。
        - $x_{l,c} = 0$ ならば、講師 $l$ は講座 $c$ に割り当てられません。

        **連日ペア割り当て変数 (ステップ3で追加):**
        - 特定の講師 $l$ が特定の連日講座ペア $p$ をまとめて担当するかどうかを示すバイナリ変数 $y_{l,p}$。
        $$
        y_{l,p} \in \{0, 1\} \quad (\forall l \in L_p, \forall p \in P)
        $$
        ここで、
        - $P$ は連日講座ペアの集合
        - $L_p$ はペア $p$ の両方の講座を担当可能な特別資格を持つ講師の集合

        **補助変数 (主に整数変数またはブール変数):**
        これらは基本決定変数 $x_{l,c}$ から導出され、制約の定義や目的関数の計算を容易にするために使用されます。
        - $\text{num\_total\_assignments}_l$: 講師 $l$ の総割り当て数。
        - $\text{extra\_assignments}_l$: 講師 $l$ のペナルティ対象となる「追加の」割り当て数 (総割り当て数が1を超えた分)。
        """
    )
    st.markdown("**対応するPythonコード (抜粋):**")
    st.code(
        """
# ... ループ内で各講師と講座の組み合わせに対して ...
var = model.NewBoolVar(f'x_{lecturer_id}_{course_id}')
pair_var = model.NewBoolVar(f'y_{lecturer_id_loop_pair}_{pair_id}') # 連日ペア割り当て用

# 補助変数の例
num_total_assignments_l = model.NewIntVar(0, len(courses_data), f'num_total_assignments_{lecturer_id}')
extra_assignments_l = model.NewIntVar(0, len(courses_data), f'extra_assign_{lecturer_id}')
shortage_var = model.NewIntVar(0, target_assignment_count, f'shortage_var_{course_id}') # 割り当て不足数

        """, language="python"
    )
    with st.expander("コード解説", expanded=False):
        st.markdown(
            """
            - `model.NewBoolVar(f'x_{lecturer_id}_{course_id}')`: 各講師と講座のペアに対する基本決定変数 $x_{l,c}$ を作成します。
            - 変数名は、デバッグしやすいように講師IDと講座IDを含む一意な文字列 (`x_L1_C1` など）としています。
            - 作成された変数は、他の情報（講師ID、講座ID、後で計算されるコストなど）と共に `possible_assignments` リストに辞書として格納され、後で制約や目的関数の定義に使用されます。
            - `model.NewBoolVar(f'y_{...}')`: 連日ペア割り当て用のブール変数を作成します。
            - `model.NewIntVar(...)`: 補助的な整数変数（例: 総割り当て回数 `num_total_assignments_l`、追加割り当て回数 `extra_assignments_l`、割り当て不足数 `shortage_var`）を定義します。範囲 (最小値、最大値) と名前を指定します。
            - `model.Add(...)`: 変数間の関係を定義する制約を追加します。例えば、`num_total_assignments_l` がその講師に割り当てられた $x_{l,c}$ の合計と等しくなるようにします。

            """
        )

    st.subheader("2. 制約 (Constraints)")
    st.markdown(
        r"""
        制約は、決定変数が取りうる値の範囲や、変数間の関係を定義する条件です。
        これにより、実行可能な解（許容される割り当てパターン）の範囲が定まります。

        **主な制約:**
        - **各講座への割り当て制約:** 各講座 $c$ には、担当可能な講師候補が存在する場合、その講座の場所（特定地域か否か）とUIの「許容条件」設定に応じて、目標人数が割り当てられます。
            - 講座 $c$ の目標割り当て人数を $\text{TargetCount}_c$ とします。
                - 東京、愛知、大阪の教室の場合: $\text{TargetCount}_c = 2$
                - その他の教室の場合: $\text{TargetCount}_c = 1$
            - **割り当て不足を許容しない場合:**
              $$ \sum_{l \in L_c} x_{l,c} = \text{TargetCount}_c \quad (\forall c \in C \text{ s.t. } L_c \neq \emptyset \text{ and not allow\_under\_assignment}) $$
            - **割り当て不足を許容する場合 (`allow_under_assignment` が True):**
              $$ \sum_{l \in L_c} x_{l,c} \le \text{TargetCount}_c \quad (\forall c \in C \text{ s.t. } L_c \neq \emptyset) $$
              この場合、割り当て不足数 $\text{shortage\_var}_c$ が定義され、目的関数でペナルティが科されます。
              $$ \text{shortage\_var}_c \ge \text{TargetCount}_c - \sum_{l \in L_c} x_{l,c} \quad (\text{if allow\_under\_assignment and } w_{\text{shortage}} > 0) $$
              $$ \text{shortage\_var}_c \ge 0 \quad (\text{IntVarの定義による}) $$
          ここで、
            - $L_c$ は講座 $c$ を担当可能な講師の集合。
            - $w_{\text{shortage}}$ は割り当て不足ペナルティの重み。

        - **講師の割り当て集中度に関するペナルティのための変数定義:**
          UIで講師の割り当て集中度ペナルティの重み $w_{\text{concentration}}$ が0より大きい場合、以下の変数が定義されます。
          - **講師ごとの総割り当て数:**
            $$ \text{num\_total\_assignments}_l = \sum_{c \in C} x_{l,c} \quad (\forall l \in L) $$
          - **ペナルティ対象の「追加の」割り当て数:** (総割り当て数が1回を超える分)
            $$ \text{extra\_assignments}_l \ge \text{num\_total\_assignments}_l - 1 $$
            $$ \text{extra\_assignments}_l \ge 0 $$
          この $\text{extra\_assignments}_l$ が目的関数でペナルティコストと乗算されます。

        - **連日ペア割り当ての関連付け制約 (ステップ3で追加):**
          UIで連日割り当て報酬の重み $w_{\text{consecutive}}$ が0より大きく、かつ該当する連日講座ペアが存在する場合、以下の制約が追加されます。
          講師 $l$ が連日講座ペア $p$（講座 $c_1$ と $c_2$ から成る）をまとめて担当することを示す変数 $y_{l,p}$ が1の場合、
          その講師 $l$ が個別の講座 $c_1$ と $c_2$ にも割り当てられることを保証します。
          $$ y_{l,p} \le x_{l,c_1} \quad (\forall l \in L_p, \forall p=(c_1,c_2) \in P) $$
          $$ y_{l,p} \le x_{l,c_2} \quad (\forall l \in L_p, \forall p=(c_1,c_2) \in P) $$

        - **暗黙的な制約:**
          ソースコード上では、以下の条件を満たさない講師と講座の組み合わせは、そもそも決定変数 $x_{l,c}$ が生成される前の段階で除外されます。これは、それらの組み合わせに対する $x_{l,c}$ が実質的に 0 に固定される制約と見なせます。
            - 講師の資格ランクが講座の要求ランクを満たしている。
                - 一般講座: 講師の一般資格ランク $\le$ 講座ランク、または講師が特別資格を持つ。
                - 特別講座: 講師が特別資格を持ち、その特別資格ランク $\le$ 講座ランク。
            - 講師のスケジュールが講座のスケジュールに適合している。
          連日ペア割り当て変数 $y_{l,p}$ についても同様に、講師が特別資格を持たない場合や、ペアの両方の講座を担当できない場合は、変数が生成されません。
        """
    )
    st.markdown("**対応するPythonコード (抜粋):**")
    st.code(
        """
# 各講座への割り当て制約 (UIの許容条件に応じて変動)
course_id = course_item["id"]
possible_assignments_for_course = assignments_by_course.get(course_id, [])
if possible_assignments_for_course:
    # ... target_assignment_count の決定ロジック (東京、愛知、大阪なら2、他は1) ...
    if allow_under_assignment:
        model.Add(sum(possible_assignments_for_course) <= target_assignment_count)
        if weight_assignment_shortage > 0:
            shortage_var = model.NewIntVar(0, target_assignment_count, f'shortage_var_{course_id}')
            model.Add(shortage_var >= target_assignment_count - sum(possible_assignments_for_course))
            # shortage_penalty_terms リストに shortage_var * actual_penalty_for_shortage を追加
    else:
        model.Add(sum(possible_assignments_for_course) == target_assignment_count)

# 講師の割り当て集中ペナルティのための変数定義 (講師ごとのループ内)
if weight_lecturer_concentration > 0 and actual_penalty_concentration > 0:
    for lecturer_id_loop, lecturer_vars in assignments_by_lecturer.items():
        if not lecturer_vars or len(lecturer_vars) <= 1:
            continue
        num_total_assignments_l = model.NewIntVar(0, len(courses_dict), f'num_total_assignments_{lecturer_id_loop}')
        model.Add(num_total_assignments_l == sum(lecturer_vars))
        extra_assignments_l = model.NewIntVar(0, len(courses_dict), f'extra_assign_{lecturer_id_loop}')
        model.Add(extra_assignments_l >= num_total_assignments_l - 1)
        # objective_terms リストに extra_assignments_l * actual_penalty_concentration を追加

# 連日ペア割り当ての関連付け制約
if weight_consecutive_assignment > 0 and consecutive_day_pairs:
    for pair_detail in consecutive_assignment_pair_vars_details: # 対象講師が見つかったペアのみ
        pair_var = pair_detail["variable"]
        individual_var_c1 = possible_assignments_dict[(pair_detail["lecturer_id"], pair_detail["course1_id"])]["variable"]
        individual_var_c2 = possible_assignments_dict[(pair_detail["lecturer_id"], pair_detail["course2_id"])]["variable"]
        model.Add(pair_var <= individual_var_c1)
        model.Add(pair_var <= individual_var_c2)
        # objective_terms リストに pair_var * -actual_reward_for_pair を追加 (報酬の場合)
        """, language="python"
    )
    with st.expander("コード解説", expanded=False):
        st.markdown(
            r"""
            **各講座への割り当て制約:**
            - `allow_under_assignment` が `False` の場合: `model.Add(sum(...) == target_assignment_count)` で、目標人数ちょうどの割り当てを強制します。
                - `target_assignment_count` は、講座の開催地（東京・愛知・大阪なら2、他は1）によって決まります。
            - `allow_under_assignment` が `True` の場合: `model.Add(sum(...) <= target_assignment_count)` で、目標人数以下の割り当てを許容します。
            - さらに `weight_assignment_shortage > 0` の場合、不足数を表す `shortage_var` を定義し、`shortage_var >= target_assignment_count - sum(...)` で不足数を計算します。この `shortage_var` が目的関数でペナルティコストと乗算されます。

            **講師の割り当て集中ペナルティのための変数定義:**
            - `weight_lecturer_concentration > 0` かつ計算されたペナルティ `actual_penalty_concentration > 0` の場合に実行されます。
            - `num_total_assignments_l = model.NewIntVar(...)`: 講師ごとの総割り当て数を格納する整数変数を定義します。
            - `model.Add(num_total_assignments_l == sum(assignments_for_lecturer_vars))`: 総割り当て数を、その講師に関連する全ての $x_{l,c}$ 変数の合計として定義します。
            - `extra_assignments_l >= num_total_assignments_l - 1`: 総割り当て数が1を超えた部分（ペナルティ対象）を計算します。この変数が目的関数でペナルティコストと乗算されます。

            **連日ペア割り当ての関連付け制約:**
            - `model.Add(pair_var <= individual_var_c1)` と `model.Add(pair_var <= individual_var_c2)`: 連日ペア割り当て変数 `pair_var` が1の場合、対応する個別の講座割り当て変数も1になることを保証します。
            """
        )

    st.subheader("3. 目的関数 (Objective Function)")
    st.markdown(
        r"""
        目的関数は、最適化の目標を定義する数式です。このシステムでは、以下の要素の重み付き合計を**最小化**することが目的です。
        報酬は負のコストとして扱われます。

        $$
        \text{Minimize} \quad Z = \sum_{l,c} (x_{l,c} \cdot \text{Cost}_{l,c}) \\
        \quad + \sum_{c \text{ s.t. allow\_under\_assignment and } w_{\text{shortage}}>0} (\text{shortage\_var}_c \cdot \text{PenaltyShortage}_c) \\
        \quad + \sum_{l \text{ s.t. } w_{\text{concentration}}>0} (\text{extra\_assignments}_l \cdot \text{PenaltyConcentration}_l) \\
        \quad - \sum_{l,p \text{ s.t. } w_{\text{consecutive}}>0} (y_{l,p} \cdot \text{RewardConsecutive}_{l,p})
        $$

        ここで、
        - $x_{l,c}$: 講師 $l$ が講座 $c$ に割り当てられるかを示す変数 (0 or 1)。
        - $\text{Cost}_{l,c}$: 講師 $l$ が講座 $c$ に割り当てられた場合の基本コスト。これは以下の要素の重み付き合計です（コストは整数にスケーリングされます）。
            $$
            \text{Cost}_{l,c} = \text{int} \left( \left( w_{\text{travel}} \cdot \text{TravelCost}_{l,c} + w_{\text{age}} \cdot \text{AgeCost}_l + w_{\text{frequency}} \cdot \text{FrequencyCost}_l \\
            \quad + w_{\text{qualification}} \cdot \text{QualificationCost}_{l,c} + w_{\text{recency}} \cdot \text{RecencyCost}_{l,c} \right) \cdot 100 \right)
            $$
            - $w_{\text{...}}$: UIで設定される各コスト要素の重み。
            - $\text{TravelCost}_{l,c}$: 講師 $l$ の自宅教室から講座 $c$ の教室への移動コスト。
            - $\text{AgeCost}_l$: 講師 $l$ の年齢。
            - $\text{FrequencyCost}_l$: 講師 $l$ の過去の総割り当て回数。
            - $\text{QualificationCost}_{l,c}$: 講師 $l$ の資格ランクに基づくコスト（講座タイプとランクによる）。
            - $\text{RecencyCost}_{l,c}$: 講師 $l$ が講座 $c$ と同じ教室に最後に割り当てられてからの経過日数に基づくコスト（日数が少ないほど高コスト）。
        - $\text{shortage\_var}_c$: 講座 $c$ の割り当て不足数。
        - $\text{PenaltyShortage}_c$: 講座 $c$ の割り当て不足1件あたりのペナルティ（$w_{\text{shortage}}$ と基本ペナルティ値から計算）。
        - $\text{extra\_assignments}_l$: 講師 $l$ のペナルティ対象となる追加割り当て数。
        - $\text{PenaltyConcentration}_l$: 講師 $l$ の追加割り当て1回あたりのペナルティ（$w_{\text{concentration}}$ と基本ペナルティ値から計算）。
        - $y_{l,p}$: 講師 $l$ が連日ペア $p$ をまとめて担当するかを示す変数 (0 or 1)。
        - $\text{RewardConsecutive}_{l,p}$: 講師 $l$ が連日ペア $p$ を担当した場合の報酬（$w_{\text{consecutive}}$ と基本報酬値から計算）。目的関数上は負のコストとして加算。
        $$
        """
    )
    st.markdown("**対応するPythonコード (目的関数の構築部分抜粋):**")
    st.code(
        """
# 1. 基本コスト項 (各割り当て候補 x_{l,c} * Cost_{l,c})
objective_terms = [data["variable"] * data["cost"] for data in possible_assignments_dict.values()]

# 2. 割り当て不足ペナルティ項 (shortage_var_c * PenaltyShortage_c)
if shortage_penalty_terms: # shortage_penalty_terms は事前に (shortage_var * actual_penalty_for_shortage) でリスト化されている
    objective_terms.extend(shortage_penalty_terms)

# 3. 講師の割り当て集中ペナルティ項 (extra_assignments_l * PenaltyConcentration_l)
if weight_lecturer_concentration > 0 and actual_penalty_concentration > 0:
    for lecturer_id_loop, lecturer_vars in assignments_by_lecturer.items():
        # ... (num_total_assignments_l, extra_assignments_l の定義) ...
        if extra_assignments_l: # extra_assignments_l 変数が実際に作成された場合
            objective_terms.append(extra_assignments_l * actual_penalty_concentration)

# 4. 連日割り当ての報酬項 (y_{l,p} * -RewardConsecutive_{l,p})
if weight_consecutive_assignment > 0 and consecutive_day_pairs:
    for pair_detail in consecutive_assignment_pair_vars_details:
        pair_var = pair_detail["variable"]
        # ... actual_reward_for_pair の計算 ...
        if actual_reward_for_pair > 0:
            objective_terms.append(pair_var * -actual_reward_for_pair)

if objective_terms:
    model.Minimize(sum(objective_terms))
else:
    model.Minimize(0) # 目的項がない場合 (通常は発生しない)
        """, language="python"
    )
    with st.expander("コード解説", expanded=False):
        st.markdown(
            r"""
            - **各割り当て候補のコスト計算**:
                - `possible_assignments_dict` の各エントリの `"cost"` キーには、上記数式で示された $\text{Cost}_{l,c}$ が事前に計算・格納されています。
            - **目的関数の設定**:
                - `objective_terms` リストに、上記の目的関数の各項（基本コスト、割り当て不足ペナルティ、講師集中ペナルティ、連日割り当て報酬（負のコスト））を順次追加していきます。
                    - 基本コスト項: `data["variable"] * data["cost"]`
                    - 割り当て不足ペナルティ項: `shortage_var * actual_penalty_for_shortage` (事前に `shortage_penalty_terms` リストに格納)
                    - 講師集中ペナルティ項: `extra_assignments_l * actual_penalty_concentration`
                    - 連日割り当て報酬項: `pair_var * -actual_reward_for_pair`
                - `model.Minimize(sum(objective_terms))`: 全てのコスト項、ペナルティ項、報酬項（負のコスト）の合計を最小化するようにソルバーに指示します。
            """
        )
    logger.info("Objective function explanation display complete.")

def display_optimization_result_view(gemini_api_key: Optional[str]):
    """「最適化結果」ビューを描画する"""
    logger = logging.getLogger('app')
    st.header("最適化結果") # ヘッダーは最初に表示
    logger.info("Displaying optimization result.")

    if not st.session_state.get("solution_executed", False):
        st.info("サイドバーの「最適割り当てを実行」ボタンを押して最適化を実行してください。")
    else: # solution_executed is True
        if "solver_result_cache" not in st.session_state:
            # solver_result_cache がない場合、まず保存されたエラーメッセージを確認
            if "optimization_error_message" in st.session_state and st.session_state.optimization_error_message:
                logger.warning("Optimization error occurred. Displaying error message.")
                st.error("最適化処理でエラーが発生しました。詳細は以下をご確認ください。")
                # st.error(st.session_state.optimization_error_message) # エラーメッセージ全体を表示
                with st.expander("エラー詳細", expanded=True):
                    st.code(st.session_state.optimization_error_message, language=None)
            else:
                logger.info("No solver_result_cache and no optimization_error_message. Prompting user to run optimization.")
                # エラーメッセージもなく、キャッシュもない場合は、従来通りのメッセージ
                st.warning(
                    "最適化結果のデータは現在ありません。\n"
                    "再度結果を表示するには、サイドバーの「最適割り当てを実行」ボタンを押してください。"
                )
        else: # solution_executed is True and solver_result_cache exists
            logger.info("solver_result_cache found. Displaying results.")
            solver_result = st.session_state.solver_result_cache
            st.subheader(f"求解ステータス: {solver_result['solution_status']}")

            metric_cols = st.columns(2)
            with metric_cols[0]:
                if solver_result['objective_value'] is not None:
                    st.metric("総コスト (目的値)", f"{solver_result['objective_value']:.2f}")
            with metric_cols[1]:
                if 'optimization_duration' in st.session_state:
                    st.metric("処理時間", f"{st.session_state.optimization_duration:.2f} 秒", help="データ準備から最適化完了までの時間です。")

            if solver_result['raw_solver_status_code'] in [cp_model.FEASIBLE, cp_model.UNKNOWN]:
                st.warning(
                    """
                    時間制限(Time Limit)内に最適解が見つかりませんでした。現在の最良の解を表示します。

                    もう一度やり直す場合は、余り必要としない最適化目標の重みを0.0に設定することで、その最適化目標が除外されて計算時間が短縮される可能性があります。
                    """
                )

            # --- [リファクタリング] 結果表示ロジック ---
            # assignments が存在し、空でない場合のみ結果を表示する
            if solver_result.get('assignments_df'):
                results_df = pd.DataFrame(solver_result['assignments_df'])

                # 全講座割り当てメッセージ
                assigned_course_ids = set(results_df["講座ID"])
                unassigned_courses = [c for c in st.session_state.DEFAULT_COURSES_DATA if c["id"] not in assigned_course_ids]
                if not unassigned_courses:
                    st.success("全ての講座が割り当てられました。")

                # --- サマリー表示 ---
                st.subheader("割り当て結果サマリー")
                summary_data = []
                summary_data.append(("**総割り当て件数**", f"{len(results_df)}件"))

                total_travel_cost = results_df["移動コスト(元)"].sum()
                summary_data.append(("**移動コストの合計値**", f"{total_travel_cost} 円"))

                assigned_lecturer_ids = results_df["講師ID"].unique()
                temp_assigned_lecturers = [l for l in st.session_state.DEFAULT_LECTURERS_DATA if l["id"] in assigned_lecturer_ids]

                if temp_assigned_lecturers:
                    avg_age = sum(l.get("age", 0) for l in temp_assigned_lecturers) / len(temp_assigned_lecturers)
                    summary_data.append(("**平均年齢**", f"{avg_age:.1f}才"))
                    avg_frequency = sum(len(l.get("past_assignments", [])) for l in temp_assigned_lecturers) / len(temp_assigned_lecturers)
                    summary_data.append(("**平均頻度**", f"{avg_frequency:.1f}回"))

                    summary_data.append(("**一般資格ランク別割り当て**", "(講師が保有する一般資格ランク / 全講師中の同ランク保有者数)"))
                    general_rank_total_counts = {i: 0 for i in range(1, 6)}
                    for lecturer in st.session_state.DEFAULT_LECTURERS_DATA:
                        rank = lecturer.get("qualification_general_rank")
                        if rank in general_rank_total_counts:
                            general_rank_total_counts[rank] += 1
                    assigned_general_rank_counts = {i: 0 for i in range(1, 6)}
                    for l_assigned in temp_assigned_lecturers:
                        rank = l_assigned.get("qualification_general_rank")
                        if rank in assigned_general_rank_counts:
                            assigned_general_rank_counts[rank] += 1
                    for rank_num in range(1, 6):
                        summary_data.append((f"　一般ランク{rank_num}", f"{assigned_general_rank_counts.get(rank_num, 0)}人 / {general_rank_total_counts.get(rank_num, 0)}人中"))

                    summary_data.append(("**特別資格ランク別割り当て**", "(講師が保有する特別資格ランク / 全講師中の同ランク保有者数)"))
                    special_rank_total_counts = {i: 0 for i in range(1, 6)}
                    for lecturer in st.session_state.DEFAULT_LECTURERS_DATA:
                        rank = lecturer.get("qualification_special_rank")
                        if rank is not None and rank in special_rank_total_counts:
                            special_rank_total_counts[rank] += 1
                    assigned_special_rank_counts = {i: 0 for i in range(1, 6)}
                    for l_assigned in temp_assigned_lecturers:
                        rank = l_assigned.get("qualification_special_rank")
                        if rank is not None and rank in assigned_special_rank_counts:
                            assigned_special_rank_counts[rank] += 1
                    for rank_num in range(1, 6):
                        summary_data.append((f"　特別ランク{rank_num}", f"{assigned_special_rank_counts.get(rank_num, 0)}人 / {special_rank_total_counts.get(rank_num, 0)}人中"))

                if '今回の割り当て回数' in results_df.columns:
                    counts_of_lecturers_by_assignment_num = results_df['講師ID'].value_counts().value_counts().sort_index()
                    summary_data.append(("**講師の割り当て回数別**", "(今回の最適化での担当講座数)"))
                    for num_assignments, num_lecturers in counts_of_lecturers_by_assignment_num.items():
                        if num_assignments >= 1:
                            summary_data.append((f"　{num_assignments}回 担当した講師", f"{num_lecturers}人"))

                past_assignment_new_count = results_df[results_df["当該教室最終割当日からの日数"] < 0].shape[0]
                past_assignment_existing_count = results_df.shape[0] - past_assignment_new_count
                summary_data.append(("**同教室への過去の割り当て**", "(実績なし優先コスト計算に基づく)"))
                summary_data.append(("　新規", f"{past_assignment_new_count}人"))
                summary_data.append(("　割当て実績あり", f"{past_assignment_existing_count}人"))

                markdown_table = "| 項目 | 値 |\n| :---- | :---- |\n"
                for item, value in summary_data:
                    markdown_table += f"| {item} | {value} |\n"
                st.markdown(markdown_table)
                st.markdown("---")

                # --- 割り当て変更サマリーの表示 ---
                if "pending_change_summary_info" in st.session_state and st.session_state.pending_change_summary_info:
                    st.subheader("今回の割り当て変更による影響")
                    change_details_markdown = ""
                    for change_item in st.session_state.pending_change_summary_info:
                        original_lecturer_id = change_item['lecturer_id']
                        original_lecturer_name = change_item['lecturer_name']
                        course_id_changed = change_item['course_id']
                        course_name_changed = change_item['course_name']

                        new_assignment_for_course_df = results_df[results_df['講座ID'] == course_id_changed]

                        new_assignment_str = "割り当てなし"
                        if not new_assignment_for_course_df.empty:
                            new_lecturers_info = [f"{new_row['講師名']} (`{new_row['講師ID']}`)" for _, new_row in new_assignment_for_course_df.iterrows()]
                            new_assignment_str = "、".join(new_lecturers_info)

                        change_details_markdown += f"- **講座:** {course_name_changed} (`{course_id_changed}`)\n  - **変更前:** {original_lecturer_name} (`{original_lecturer_id}`)\n  - **変更後:** {new_assignment_str}\n"

                    if change_details_markdown:
                        st.markdown(change_details_markdown)
                    st.markdown("---")
                    del st.session_state.pending_change_summary_info # 表示後にクリア

                # --- 詳細結果と未割り当て講座 ---
                st.subheader("割り当て結果詳細")
                st.dataframe(results_df)
                st.markdown("---")

                if unassigned_courses:
                    st.subheader("割り当てられなかった講座")
                    st.dataframe(pd.DataFrame(unassigned_courses))
                    st.caption("上記の講座は、制約（資格ランクなど）により割り当て可能な講師が見つからなかったか、または他の割り当てと比較してコストが高すぎると判断された可能性があります。")

            else: # solver_result['assignments'] が存在しないか、空の場合
                if solver_result['raw_solver_status_code'] in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                    st.error("解が見つかりましたが、実際の割り当ては行われませんでした。")
                    st.warning(
                        "考えられる原因:\n"
                        "- 割り当て可能な講師と講座のペアが元々存在しない (制約が厳しすぎる、データ不適合)。\n"
                        "- 結果として、総コスト 0.00 (何も割り当てない) が最適と判断された可能性があります。"
                    )
                    st.subheader("全ての講座が割り当てられませんでした")
                    st.dataframe(pd.DataFrame(st.session_state.DEFAULT_COURSES_DATA))
                elif solver_result['raw_solver_status_code'] == cp_model.INFEASIBLE:
                    st.warning("指定された条件では、実行可能な割り当てが見つかりませんでした。制約やデータを見直してください。")
                else:
                    st.error(solver_result['solution_status_str'])
            # --- [リファクタリングここまで] ---

            if gemini_api_key:
                if st.button("Gemini API によるログ解説を実行", key="run_gemini_explanation_button"):
                    st.session_state.gemini_api_requested = True
                    if "gemini_explanation" in st.session_state: del st.session_state.gemini_explanation
                    if "gemini_api_error" in st.session_state: del st.session_state.gemini_api_error
                    st.rerun()

            # ログダウンロードセクション
            if st.session_state.get("solution_executed"):
                st.markdown("---") # 区切り線
                st.subheader("ログのダウンロード")
                dl_cols_1 = st.columns(2)
                dl_cols_2 = st.columns(2)

                with dl_cols_1[0]:
                    if st.session_state.get("optimization_gateway_log_for_download"):
                        st.download_button(
                            label="最適化ゲートウェイのログ",
                            data=st.session_state.optimization_gateway_log_for_download,
                            file_name="optimization_gateway.log",
                            mime="text/plain",
                            key="download_optimization_gateway_log_button",
                            help="データバリデーション、プロセス監視、最適化エンジン呼び出しに関するログです。"
                        )
                with dl_cols_1[1]:
                    if st.session_state.get("optimization_engine_log_for_download_from_file"):
                        st.download_button(
                            label="最適化エンジン内部ログ",
                            data=st.session_state.optimization_engine_log_for_download_from_file,
                            file_name="optimization_engine_internal.log",
                            mime="text/plain",
                            key="download_engine_internal_log_button", # key は変更なし
                            help="最適化エンジンの内部でキャプチャされた、割り当て候補のフィルタリングやコスト計算、制約構築などの詳細ログです。"
                        )
                with dl_cols_2[0]:
                    if st.session_state.get("solver_log_for_download"):
                        st.download_button(
                            label="OR-Toolsソルバーのログ",
                            data=st.session_state.solver_log_for_download,
                            file_name="solver_log.txt",
                            mime="text/plain",
                            key="download_solver_log_button",
                            # OR-Toolsソルバーのログは optimization_engine.py が直接 logger に出力するため、
                            # optimization_engine.log の一部として含まれる。
                            # ここでは純粋なソルバーログをダウンロードするボタンとして残す。
                            help="OR-Toolsソルバーが生成した、求解過程に関する技術的なログです。"
                        )
                with dl_cols_2[1]:
                    if st.session_state.get("app_log_for_download"):
                        st.download_button(
                            label="その他のシステムログ",
                            data=st.session_state.app_log_for_download,
                            file_name="app.log",
                            mime="text/plain",
                            key="download_app_log_button",
                            help="UI操作やアプリケーション全体の高レベルなイベントに関するログです。"
                        )

            if not gemini_api_key and st.session_state.get("solution_executed"):
                st.info("Gemini APIキーが設定されていません。ログ関連機能を利用するには設定が必要です。")
                logger.info("Gemini API key not set. Log features disabled.")

            if st.session_state.get("gemini_api_requested") and \
               "gemini_explanation" not in st.session_state and \
               "gemini_api_error" not in st.session_state:
                logger.info("Gemini API explanation requested. Calling API.")
                with st.spinner("Gemini API でログを解説中..."):
                    # Gemini API に渡すログは、ファイルから読み込んだものを結合して渡す
                    full_log_to_filter = st.session_state.app_log_for_download + st.session_state.optimization_gateway_log_for_download + st.session_state.optimization_engine_log_for_download_from_file
                    filtered_log_for_gemini = filter_log_for_gemini(full_log_to_filter)
                    solver_cache = st.session_state.solver_result_cache
                    solver_status = solver_cache["solution_status"]
                    objective_value = solver_cache["objective_value"]
                    assignments_list = solver_cache.get("assignments_df", [])
                    assignments_summary_df = pd.DataFrame(assignments_list) if assignments_list else None

                    explanation_or_error_text, full_prompt_for_gemini = get_gemini_explanation(
                        filtered_log_for_gemini, gemini_api_key,
                        solver_status, objective_value, assignments_summary_df
                    )
                    st.session_state.last_full_prompt_for_gemini = full_prompt_for_gemini # 常に保存

                    # APIキーエラー、またはGemini API自体のエラーをチェック
                    if explanation_or_error_text.startswith("エラー: Gemini API キーが設定されていません。") or \
                       explanation_or_error_text.startswith("Gemini APIエラー:"):
                        logger.error(f"Gemini related error: {explanation_or_error_text}")
                        st.session_state.gemini_api_error = explanation_or_error_text
                    else:
                        logger.info("Gemini API explanation processed (might include README load error in prompt).")
                        st.session_state.gemini_explanation = explanation_or_error_text
                        if "gemini_api_error" in st.session_state: del st.session_state.gemini_api_error
                    st.session_state.gemini_api_requested = False
                    st.rerun()

            if "gemini_api_error" in st.session_state and st.session_state.gemini_api_error:
                logger.info("Displaying Gemini API error.")
                st.error(st.session_state.gemini_api_error)
            elif "gemini_explanation" in st.session_state and st.session_state.gemini_explanation:
                logger.info("Displaying Gemini API explanation.")
                with st.expander("Gemini API によるログ解説", expanded=True):
                    st.markdown(st.session_state.gemini_explanation)

            # Gemini API送信後にプロンプト全体をダウンロードするボタン
            if "last_full_prompt_for_gemini" in st.session_state and st.session_state.last_full_prompt_for_gemini:
                st.download_button(
                    label="Gemini API送信用プロンプト全体のダウンロード",
                    data=st.session_state.last_full_prompt_for_gemini,
                    file_name="full_prompt_for_gemini.txt",
                    mime="text/plain",
                    key="download_full_prompt_button_after_api",
                    help="Gemini APIに送信されたプロンプト全体（システム仕様、最適化結果サマリー、ソルバーログを含む）です。"
                )

        logger.info("Optimization result display complete.")

def display_change_assignment_view():
    """「割り当ての変更」ビューを描画する"""
    logger = logging.getLogger('app')
    st.header("割り当ての変更")
    logger.info("Displaying change assignment view.")

    if not st.session_state.get("solution_executed", False) or \
       "solver_result_cache" not in st.session_state or \
       not st.session_state.solver_result_cache.get("assignments_df"):
        st.warning("割り当て結果が存在しないため、この機能は利用できません。まず最適化を実行してください。")
    else: # 割り当て結果が存在する場合
        solver_result = st.session_state.solver_result_cache
        results_df = pd.DataFrame(solver_result['assignments_df'])

        if results_df.empty:
            st.info("変更対象の割り当てがありません。")
        else:
            st.markdown("交代させたい講師の割り当てを選択し、「交代リスト」に追加してください。リスト作成後、「選択した割り当ての講師を変更して再最適化」ボタンで実行します。")

            # --- 検索フィルター ---
            st.subheader("割り当て検索フィルター")
            filter_cols = st.columns(3)
            with filter_cols[0]:
                search_lecturer_name = st.text_input("講師名で絞り込み", key="change_search_lecturer_name").lower()
            with filter_cols[1]:
                search_course_name = st.text_input("講座名で絞り込み", key="change_search_course_name").lower()
            with filter_cols[2]:
                search_classroom_name = st.text_input("教室名で絞り込み", key="change_search_classroom_name").lower()

            filtered_assignments_df = results_df.copy()
            if search_lecturer_name:
                filtered_assignments_df = filtered_assignments_df[filtered_assignments_df['講師名'].str.lower().str.contains(search_lecturer_name, na=False)]
            if search_course_name:
                filtered_assignments_df = filtered_assignments_df[filtered_assignments_df['講座名'].str.lower().str.contains(search_course_name, na=False)]
            if search_classroom_name: # 教室名で検索 (results_df に '教室名' がある前提)
                if '教室名' in filtered_assignments_df.columns:
                    filtered_assignments_df = filtered_assignments_df[filtered_assignments_df['教室名'].str.lower().str.contains(search_classroom_name, na=False)]
                else:
                    st.warning("結果データに教室名列が見つかりません。教室IDでの絞り込みを試みてください。")

            st.markdown("---")
            st.subheader("現在の割り当て一覧 (フィルター結果)")
            if filtered_assignments_df.empty:
                st.info("検索条件に一致する割り当てがありません。")
            else:
                for index, row in filtered_assignments_df.iterrows():
                    item_tuple = (
                        row['講師ID'], row['講座ID'],
                        row['講師名'], row['講座名'],
                        row['教室名'], row['スケジュール'] # 教室名とスケジュールもタプルに含める
                    )
                    is_selected = item_tuple in st.session_state.assignments_to_change_list

                    checkbox_label = f"講師: {row['講師名']} (`{row['講師ID']}`), 講座: {row['講座名']} (`{row['講座ID']}`), 教室: {row['教室名']} @ {row['スケジュール']}"

                    # チェックボックスの状態変更で直接リストを更新
                    if st.checkbox(checkbox_label, value=is_selected, key=f"cb_change_{row['講師ID']}_{row['講座ID']}"):
                        if not is_selected: # 以前選択されていなくて、今チェックされた
                            st.session_state.assignments_to_change_list.append(item_tuple)
                            # st.experimental_rerun() # 即時反映のため
                    else: # チェックが外された場合
                        if is_selected: # 以前選択されていて、今チェックが外された
                            st.session_state.assignments_to_change_list.remove(item_tuple)
                            # st.experimental_rerun() # 即時反映のため
                    st.markdown("---")

            # --- 交代リストの表示と管理 ---
            st.sidebar.markdown("---")
            st.sidebar.subheader("交代予定の割り当てリスト")
            if not st.session_state.assignments_to_change_list:
                st.sidebar.info("交代する割り当てはありません。")
            else:
                for i, item in enumerate(st.session_state.assignments_to_change_list):
                    # item: (lecturer_id, course_id, lecturer_name, course_name, classroom_name, schedule)
                    st.sidebar.markdown(f"- **講師:** {item[2]}, **講座:** {item[3]}")
                    if st.sidebar.button(f"リストから削除 ({item[2]}-{item[3]})", key=f"remove_change_{item[0]}_{item[1]}_{i}"):
                        st.session_state.assignments_to_change_list.pop(i)
                        st.rerun() # リスト変更を即時反映
                st.sidebar.markdown("---")

            # --- 変更実行ボタン ---
            if st.button("選択した割り当ての講師を変更して再最適化",
                           type="primary",
                           use_container_width=True,
                           disabled=not st.session_state.assignments_to_change_list,
                           on_click=handle_execute_changes_callback): # 新しいコールバックを呼ぶ
                pass # on_click で処理される
    logger.info("Change assignment view display complete.")

def main():
    # --- ロガーやデータ初期化など ---
    setup_logging()
    logger = logging.getLogger('app')
    st.set_page_config(page_title="講師割り当てシステムデモ", layout="wide")
    initialize_app_data() # 初回呼び出し (force_regenerate=False デフォルト)
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

    # --- セッション状態の初期化 ---
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "sample_data"
    if "assignments_to_change_list" not in st.session_state: # 交代リストの初期化
        st.session_state.assignments_to_change_list = []
    if "solution_executed" not in st.session_state:
        st.session_state.solution_executed = False
    if "allow_under_assignment_cb" not in st.session_state: # 追加: チェックボックスの初期値をセッションステートに設定
        st.session_state.allow_under_assignment_cb = True

    # --- UIの描画 ---
    st.title("講師割り当てシステム(OR-Tools)-プロトタイプ")

    # --- ナビゲーションボタン ---
    nav_cols = st.columns([2, 2, 2, 1])  # ボタン数を3つに合わせ、比率を調整
    with nav_cols[0]:
        if st.button("サンプルデータ", use_container_width=True, type="primary" if st.session_state.view_mode == "sample_data" else "secondary"):
            st.session_state.view_mode = "sample_data"
            st.rerun()
    with nav_cols[1]:
        if st.button("ソルバーとmodelオブジェクト", use_container_width=True, type="primary" if st.session_state.view_mode == "objective_function" else "secondary"):
            st.session_state.view_mode = "objective_function"
            st.rerun()
    with nav_cols[2]:
        if st.session_state.get("solution_executed", False):
            if st.button("最適化結果", use_container_width=True, type="primary" if st.session_state.view_mode == "optimization_result" else "secondary"):
                st.session_state.view_mode = "optimization_result"
                st.rerun()

    # --- サイドバー ---
    st.sidebar.markdown(
        "【制約】【許容条件】【最適化目標】を設定すれば、数理モデル最適化手法により自動的に最適な講師割り当てを実行します。"
        "また最適化目標に重み付けすることで割り当て結果をチューニングすることができます。"
    )
    st.sidebar.button("最適割り当てを実行", type="primary", on_click=run_optimization)
    st.sidebar.markdown("---")

    # New "割り当て結果を変更" (Change Assignment Results) button in the sidebar
    # This button appears only if optimization has run and produced assignments.
    if st.session_state.get("solution_executed", False) and \
       st.session_state.get("solver_result_cache") and \
       st.session_state.solver_result_cache.get("assignments_df"):
        if st.sidebar.button("割り当て結果を変更", key="change_assignment_view_button", type="secondary" if st.session_state.view_mode != "change_assignment_view" else "primary"):
            st.session_state.view_mode = "change_assignment_view"
            st.rerun()
    st.sidebar.markdown("---")

    
    with st.sidebar.expander("【制約】", expanded=False):
        st.markdown("- 1.講師は、資格ランクを超える講座への割り当てはできない") # 文言変更
        st.markdown("- 2.講師は、個人スケジュールに適合しない講座への割り当てはできない。") # 追加
        st.markdown("- 3.講師は、東京、名古屋、大阪の教室には2名を割り当て、それ以外には1名を割り当てる。") # 追加

    with st.sidebar.expander("【許容条件】", expanded=False): # 「ソフト制約」を「許容条件」に変更
        st.checkbox(
            "上記ハード制約3に対し、割り当て不足を許容する",
            key="allow_under_assignment_cb",
            help="チェックを入れると、東京・名古屋・大阪の教室は最大2名（0名または1名も可）、その他の教室は最大1名（0名も可）の割り当てとなります。チェックを外すと、必ず指定された人数（東京・名古屋・大阪は2名、他は1名）を割り当てようとします（担当可能な講師がいない場合は割り当てられません）。"
        )

    with st.sidebar.expander("【最適化目標】", expanded=False): # 名称変更
        st.caption(
            "各最適化目標の相対的な重要度を重みで設定します。\n"
            "不要な最適化目標は重みを0にしてください（最適化目標から除外されます）。"
        )
        st.markdown("**移動コストが低い人を優先**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど移動コストが低い人を重視します。", key="weight_travel_exp")
        st.markdown("**年齢の若い人を優先**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど年齢が若い人を重視します。", key="weight_age_exp")
        st.markdown("**割り当て頻度の少ない人を優先**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど全講座割当回数が少ない人を重視します。", key="weight_frequency_exp")
        st.markdown("**講師資格が高い人を優先**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど講師資格ランクが高い人が重視されます。", key="weight_qualification_exp")
        st.markdown("**同教室への割り当て実績が無い人を優先**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど同教室への割り当て実績が無い人、或いは最後に割り当てられた日からの経過日数が長い人が重視されます。", key="weight_past_assignment_exp")
        st.markdown("**割り当て不足を最小化**") # 新しい目的
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど、各講座の目標割り当て人数に対する不足を減らそうとします。「許容条件」で割り当て不足を許容している場合に有効です。", key="weight_assignment_shortage_exp")
        st.markdown("**講師の割り当て集中度を低くする（今回の割り当て内）**") # 新しい目的
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど、一人の講師が今回の最適化で複数の講座を担当することへのペナルティが大きくなります。", key="weight_lecturer_concentration_exp")

        st.markdown("**連日講座への連続割り当てを優先**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど、特別資格を持つ講師が一般講座と特別講座の連日ペアをまとめて担当することを重視します（報酬が増加）。", key="weight_consecutive_assignment_exp") # デフォルト値を0.5に変更

    # --- メインエリアの表示制御 ---
    logger.info(f"Starting main area display. Current view_mode: {st.session_state.view_mode}")
    if st.session_state.view_mode == "sample_data":
        display_sample_data_view()
    elif st.session_state.view_mode == "objective_function":
        display_objective_function_view()
    elif st.session_state.view_mode == "optimization_result":
        display_optimization_result_view(gemini_api_key=GEMINI_API_KEY)
    elif st.session_state.view_mode == "change_assignment_view":
        display_change_assignment_view()
    else: # view_mode が予期せぬ値の場合 (フォールバック)
        logger.warning(f"Unexpected view_mode: {st.session_state.view_mode}. Displaying fallback info.")
        st.info("サイドバーから表示するデータを選択してください。")
    logger.info("Exiting main function.")
if __name__ == "__main__":
    try:
        # Streamlit環境で安全にmultiprocessingを使用するため、'spawn'メソッドを強制的に設定
        # これはアプリケーションの起動時に一度だけ実行されるべき。
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Streamlitの内部的な再実行サイクルなどで既に設定されている場合があるので無視する
        pass
    main()
