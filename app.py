import streamlit as st
from ortools.sat.python import cp_model
# import streamlit.components.v1 as components # 削除
import pandas as pd
import io
import re # 正規表現モジュール
import datetime # 日付処理用に追加
import google.generativeai as genai # Gemini API 用
# from streamlit_oauth import OAuth2Component # OIDC認証用 # 削除
# from google.oauth2 import id_token # IDトークン検証用 # 削除
# from google.auth.transport import requests as google_requests # IDトークン検証用 # 削除
import random # データ生成用
import os # CPUコア数を取得するために追加
import numpy as np # データ型変換のために追加
# dateutil.relativedelta を使用するため、インストールが必要な場合があります。
# pip install python-dateutil
from dateutil.relativedelta import relativedelta
import logging # logging モジュールをインポート
from typing import TypedDict, List, Optional, Any, Tuple # 他のimport文と合わせて先頭に移動

# --- グローバル定数 (ログマーカー) ---
SOLVER_LOG_START_MARKER = "--- Solver Log (Captured by app.py) ---"
SOLVER_LOG_END_MARKER = "--- End Solver Log (Captured by app.py) ---"

# --- 1. データ定義 (LOG_EXPLANATIONS と _get_log_explanation は削除) ---

# --- Gemini API送信用ログのフィルタリング関数 (グローバルスコープに移動) ---
def filter_log_for_gemini(log_content: str) -> str:
    lines = log_content.splitlines()
    gemini_log_lines_final = [] # このリストには、処理されたソルバーログのみが格納されるようになります。
    
    solver_log_block = []
    app_summary_lines = [] # 詳細パターンに一致しない、かつソルバーブロック外のアプリログ
    app_detailed_lines_collected = [] # 詳細パターンに一致するアプリログ

    in_solver_log_block = False
    
    # 除外するアプリケーションログのパターン (詳細な割り当て試行ログ)
    detailed_app_log_patterns = [
        r"^\s*\+ Potential assignment:",
        r"^\s*- Filtered out:", # 資格ランクや過去教室の重複による除外
        r"^\s*Cost for ",
        r"^\s*- Schedule incompatible", # スケジュール不一致 (許容されるがログは出る)
        r"^\s*    Warning: Could not parse date", # 日付パースエラー
    ]
    
    # グローバル定数を参照
    # solver_log_start_marker = "--- Solver Log (Captured by app.py) ---" # 削除
    # solver_log_end_marker = "--- End Solver Log (Captured by app.py) ---" # 削除

    for line in lines:
        if SOLVER_LOG_START_MARKER in line: # 定数を参照
            in_solver_log_block = True
            solver_log_block.append(line)
            continue 
        
        if SOLVER_LOG_END_MARKER in line: # 定数を参照
            solver_log_block.append(line)
            in_solver_log_block = False
            continue

        if in_solver_log_block:
            solver_log_block.append(line)
        else: # Application log
            is_detailed = any(re.search(pattern, line) for pattern in detailed_app_log_patterns)
            if is_detailed:
                app_detailed_lines_collected.append(line)
            else:
                app_summary_lines.append(line)
    
    # アプリケーションログ（サマリーおよび詳細）は gemini_log_lines_final に追加されなくなります。

    # Gemini用に solver_log_block のみを処理します
    # ソルバーログの切り詰め処理を削除し、solver_log_block全体をそのまま使用します。
    gemini_log_lines_final.extend(solver_log_block)
    
    return "\n".join(gemini_log_lines_final)

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
class SolverOutput(TypedDict): # 提案: 戻り値を構造化するための型定義
    solution_status_str: str
    objective_value: Optional[float] # None の可能性を明示
    assignments: List[dict]
    all_courses: List[dict]
    all_lecturers: List[dict]
    solver_raw_status_code: int
    full_application_and_solver_log: str
    pure_solver_log: str # 純粋なソルバーログ（マーカーなし）
    application_log: str # アプリケーションログ

def solve_assignment(lecturers_data, courses_data, classrooms_data, # classrooms_data は現在未使用だが、将来のために残す
                     travel_costs_matrix,
                     weight_past_assignment_recency, weight_qualification,
                     weight_travel, weight_age, weight_frequency, # 既存の重み
                     weight_assignment_shortage, # 追加: 割り当て不足ペナルティの重み
                     weight_lecturer_concentration, # 追加: 講師割り当て集中ペナルティの重み
                     weight_consecutive_assignment, # 追加: 連日割り当て報酬の重み
                     allow_under_assignment: bool, # 割り当て不足を許容するかのフラグ
                     today_date, # デフォルト値を持つ引数の前に移動
                     fixed_assignments: Optional[List[Tuple[str, str]]] = None, # ピン留めする割り当て
                     forced_unassignments: Optional[List[Tuple[str, str]]] = None) -> SolverOutput: # 強制的に割り当てないペア
    model = cp_model.CpModel()

    # --- 1. データ前処理: リストをIDをキーとする辞書に変換 ---
    lecturers_dict = {lecturer['id']: lecturer for lecturer in lecturers_data}
    courses_dict = {course['id']: course for course in courses_data}
    classrooms_dict = {classroom['id']: classroom for classroom in classrooms_data} # 教室データも辞書に変換

    # --- ログキャプチャ用の StringIO ---
    app_log_stream = io.StringIO() # アプリケーションログ用
    solver_capture_stream = io.StringIO() # ソルバーログキャプチャ用 (以前の full_log_stream)
    logger = logging.getLogger(__name__) # solve_assignment内でロガーを取得

    # アプリケーションログ出力関数 (標準ロガーとapp_log_streamに出力)
    def log_to_stream(message):
        log_line = f"[SolverAppLog] {message}\n"
        logger.info(log_line.strip()) # 標準ロガーには改行なしで
        app_log_stream.write(log_line) # app_log_stream には改行ありでキャプチャ

    # --- ステップ1: 連日講座ペアのリストアップ ---
    consecutive_day_pairs = []
    log_to_stream("Starting search for consecutive general-special course pairs.")
    parsed_courses_for_pairing = []
    for course_id_loop, course_item_loop in courses_dict.items(): # courses_data の代わりに courses_dict.values() を使用
        try:
            schedule_date = datetime.datetime.strptime(course_item_loop['schedule'], "%Y-%m-%d").date()
            # courses_dict の値を直接変更せず、新しいリストに情報を追加
            parsed_courses_for_pairing.append({**course_item_loop, 'schedule_date_obj': schedule_date})
        except ValueError:
            log_to_stream(f"  Warning: Could not parse schedule date {course_item_loop['schedule']} for course {course_item_loop['id']} during pair finding. Skipping.")
            continue

    courses_by_classroom_for_pairing = {}
    for course_in_list in parsed_courses_for_pairing:
        cid = course_in_list['classroom_id']
        if cid not in courses_by_classroom_for_pairing:
            courses_by_classroom_for_pairing[cid] = []
        courses_by_classroom_for_pairing[cid].append(course_in_list)

    for cid_loop, classroom_courses_list in courses_by_classroom_for_pairing.items():
        classroom_courses_list.sort(key=lambda c: c['schedule_date_obj'])
        for i in range(len(classroom_courses_list) - 1):
            course1 = classroom_courses_list[i]
            course2 = classroom_courses_list[i+1]
            
            is_general_special_pair = (course1['course_type'] == 'general' and course2['course_type'] == 'special') or \
                                      (course1['course_type'] == 'special' and course2['course_type'] == 'general')
            
            if is_general_special_pair and (course2['schedule_date_obj'] - course1['schedule_date_obj']).days == 1:
                pair_c1_obj, pair_c2_obj = course1, course2 # 日付順
                consecutive_day_pairs.append({
                    "pair_id": f"CDP_{pair_c1_obj['id']}_{pair_c2_obj['id']}",
                    "course1_id": pair_c1_obj['id'], "course2_id": pair_c2_obj['id'],
                    "classroom_id": cid_loop
                })
                log_to_stream(f"  + Found consecutive day pair: {pair_c1_obj['id']} ({pair_c1_obj['schedule']}) and {pair_c2_obj['id']} ({pair_c2_obj['schedule']}) at {cid_loop}")


    # --- Main logic for model building and solving ---
    # possible_assignments_temp_data: 生のコストと変数などを一時的に格納
    possible_assignments_temp_data = {}
    potential_assignment_count = 0
    log_to_stream(f"Initial lecturers: {len(lecturers_data)}, Initial courses: {len(courses_data)}")

    # 強制的に割り当てないペアをセットに変換して高速検索
    forced_unassignments_set = set(forced_unassignments) if forced_unassignments else set()
    if forced_unassignments_set:
        log_to_stream(f"  Forced unassignments specified: {forced_unassignments_set}")

    # 実績なし優先コストの逆数モデル用定数
    RECENCY_COST_CONSTANT = 100000.0



    # lecturers_data と courses_data の代わりに、事前に作成した辞書の値を反復処理
    for lecturer_id_loop, lecturer in lecturers_dict.items(): # lecturers_data の代わりに lecturers_dict.values() を使用
        for course_id_loop, course in courses_dict.items():   # courses_data の代わりに courses_dict.values() を使用
            lecturer_id = lecturer["id"]
            course_id = course["id"]

            # 強制的に割り当てないペアかチェック
            if (lecturer_id, course_id) in forced_unassignments_set:
                log_to_stream(f"  - Filtered out (forced unassignment): {lecturer_id} for {course_id}")
                continue


            # 新しい資格ランクチェックロジック
            course_type = course["course_type"]
            course_rank = course["rank"]
            lecturer_general_rank = lecturer["qualification_general_rank"]
            lecturer_special_rank = lecturer.get("qualification_special_rank") # None の可能性あり

            can_assign_by_qualification = False
            qualification_cost_for_this_assignment = 0

            if course_type == "general":
                if lecturer_special_rank is not None: # 特別資格持ちは一般講座OK
                    can_assign_by_qualification = True
                    qualification_cost_for_this_assignment = lecturer_general_rank # コストは一般ランクで
                elif lecturer_general_rank <= course_rank: # 一般資格のみの場合、ランク比較
                    can_assign_by_qualification = True
                    qualification_cost_for_this_assignment = lecturer_general_rank
            elif course_type == "special":
                if lecturer_special_rank is not None and lecturer_special_rank <= course_rank:
                    can_assign_by_qualification = True
                    qualification_cost_for_this_assignment = lecturer_special_rank
            
            if not can_assign_by_qualification:
                log_to_stream(f"  - Filtered out: {lecturer_id} for {course_id} (Qualification insufficient. Course: {course_type} Rank {course_rank}. Lecturer: GenRank {lecturer_general_rank}, SpecRank {lecturer_special_rank})")
                continue

            # スケジュールチェック
            schedule_available = course["schedule"] in lecturer["availability"]
            if not schedule_available:
                log_to_stream(f"  - Filtered out: {lecturer_id} for {course_id} (Schedule unavailable: Course_schedule={course['schedule']}, Lecturer_avail_sample={lecturer['availability'][:3]}...)")
                continue
            # else: スケジュールが合う場合はペナルティなし

            potential_assignment_count += 1
            log_to_stream(f"  + Potential assignment: {lecturer_id} to {course_id} on {course['schedule']}")
            var = model.NewBoolVar(f'x_{lecturer_id}_{course_id}')
            
            travel_cost = travel_costs_matrix.get((lecturer["home_classroom_id"], course["classroom_id"]), 999)
            age_cost = float(lecturer.get("age", 99)) # 実年齢をコストとして使用。未設定の場合は大きな値。
            # 実際の過去の総割り当て回数を頻度コストとする (少ないほど良い)
            frequency_cost = float(len(lecturer.get("past_assignments", [])))
            qualification_cost = float(qualification_cost_for_this_assignment) # 上で計算した、この割り当てにおける資格コスト

            # 過去割り当ての近さによるコスト計算
            past_assignment_recency_cost = 0.0 # 実績なしの場合はコスト0
            days_since_last_assignment_to_classroom = -1 # 実績なしを示す値 (-1 や None など)

            if lecturer.get("past_assignments"):
                relevant_past_assignments_to_this_classroom = [
                    pa for pa in lecturer["past_assignments"]
                    if pa["classroom_id"] == course["classroom_id"]
                ]
                if relevant_past_assignments_to_this_classroom:
                    # past_assignments は日付降順ソート済みなので、リストの最初のものが最新の割り当て
                    latest_assignment_date_str = relevant_past_assignments_to_this_classroom[0]["date"]
                    try:
                        latest_assignment_date = datetime.datetime.strptime(latest_assignment_date_str, "%Y-%m-%d").date() # type: ignore
                        days_since_last_assignment_to_classroom = (today_date - latest_assignment_date).days
                        # 逆数モデル: 定数 / (経過日数 + 1)
                        # 経過日数が0 (昨日) の場合、コストは RECENCY_COST_CONSTANT / 1
                        # 経過日数が大きいほどコストは小さくなる
                        if days_since_last_assignment_to_classroom >= 0: # 念のため確認
                            past_assignment_recency_cost = RECENCY_COST_CONSTANT / (days_since_last_assignment_to_classroom + 1.0)
                        else: # 通常ありえないが、未来の日付など
                            past_assignment_recency_cost = 0.0
                            days_since_last_assignment_to_classroom = -2 # パースは成功したが日付がおかしい場合
                    except ValueError:
                        log_to_stream(f"    Warning: Could not parse date '{latest_assignment_date_str}' for {lecturer_id} and classroom {course['classroom_id']}")
                        past_assignment_recency_cost = 0.0 # パース失敗時はコスト0
                        days_since_last_assignment_to_classroom = -3 # パース失敗を示す
            else: # 過去の割り当て履歴自体がない場合
                past_assignment_recency_cost = 0.0
                days_since_last_assignment_to_classroom = -1 # 実績なし

            log_to_stream(f"    Raw costs for {lecturer_id} to {course_id}: travel={travel_cost}, age={age_cost}, freq={frequency_cost}, qual={qualification_cost}, recency={past_assignment_recency_cost:.2f} (days_since_last={days_since_last_assignment_to_classroom})")

            # 一時辞書に生のコストと変数を格納
            assignment_key = (lecturer_id, course_id)
            possible_assignments_temp_data[assignment_key] = {
                "lecturer_id": lecturer_id, "course_id": course_id,
                "variable": var,
                "raw_costs": {
                    "travel": travel_cost, "age": age_cost, "frequency": frequency_cost,
                    "qualification": qualification_cost, "recency": past_assignment_recency_cost
                },
                "debug_days_since_last_assignment": days_since_last_assignment_to_classroom
            }

    log_to_stream(f"Total potential assignments after filtering: {potential_assignment_count}")
    log_to_stream(f"Number of entries in possible_assignments_temp_data: {len(possible_assignments_temp_data)}")

    if not possible_assignments_temp_data:
        log_to_stream("No possible assignments found after filtering. Optimization will likely result in no assignments.")
        captured_app_log_early = app_log_stream.getvalue()
        # ソルバーは実行されていないので、ソルバーログ関連は空
        return SolverOutput(
            solution_status_str="前提条件エラー (割り当て候補なし)",
            objective_value=None,
            assignments=[],
            all_courses=list(courses_dict.values()), # 辞書の値を使用
            all_lecturers=list(lecturers_dict.values()), # 辞書の値を使用
            solver_raw_status_code=cp_model.UNKNOWN, 
            full_application_and_solver_log=captured_app_log_early, # アプリログのみ
            pure_solver_log="",
            application_log=captured_app_log_early
        )
    # --- 動的正規化係数の計算 ---
    def get_norm_factor(cost_list, name):
        if not cost_list: return 1.0
        avg = np.mean(cost_list)
        factor = avg if avg > 1e-9 else 1.0 # 0除算や非常に小さい値での除算を避ける
        log_to_stream(f"  Normalization factor for {name}: {factor:.4f} (avg: {avg:.4f}, count: {len(cost_list)})")
        return factor

    norm_factor_travel = get_norm_factor([d["raw_costs"]["travel"] for d in possible_assignments_temp_data.values() if "travel" in d["raw_costs"]], "travel")
    norm_factor_age = get_norm_factor([d["raw_costs"]["age"] for d in possible_assignments_temp_data.values() if "age" in d["raw_costs"]], "age")
    norm_factor_frequency = get_norm_factor([d["raw_costs"]["frequency"] for d in possible_assignments_temp_data.values() if "frequency" in d["raw_costs"]], "frequency")
    norm_factor_qualification = get_norm_factor([d["raw_costs"]["qualification"] for d in possible_assignments_temp_data.values() if "qualification" in d["raw_costs"]], "qualification")
    norm_factor_recency = get_norm_factor([d["raw_costs"]["recency"] for d in possible_assignments_temp_data.values() if "recency" in d["raw_costs"]], "recency")

    # --- 最終的なコスト計算と possible_assignments_dict の構築 ---
    possible_assignments_dict = {}
    for key, temp_data in possible_assignments_temp_data.items():
        raw = temp_data["raw_costs"]
        
        norm_travel = raw["travel"] / norm_factor_travel
        norm_age = raw["age"] / norm_factor_age
        norm_frequency = raw["frequency"] / norm_factor_frequency
        norm_qualification = raw["qualification"] / norm_factor_qualification
        norm_recency = raw["recency"] / norm_factor_recency

        total_weighted_cost_float = (
            weight_travel * norm_travel +
            weight_age * norm_age +
            weight_frequency * norm_frequency +
            weight_qualification * norm_qualification +
            weight_past_assignment_recency * norm_recency
        )
        total_weighted_cost_int = int(total_weighted_cost_float * 100) # スケーリング
        log_to_stream(f"    Final cost for {key[0]}-{key[1]}: total_weighted_int={total_weighted_cost_int} (norm_travel={norm_travel:.2f}, norm_age={norm_age:.2f}, norm_freq={norm_frequency:.2f}, norm_qual={norm_qualification:.2f}, norm_recency={norm_recency:.2f})")
        possible_assignments_dict[key] = {**temp_data, "cost": total_weighted_cost_int}

    # --- 事前に割り当て変数をグループ化 ---
    assignments_by_course = {}
    for (lecturer_id_group, course_id_group), data_group in possible_assignments_dict.items():
        variable_group = data_group["variable"]        
        if course_id_group not in assignments_by_course:
            assignments_by_course[course_id_group] = []
        assignments_by_course[course_id_group].append(variable_group)
    
    # 講師IDで割り当て変数をグループ化 (講師集中ペナルティ用)
    assignments_by_lecturer = {lect_id: [] for lect_id in lecturers_dict}
    for (lecturer_id_group, course_id_group), data_group in possible_assignments_dict.items():
        # variable_group は既に上で取得済みなので再利用はできないが、ロジックは同じ
        assignments_by_lecturer[lecturer_id_group].append(data_group["variable"])

    # 特定の都道府県リスト
    TARGET_PREFECTURES_FOR_TWO_LECTURERS = ["東京都", "愛知県", "大阪府"]
    
    shortage_penalty_terms = [] # 割り当て不足ペナルティ項を格納するリスト

    for course_item in courses_dict.values(): # courses_data の代わりに courses_dict.values() を使用
        course_id = course_item["id"]
        possible_assignments_for_course = assignments_by_course.get(course_id, [])
        if possible_assignments_for_course: # 担当可能な講師候補がいる場合のみ制約を追加
            course_classroom_id = course_item["classroom_id"]
            course_location = classrooms_dict[course_classroom_id]["location"]

            target_assignment_count = 1
            if course_location in TARGET_PREFECTURES_FOR_TWO_LECTURERS:
                target_assignment_count = 2
            
            if allow_under_assignment:
                # 割り当て不足を許容する場合 (0名、1名、または target_assignment_count 名まで)
                model.Add(sum(possible_assignments_for_course) <= target_assignment_count)
                # --- 割り当て不足ペナルティの計算 ---
                if weight_assignment_shortage > 0:
                    # 不足数を表す変数 shortage_var = max(0, target_assignment_count - sum(vars))
                    # shortage_var >= target_assignment_count - sum(vars)
                    # shortage_var >= 0 (IntVarの定義でカバー)
                    shortage_var = model.NewIntVar(0, target_assignment_count, f'shortage_var_{course_id}')
                    model.Add(shortage_var >= target_assignment_count - sum(possible_assignments_for_course))
                    
                    # ペナルティコストのスケールを他のコストと合わせることを検討
                    # 他のコストは *100 されている。ペナルティのベース値もそれに合わせる。
                    # 例: 1件不足で基本コスト50,000 (スケーリング前)、重み1.0なら目的関数値 5,000,000
                    base_penalty_shortage_scaled = 50000 * 100 # 500万
                    actual_penalty_for_shortage = int(weight_assignment_shortage * base_penalty_shortage_scaled)
                    if actual_penalty_for_shortage > 0:
                        shortage_penalty_terms.append(shortage_var * actual_penalty_for_shortage)
                        log_to_stream(f"  + Course {course_id}: Added shortage penalty term (shortage_var * {actual_penalty_for_shortage}) for target {target_assignment_count}.")
            else:
                # 割り当て不足を許容しない場合 (必ず target_assignment_count 名)
                model.Add(sum(possible_assignments_for_course) == target_assignment_count)

    # --- 目的関数の構築 ---
    # possible_assignments_dict の値からコスト項を生成
    objective_terms = [data["variable"] * data["cost"] for data in possible_assignments_dict.values()] # type: ignore

    # 割り当て不足ペナルティ項を目的関数に追加
    if shortage_penalty_terms:
        objective_terms.extend(shortage_penalty_terms)
        log_to_stream(f"  + Added {len(shortage_penalty_terms)} shortage penalty terms to objective.")

    # --- 講師の割り当て集中ペナルティ ---
    if weight_lecturer_concentration > 0:
        # 1回の追加割り当てに対する基本ペナルティコスト (スケーリング後)
        # 例: 1件超過で基本コスト20,000 (スケーリング前)、重み1.0なら目的関数値 2,000,000
        base_penalty_concentration_scaled = 20000 * 100 # 200万
        actual_penalty_concentration = int(weight_lecturer_concentration * base_penalty_concentration_scaled)

        if actual_penalty_concentration > 0:
            for lecturer_id_loop, lecturer_vars in assignments_by_lecturer.items():
                if not lecturer_vars or len(lecturer_vars) <= 1: # 割り当て候補がないか、1つ以下なら集中ペナルティ不要
                    continue
                
                num_total_assignments_l = model.NewIntVar(0, len(courses_dict), f'num_total_assignments_{lecturer_id_loop}')
                model.Add(num_total_assignments_l == sum(lecturer_vars))
                
                # 1回を超える割り当て数を表す変数 extra_assignments_l = max(0, num_total_assignments_l - 1)
                extra_assignments_l = model.NewIntVar(0, len(courses_dict), f'extra_assign_{lecturer_id_loop}')
                model.Add(extra_assignments_l >= num_total_assignments_l - 1)
                # extra_assignments_l >= 0 はIntVarの定義でカバー
                
                objective_terms.append(extra_assignments_l * actual_penalty_concentration)
                log_to_stream(f"  + Lecturer {lecturer_id_loop}: Added concentration penalty term (extra_assign * {actual_penalty_concentration}).")
    
    # --- ステップ3: 連日割り当ての報酬と制約 ---
    consecutive_assignment_pair_vars_details = [] # ペア割り当て変数とその詳細を格納
    if weight_consecutive_assignment > 0 and consecutive_day_pairs:
        log_to_stream(f"Processing {len(consecutive_day_pairs)} consecutive day pairs for special assignment reward.")
        for pair_info in consecutive_day_pairs:
            pair_id = pair_info["pair_id"]
            c1_id = pair_info["course1_id"]
            c2_id = pair_info["course2_id"]
            # course1_obj = courses_dict[c1_id] # courses_dict から取得
            # course2_obj = courses_dict[c2_id] # courses_dict から取得

            for lecturer_id_loop_pair, lecturer_pair in lecturers_dict.items():
                # 1. 講師が特別資格を持っているか
                if lecturer_pair.get("qualification_special_rank") is None:
                    continue

                # 2. 講師が両方の講座の資格要件とスケジュールを満たすか (possible_assignments_dict で確認)
                key1 = (lecturer_id_loop_pair, c1_id)
                key2 = (lecturer_id_loop_pair, c2_id)
                if key1 not in possible_assignments_dict or key2 not in possible_assignments_dict:
                    continue

                log_to_stream(f"  + Potential consecutive pair assignment: Lecturer {lecturer_id_loop_pair} for pair {pair_id} ({c1_id}, {c2_id})")
                pair_var_name = f"y_{lecturer_id_loop_pair}_{pair_id}"
                pair_var = model.NewBoolVar(pair_var_name)
                consecutive_assignment_pair_vars_details.append({
                    "variable": pair_var, "lecturer_id": lecturer_id_loop_pair,
                    "course1_id": c1_id, "course2_id": c2_id, "pair_id": pair_id
                })

                # 関連付け制約: pair_var = 1 => x_lecturer_c1 = 1 AND x_lecturer_c2 = 1
                individual_var_c1 = possible_assignments_dict[key1]["variable"]
                individual_var_c2 = possible_assignments_dict[key2]["variable"]
                model.Add(pair_var <= individual_var_c1) # If pair_var is true, individual_var_c1 must be true
                model.Add(pair_var <= individual_var_c2) # If pair_var is true, individual_var_c2 must be true
                # (ソルバーは報酬のために pair_var を True にしようとし、この制約が個別の割り当ても True にする)

                # 目的関数への追加 (報酬)
                # 集中ペナルティ (例:200万) を考慮し、それを上回る可能性のある報酬を設定
                base_reward_consecutive_scaled = 30000 * 100 # 例: 300万 (スケーリング後)
                actual_reward_for_pair = int(weight_consecutive_assignment * base_reward_consecutive_scaled)
                
                if actual_reward_for_pair > 0:
                    objective_terms.append(pair_var * -actual_reward_for_pair) # 最小化なので負のコスト
                    log_to_stream(f"    Added reward {-actual_reward_for_pair} for pair_var {pair_var_name}")
    else:
        if weight_consecutive_assignment > 0:
            log_to_stream("No consecutive day pairs found, or weight_consecutive_assignment is zero. Skipping reward logic.")

    # --- ステップ4: ピン留めされた割り当ての制約追加 ---
    if fixed_assignments:
        log_to_stream(f"Processing {len(fixed_assignments)} fixed assignments (pinning).")
        for fixed_lect_id, fixed_course_id in fixed_assignments:
            assignment_key = (fixed_lect_id, fixed_course_id)
            if assignment_key in possible_assignments_dict:
                var_to_pin = possible_assignments_dict[assignment_key]["variable"]
                model.Add(var_to_pin == 1)
                log_to_stream(f"  + Pinned assignment: {fixed_lect_id} to {fixed_course_id} (variable {var_to_pin.Name()} forced to 1).")
            else:
                # ピン留めしようとした割り当てが、そもそも候補にない（資格がない、スケジュールが合わない、または forced_unassignments で除外された）
                # この場合、実行不可能な問題になる可能性が高い
                log_to_stream(f"  WARNING: Attempted to pin assignment ({fixed_lect_id}, {fixed_course_id}) but it's not a possible assignment. This may lead to an INFEASIBLE solution.")
    else:
        log_to_stream("No fixed assignments specified.")

    if weight_consecutive_assignment > 0: # 修正: このログは以前のブロックから移動
        if weight_consecutive_assignment > 0:
            log_to_stream("No consecutive day pairs found, or weight_consecutive_assignment is zero. Skipping reward logic.")


    if objective_terms:
        model.Minimize(sum(objective_terms))
    else:
        # This case should ideally not be reached if 'possible_assignments' was non-empty,
        # as 'assignment_costs' would initialize 'objective_terms'.
        # However, if it is reached, minimizing 0 is valid, and the solver should still run.
        log_to_stream("Warning: Objective terms list was empty. Minimizing 0.")
        model.Minimize(0) 
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True # ソルバーのログ出力を有効化
    # --- ソルバーの並列処理有効化 (一旦コメントアウトして安定性を確認) ---
    # num_workers = os.cpu_count()
    # if num_workers: 
    #     solver.parameters.num_search_workers = num_workers
    #     log_to_stream(f"Solver configured to use {num_workers} workers (CPU cores).")

    # solver.log_callback を使用してソルバーログをキャプチャ
    solver.log_callback = lambda msg: solver_capture_stream.write(msg + "\n") # メッセージごとに改行を追加

    # ソルバーログの開始マーカーを solver_capture_stream に直接書き込む
    solver_capture_stream.write(f"{SOLVER_LOG_START_MARKER}\n")
    
    status_code = cp_model.UNKNOWN # Initialize status_code
    status_code = solver.Solve(model) # コールバックインスタンスは渡さない

    # ソルバーログの終了マーカーを solver_capture_stream に直接書き込む
    solver_capture_stream.write(f"\n{SOLVER_LOG_END_MARKER}\n") # 前後に改行を追加して区切りを明確に

    captured_solver_log_with_markers = solver_capture_stream.getvalue()
    captured_app_log = app_log_stream.getvalue()

    # 純粋なソルバーログ（マーカーなし）を抽出
    pure_solver_lines = []
    capturing_solver_log = False
    # captured_solver_log_with_markers から抽出する
    for line in captured_solver_log_with_markers.splitlines(keepends=False):
        if line == SOLVER_LOG_START_MARKER:
            capturing_solver_log = True
            continue # マーカー自体は含めない
        if line == SOLVER_LOG_END_MARKER:
            capturing_solver_log = False
            break # マーカー自体は含めない
        if capturing_solver_log:
            pure_solver_lines.append(line)
    
    pure_solver_log_content = "\n".join(pure_solver_lines)
    if pure_solver_lines: # 何か行があれば最後に改行を追加
        pure_solver_log_content += "\n"

    # Gemini送信用・rawログ保存用の結合ログ
    full_application_and_solver_log_content = captured_app_log + captured_solver_log_with_markers

    status_name = solver.StatusName(status_code) # Get the status name
    results = []
    objective_value = None
    solution_status_str = "解なし"

    if status_code == cp_model.OPTIMAL or status_code == cp_model.FEASIBLE:
        # solution_status_str と objective_value はこのブロック内で設定
        solution_status_str = "最適解" if status_code == cp_model.OPTIMAL else "実行可能解" # type: ignore
        objective_value = solver.ObjectiveValue() / 100 # スケーリングを戻す
        
        # まず、今回の割り当てで各講師が何回割り当てられたかを計算
        lecturer_assignment_counts_this_round = {}
        for pa_data_count_check in possible_assignments_dict.values():
            if solver.Value(pa_data_count_check["variable"]) == 1: # type: ignore
                lecturer_id_for_count = pa_data_count_check["lecturer_id"]
                lecturer_assignment_counts_this_round[lecturer_id_for_count] = \
                    lecturer_assignment_counts_this_round.get(lecturer_id_for_count, 0) + 1

        # 解決された連続ペア割り当てをマップに格納
        solved_consecutive_assignments_map = {} # key: (lecturer_id, course_id), value: pair_id
        if weight_consecutive_assignment > 0 and consecutive_assignment_pair_vars_details:
            for pair_detail in consecutive_assignment_pair_vars_details:
                if solver.Value(pair_detail["variable"]) == 1: # type: ignore
                    lect_id = pair_detail["lecturer_id"]
                    c1_id_res = pair_detail["course1_id"]
                    c2_id_res = pair_detail["course2_id"]
                    p_id_res = pair_detail["pair_id"]
                    solved_consecutive_assignments_map[(lect_id, c1_id_res)] = p_id_res
                    solved_consecutive_assignments_map[(lect_id, c2_id_res)] = p_id_res
                    log_to_stream(f"  Confirmed consecutive assignment for L:{lect_id} on Pair:{p_id_res} (C1:{c1_id_res}, C2:{c2_id_res})")
        # possible_assignments_dict を反復処理して結果を構築
        for (lecturer_id_res, course_id_res), pa_data in possible_assignments_dict.items():
            if solver.Value(pa_data["variable"]) == 1: # type: ignore
                lecturer = lecturers_dict[lecturer_id_res] # 事前処理した辞書から取得
                course = courses_dict[course_id_res] # 事前処理した辞書から取得
                results.append({
                    # ... (他のフィールドは変更なし)
                    "講師ID": lecturer["id"],
                    "講師名": lecturer["name"],
                    "講座ID": course["id"],
                    "講座名": course["name"],
                    "教室ID": course["classroom_id"],
                    "スケジュール": course['schedule'],
                    "算出コスト(x100)": pa_data["cost"],
                    "教室名": classrooms_dict.get(course["classroom_id"], {}).get("location", "不明"), # 教室名を追加
                    "移動コスト(元)": pa_data["raw_costs"]["travel"],
                    "年齢コスト(元)": pa_data["raw_costs"]["age"],
                    "頻度コスト(元)": pa_data["raw_costs"]["frequency"],
                    "資格コスト(元)": pa_data["raw_costs"]["qualification"],
                    "当該教室最終割当日からの日数": pa_data["debug_days_since_last_assignment"], # これはそのまま
                    "講師一般ランク": lecturer.get("qualification_general_rank"),
                    "講師特別ランク": lecturer.get("qualification_special_rank", "なし"),
                    "講座タイプ": course.get("course_type"),
                    "講座ランク": course.get("rank"),
                    "今回の割り当て回数": lecturer_assignment_counts_this_round.get(lecturer["id"], 0),
                    "連続ペア割当": solved_consecutive_assignments_map.get((lecturer["id"], course["id"]), "なし") # 元の計算ロジックを復元
                })
    elif status_code == cp_model.INFEASIBLE:
        solution_status_str = "実行不可能 (制約を満たす解なし)"
    else:
        solution_status_str = f"解探索失敗 (ステータス: {status_name} [{status_code}])" # Include name and code

    return SolverOutput(
        solution_status_str=solution_status_str,
        objective_value=objective_value,
        assignments=results,
        all_courses=list(courses_dict.values()), # courses_data の代わりに辞書の値を渡す
        all_lecturers=list(lecturers_dict.values()), # lecturers_data の代わりに辞書の値を渡す
        solver_raw_status_code=status_code,
        full_application_and_solver_log=full_application_and_solver_log_content,
        pure_solver_log=pure_solver_log_content,
        application_log=captured_app_log
    )

# --- 3. Streamlit UI ---
def initialize_app_data(force_regenerate: bool = False):
    """
    アプリケーションの初期データを生成し、セッション状態に保存する。
    force_regenerate=True の場合、既存のデータがあっても強制的に再生成する。
    """
    logger = logging.getLogger(__name__)
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

def main():
    # --- ロガーやデータ初期化など ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    st.set_page_config(page_title="講師割り当てシステムデモ", layout="wide")
    initialize_app_data() # 初回呼び出し (force_regenerate=False デフォルト)
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

    # --- コールバック関数の定義 ---
    def handle_regenerate_sample_data():
        logger.info("Regenerate sample data button clicked, callback triggered.")
        initialize_app_data(force_regenerate=True)
        st.session_state.show_regenerate_success_message = True # メッセージ表示用フラグ

    def run_optimization():
        """最適化を実行し、結果をセッション状態に保存するコールバック関数"""
        keys_to_clear_on_execute = [
            "solver_result_cache", "raw_log_on_server", 
            "solver_log_for_download", "optimization_error_message", # solver_log_for_download を追加
            "application_log_for_download", # 追加
            "gemini_explanation", "gemini_api_requested", "gemini_api_error", "last_full_prompt_for_gemini"
        ]
        for key in keys_to_clear_on_execute:
            if key in st.session_state:
                del st.session_state[key]
        logger.info("Cleared previous optimization results from session_state.")

        try:
            logger.info("Starting optimization calculation (solve_assignment).")
            with st.spinner("最適化計算を実行中..."):
                solver_output = solve_assignment(
                    st.session_state.DEFAULT_LECTURERS_DATA, st.session_state.DEFAULT_COURSES_DATA,
                    st.session_state.DEFAULT_CLASSROOMS_DATA, st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX,
                    st.session_state.get("weight_past_assignment_exp", 0.5),
                    st.session_state.get("weight_qualification_exp", 0.5),
                    st.session_state.get("weight_travel_exp", 0.5),
                    st.session_state.get("weight_age_exp", 0.5),
                    st.session_state.get("weight_frequency_exp", 0.5),
                    st.session_state.get("weight_assignment_shortage_exp", 0.5),
                    st.session_state.get("weight_lecturer_concentration_exp", 0.5),
                    st.session_state.get("weight_consecutive_assignment_exp", 0.5),
                    st.session_state.allow_under_assignment_cb,
                    st.session_state.TODAY, # today_date を適切な位置に渡す
                    st.session_state.get("fixed_assignments_for_solver"), # 追加
                    st.session_state.get("forced_unassignments_for_solver")
                )
            logger.info("solve_assignment completed.")

            if not isinstance(solver_output, dict):
                raise TypeError(f"最適化関数の戻り値が不正です。型: {type(solver_output)}")

            required_keys = ["full_application_and_solver_log", "pure_solver_log", "application_log", "solution_status_str", "objective_value", "assignments", "all_courses", "all_lecturers", "solver_raw_status_code"]
            missing_keys = [key for key in required_keys if key not in solver_output]
            if missing_keys:
                raise KeyError(f"最適化関数の戻り値に必要なキーが不足しています。不足キー: {missing_keys}")
            st.session_state.solver_log_for_download = solver_output["pure_solver_log"] # 純粋なソルバーログを保存
            st.session_state.application_log_for_download = solver_output["application_log"] # アプリケーションログを保存

            st.session_state.raw_log_on_server = solver_output["full_application_and_solver_log"]
            st.session_state.solver_result_cache = {
                k: v for k, v in solver_output.items() 
                if k not in ["full_application_and_solver_log", "pure_solver_log", "application_log"]
            }
            # 修正実行後は、次回通常の最適化のためにこれらのパラメータをクリア
            if "fixed_assignments_for_solver" in st.session_state: del st.session_state.fixed_assignments_for_solver
            if "forced_unassignments_for_solver" in st.session_state: del st.session_state.forced_unassignments_for_solver

            st.session_state.solution_executed = True
            st.session_state.view_mode = "optimization_result"

        except Exception as e:
            logger.error(f"Unexpected error during optimization process: {e}", exc_info=True)
            import traceback
            error_trace = traceback.format_exc()
            st.session_state.optimization_error_message = f"最適化処理中にエラーが発生しました:\n\n{error_trace}"
            st.session_state.solver_log_for_download = "" # エラー時は空
            st.session_state.application_log_for_download = "" # エラー時は空
            st.session_state.raw_log_on_server = f"OPTIMIZATION FAILED:\n{st.session_state.optimization_error_message}"
            st.session_state.solution_executed = True
            st.session_state.view_mode = "optimization_result"

    # OLD CALLBACK - to be removed or replaced
    # def handle_change_lecturer_callback(lecturer_id_to_remove: str, course_id_to_reassign: str):
    def handle_execute_changes_callback():
        # logger.info(f"Callback: Change lecturer for L:{lecturer_id_to_remove}, C:{course_id_to_reassign}") # 古いログ行を削除またはコメントアウト
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
        # The "最適化結果" button in the top navigation bar
        # It should be active if a solution has been executed.
        # It will always point to the main results view (tabular data).
        # The new "Change Assignment" functionality will be via a separate sidebar button.
        # The type="primary" logic for this button remains the same.
        # The actual content of the "optimization_result" view will be modified
        # to show the table, and the interactive "change lecturer" part will move
        # to a new view mode triggered by the new sidebar button.
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
       st.session_state.solver_result_cache.get("assignments"):
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
        # (サンプルデータ表示ロジックは省略)
        st.header("入力データ")

        if st.session_state.get("show_regenerate_success_message"):
            st.success("サンプルデータを再生成しました。")
            del st.session_state.show_regenerate_success_message # メッセージ表示後にフラグを削除

        st.button("サンプルデータ再生成", key="regenerate_sample_data_button", on_click=handle_regenerate_sample_data)
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

    elif st.session_state.view_mode == "objective_function":
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

    elif st.session_state.view_mode == "optimization_result":
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
                st.subheader(f"求解ステータス: {solver_result['solution_status_str']}")
                if solver_result['objective_value'] is not None:
                    st.metric("総コスト (目的値)", f"{solver_result['objective_value']:.2f}")

                if solver_result.get('assignments') and solver_result['solver_raw_status_code'] in [cp_model.OPTIMAL, cp_model.FEASIBLE]: # assignments の存在確認を追加
                    if solver_result['assignments']:
                        assigned_course_ids_for_message = {res["講座ID"] for res in solver_result['assignments']}
                        unassigned_courses_for_message = [c for c in solver_result['all_courses'] if c["id"] not in assigned_course_ids_for_message]
                        if not unassigned_courses_for_message:
                            st.success("全ての講座が割り当てられました。")

                if solver_result['assignments']:
                    results_df = pd.DataFrame(solver_result['assignments']) # この位置は問題なし
                    st.subheader("割り当て結果サマリー")
                    # ... (サマリー表示ロジックは変更なしのため省略) ...
                    summary_data = []
                    # ▼▼▼ 関連ロジックを修正 ▼▼▼
                    assignments_count = len(results_df)
                    summary_data.append(("**総割り当て件数**", f"{assignments_count}件"))
                    # ▲▲▲ ここまで修正 ▲▲▲

                    total_travel_cost = results_df["移動コスト(元)"].sum()
                    summary_data.append(("**移動コストの合計値**", f"{total_travel_cost} 円"))
                    assigned_lecturer_ids = results_df["講師ID"].unique()
                    temp_assigned_lecturers = [l for l in st.session_state.DEFAULT_LECTURERS_DATA if l["id"] in assigned_lecturer_ids]
                    if temp_assigned_lecturers:
                        avg_age = sum(l.get("age", 0) for l in temp_assigned_lecturers) / len(temp_assigned_lecturers)
                        summary_data.append(("**平均年齢**", f"{avg_age:.1f}才"))
                        avg_frequency = sum(len(l.get("past_assignments", [])) for l in temp_assigned_lecturers) / len(temp_assigned_lecturers)
                        summary_data.append(("**平均頻度**", f"{avg_frequency:.1f}回"))
                        # ... (ランク別、割り当て回数別サマリーも同様に省略) ...
                        # 一般資格ランク別割り当て状況
                        summary_data.append(("**一般資格ランク別割り当て**", "(講師が保有する一般資格ランク / 全講師中の同ランク保有者数)"))
                        general_rank_total_counts = {i: 0 for i in range(1, 6)}
                        for lecturer in st.session_state.DEFAULT_LECTURERS_DATA: # st.session_state から取得
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

                        # 特別資格ランク別割り当て状況
                        summary_data.append(("**特別資格ランク別割り当て**", "(講師が保有する特別資格ランク / 全講師中の同ランク保有者数)"))
                        special_rank_total_counts = {i: 0 for i in range(1, 6)}
                        for lecturer in st.session_state.DEFAULT_LECTURERS_DATA: # st.session_state から取得
                            rank = lecturer.get("qualification_special_rank")
                            if rank is not None and rank in special_rank_total_counts: # None は除外
                                special_rank_total_counts[rank] += 1
                        
                        assigned_special_rank_counts = {i: 0 for i in range(1, 6)}
                        for l_assigned in temp_assigned_lecturers:
                            rank = l_assigned.get("qualification_special_rank")
                            if rank is not None and rank in assigned_special_rank_counts: # None は除外
                                assigned_special_rank_counts[rank] += 1
                        for rank_num in range(1, 6):
                            summary_data.append((f"　特別ランク{rank_num}", f"{assigned_special_rank_counts.get(rank_num, 0)}人 / {special_rank_total_counts.get(rank_num, 0)}人中"))

                    if '今回の割り当て回数' in results_df.columns:
                        lecturer_assignment_counts_per_lecturer = results_df['講師ID'].value_counts()
                        counts_of_lecturers_by_assignment_num = lecturer_assignment_counts_per_lecturer.value_counts().sort_index()
                        summary_data.append(("**講師の割り当て回数別**", "(今回の最適化での担当講座数)"))
                        for num_assignments, num_lecturers in counts_of_lecturers_by_assignment_num.items():
                            if num_assignments >= 1:
                                summary_data.append((f"　{num_assignments}回 担当した講師", f"{num_lecturers}人"))
                    
                    # 「当該教室最終割当日からの日数」が実績なしを示す値 (-1, -3 など) の件数を集計
                    # -1: 過去実績なし, -2: 日付パースは成功したが未来など異常, -3: 日付パース失敗
                    # これらは実績なし優先コスト計算でコスト0になるケース
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
                if "pending_change_summary_info" in st.session_state and \
                   st.session_state.pending_change_summary_info and \
                   solver_result.get('assignments') is not None: # solver_result['assignments'] が None でないことを確認
                    
                    st.subheader("今回の割り当て変更による影響")
                    change_details_markdown = ""
                    
                    current_assignments_df_for_summary = pd.DataFrame(solver_result.get('assignments', []))
                    
                    for change_item in st.session_state.pending_change_summary_info:
                        original_lecturer_id = change_item['lecturer_id']
                        original_lecturer_name = change_item['lecturer_name']
                        course_id_changed = change_item['course_id']
                        course_name_changed = change_item['course_name']

                        new_assignment_for_course_df = pd.DataFrame() # 空のDataFrameで初期化
                        if not current_assignments_df_for_summary.empty:
                             new_assignment_for_course_df = current_assignments_df_for_summary[current_assignments_df_for_summary['講座ID'] == course_id_changed]

                        new_assignment_str = "割り当てなし"
                        if not new_assignment_for_course_df.empty:
                            new_lecturers_info = [f"{new_row['講師名']} (`{new_row['講師ID']}`)" for _, new_row in new_assignment_for_course_df.iterrows()]
                            new_assignment_str = "、".join(new_lecturers_info)
                        
                        change_details_markdown += f"- **講座:** {course_name_changed} (`{course_id_changed}`)\n  - **変更前:** {original_lecturer_name} (`{original_lecturer_id}`)\n  - **変更後:** {new_assignment_str}\n"
                    
                    if change_details_markdown:
                        st.markdown(change_details_markdown)
                    st.markdown("---")
                    del st.session_state.pending_change_summary_info # 表示後にクリア
                if solver_result.get('assignments') and solver_result['solver_raw_status_code'] in [cp_model.OPTIMAL, cp_model.FEASIBLE]: # assignments の存在確認を追加
                    if solver_result['assignments']:
                        results_df = pd.DataFrame(solver_result['assignments']) # Use results_df
                        st.subheader("割り当て結果詳細")
                        # Display the full results dataframe
                        st.dataframe(results_df)
                        st.markdown("---")

                        assigned_course_ids = {res["講座ID"] for res in solver_result['assignments']}
                        unassigned_courses = [c for c in solver_result['all_courses'] if c["id"] not in assigned_course_ids]
                        if unassigned_courses:
                            st.subheader("割り当てられなかった講座")
                            st.dataframe(pd.DataFrame(unassigned_courses))
                            st.caption("上記の講座は、スケジュール違反を許容しても、他の制約（資格ランクなど）により割り当て可能な講師が見つからなかったか、または他の割り当てと比較してコストが高すぎると判断された可能性があります。")
                    else:
                        st.error("解が見つかりましたが、実際の割り当ては行われませんでした。")
                        st.warning(
                            "考えられる原因:\n"
                            "- 割り当て可能なペアが元々存在しない (制約が厳しすぎる、データ不適合)。\n"
                            "**結果として、総コスト 0.00 (何も割り当てない) が最適と判断された可能性があります。**"
                        )
                        st.subheader("全ての講座が割り当てられませんでした")
                        st.dataframe(pd.DataFrame(solver_result['all_courses']))
                elif solver_result['solver_raw_status_code'] == cp_model.INFEASIBLE:
                    st.warning("指定された条件では、実行可能な割り当てが見つかりませんでした。制約やデータを見直してください。")
                else:
                    st.error(solver_result['solution_status_str'])

                if GEMINI_API_KEY and "raw_log_on_server" in st.session_state and st.session_state.raw_log_on_server:
                    if st.button("Gemini API によるログ解説を実行", key="run_gemini_explanation_button"):
                        st.session_state.gemini_api_requested = True
                        if "gemini_explanation" in st.session_state: del st.session_state.gemini_explanation
                        if "gemini_api_error" in st.session_state: del st.session_state.gemini_api_error
                        st.rerun()

                # ログダウンロードセクション
                if st.session_state.get("solution_executed"):
                    st.markdown("---") # 区切り線
                    st.subheader("ログのダウンロード")
                    dl_cols = st.columns(2)
                    with dl_cols[0]:
                        if st.session_state.get("solver_log_for_download"):
                            st.download_button(
                                label="ソルバーログのダウンロード",
                                data=st.session_state.solver_log_for_download,
                                file_name="solver_log.txt",
                                mime="text/plain",
                                key="download_solver_log_button",
                                help="最適化ソルバーが生成したログです。"
                            )
                    with dl_cols[1]:
                        if st.session_state.get("application_log_for_download"):
                            st.download_button(
                                label="アプリケーションログのダウンロード",
                                data=st.session_state.application_log_for_download,
                                file_name="application_log.txt",
                                mime="text/plain",
                                key="download_application_log_button",
                                help="最適化処理中のアプリケーション内部ログです。"
                            )
                
                if not GEMINI_API_KEY and st.session_state.get("solution_executed"):
                    if not GEMINI_API_KEY:
                        st.info("Gemini APIキーが設定されていません。ログ関連機能を利用するには設定が必要です。")
                        logger.info("Gemini API key not set. Log features disabled.")

                if st.session_state.get("gemini_api_requested") and \
                   "gemini_explanation" not in st.session_state and \
                   "gemini_api_error" not in st.session_state:
                    logger.info("Gemini API explanation requested. Calling API.")
                    with st.spinner("Gemini API でログを解説中..."):
                        full_log_to_filter = st.session_state.raw_log_on_server
                        filtered_log_for_gemini = filter_log_for_gemini(full_log_to_filter)
                        solver_cache = st.session_state.solver_result_cache
                        solver_status = solver_cache["solution_status_str"]
                        objective_value = solver_cache["objective_value"]
                        assignments_list = solver_cache.get("assignments", [])
                        assignments_summary_df = pd.DataFrame(assignments_list) if assignments_list else None

                        explanation_or_error_text, full_prompt_for_gemini = get_gemini_explanation(
                            filtered_log_for_gemini, GEMINI_API_KEY,
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

    elif st.session_state.view_mode == "change_assignment_view":
        st.header("割り当ての変更")
        logger.info("Displaying change assignment view.")

        if not st.session_state.get("solution_executed", False) or \
           "solver_result_cache" not in st.session_state or \
           not st.session_state.solver_result_cache.get("assignments"):
            st.warning("割り当て結果が存在しないため、この機能は利用できません。まず最適化を実行してください。")
        else: # 割り当て結果が存在する場合
            solver_result = st.session_state.solver_result_cache
            results_df = pd.DataFrame(solver_result['assignments'])

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


    else: # view_mode が予期せぬ値の場合 (フォールバック)
        # (解説表示ロジックは省略)
        st.header("ソルバーとmodelオブジェクト") # objective_function の場合
        logger.warning(f"Unexpected view_mode: {st.session_state.view_mode}. Displaying fallback info.")
        st.info("サイドバーから表示するデータを選択してください。")
    logger.info("Exiting main function.")
if __name__ == "__main__":
    main()
