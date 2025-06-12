import streamlit as st
from ortools.sat.python import cp_model
import pandas as pd
import io
import contextlib
import re # 正規表現モジュール
import datetime # 日付処理用に追加
import google.generativeai as genai # Gemini API 用
from streamlit_oauth import OAuth2Component # OIDC認証用
from google.oauth2 import id_token # IDトークン検証用
from google.auth.transport import requests as google_requests # IDトークン検証用
import random # データ生成用
from typing import TypedDict, List, Optional, Any, Tuple # 他のimport文と合わせて先頭に移動

# --- 1. データ定義 (LOG_EXPLANATIONS と _get_log_explanation は削除) ---

# --- Gemini API送信用ログのフィルタリング関数 (グローバルスコープに移動) ---
def filter_log_for_gemini(log_content: str) -> str:
    lines = log_content.splitlines()
    gemini_log_lines_final = []
    
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
    
    solver_log_start_marker = "--- Solver Log (Captured by app.py) ---"
    solver_log_end_marker = "--- End Solver Log (Captured by app.py) ---"

    for line in lines:
        if solver_log_start_marker in line:
            in_solver_log_block = True
            solver_log_block.append(line)
            continue 
        
        if solver_log_end_marker in line: 
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
    
    gemini_log_lines_final.extend(app_summary_lines)

    # (フィルタリングと省略ロジックは変更なし - ここでは省略)
    # ... (元の filter_log_for_gemini の残りのロジック) ...
    # この部分は元の関数のままなので、diffでは省略されていますが、実際にはここに元のロジック全体が入ります。
    # 簡単のため、ここでは主要な構造のみを示し、詳細な省略ロジックは元の関数を参照してください。
    # For brevity, the detailed line omission logic from the original filter_log_for_gemini is not repeated here.
    # Assume the full logic for app_detailed_lines_collected and solver_log_block truncation is present.
    MAX_APP_DETAIL_FIRST_N_LINES = 50 # 詳細アプリログの先頭行数を増やす (例: 3 -> 50)
    MAX_APP_DETAIL_LAST_N_LINES = 50  # 詳細アプリログの末尾行数を増やす (例: 3 -> 50)
    MAX_APP_DETAIL_MIDDLE_N_LINES = 10 # 詳細アプリログの中間から取得する行数

    if len(app_detailed_lines_collected) > (MAX_APP_DETAIL_FIRST_N_LINES + MAX_APP_DETAIL_MIDDLE_N_LINES + MAX_APP_DETAIL_LAST_N_LINES):
        gemini_log_lines_final.extend(app_detailed_lines_collected[:MAX_APP_DETAIL_FIRST_N_LINES])
        middle_start_index = len(app_detailed_lines_collected) // 2 - MAX_APP_DETAIL_MIDDLE_N_LINES // 2
        middle_end_index = middle_start_index + MAX_APP_DETAIL_MIDDLE_N_LINES
        gemini_log_lines_final.append(f"\n[... 詳細なアプリケーションログ（中間部分より抜粋） ...]\n")
        gemini_log_lines_final.extend(app_detailed_lines_collected[middle_start_index:middle_end_index])
        omitted_count = len(app_detailed_lines_collected) - (MAX_APP_DETAIL_FIRST_N_LINES + MAX_APP_DETAIL_MIDDLE_N_LINES + MAX_APP_DETAIL_LAST_N_LINES)
        gemini_log_lines_final.append(f"\n[... 他 {omitted_count} 件の詳細なアプリケーションログは簡潔さのため省略 ...]\n")
        gemini_log_lines_final.extend(app_detailed_lines_collected[-MAX_APP_DETAIL_LAST_N_LINES:])
    else:
        gemini_log_lines_final.extend(app_detailed_lines_collected)

    MAX_SOLVER_LOG_FIRST_N_LINES = 200 # ソルバーログの先頭行数を増やす (例: 30 -> 200)
    MAX_SOLVER_LOG_LAST_N_LINES = 200  # ソルバーログの末尾行数を増やす (例: 30 -> 200)
    MAX_SOLVER_LOG_MIDDLE_N_LINES = 10 # ソルバーログの中間から取得する行数

    if len(solver_log_block) > (MAX_SOLVER_LOG_FIRST_N_LINES + MAX_SOLVER_LOG_MIDDLE_N_LINES + MAX_SOLVER_LOG_LAST_N_LINES):
        truncated_solver_log = solver_log_block[:MAX_SOLVER_LOG_FIRST_N_LINES]
        middle_start_index_solver = len(solver_log_block) // 2 - MAX_SOLVER_LOG_MIDDLE_N_LINES // 2
        middle_end_index_solver = middle_start_index_solver + MAX_SOLVER_LOG_MIDDLE_N_LINES
        truncated_solver_log.append(f"\n[... ソルバーログ（中間部分より抜粋） ...]\n")
        truncated_solver_log.extend(solver_log_block[middle_start_index_solver:middle_end_index_solver])
        omitted_solver_lines = len(solver_log_block) - (MAX_SOLVER_LOG_FIRST_N_LINES + MAX_SOLVER_LOG_MIDDLE_N_LINES + MAX_SOLVER_LOG_LAST_N_LINES)
        truncated_solver_log.append(f"\n[... 他 {omitted_solver_lines} 件のソルバーログ中間行は簡潔さのため省略 ...]\n")
        truncated_solver_log.extend(solver_log_block[-MAX_SOLVER_LOG_LAST_N_LINES:])
        gemini_log_lines_final.extend(truncated_solver_log)
    else:
        gemini_log_lines_final.extend(solver_log_block)
    
    return "\n".join(gemini_log_lines_final)

# --- 大規模データ生成 ---
# (変更なし)

# --- Gemini API 連携 ---
def get_gemini_explanation(log_text: str,
                           api_key: str,
                           solver_status: str,
                           objective_value: Optional[float],
                           assignments_summary: Optional[pd.DataFrame]) -> str:
    """
    指定されたログテキストと最適化結果を Gemini API に送信し、解説を取得します。
    """
    if not api_key:
        return "エラー: Gemini API キーが設定されていません。"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro') 
        prompt = f"""以下のシステムログについて、IT専門家でない人にも分かりやすく解説してください。
ログの各部分が何を示しているのか、全体としてどのような処理が行われているのかを説明してください。
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
        return response.text
    except Exception as e:
        # st.error を直接呼ばず、エラーメッセージ文字列を返す
        return f"Gemini APIエラー: {str(e)[:500]}..."

PREFECTURES = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県",
    "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県",
    "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県",
    "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"
]

# 都道府県教室ID (P1 - P47)
PREFECTURE_CLASSROOM_IDS = [f"P{i+1}" for i in range(len(PREFECTURES))]

# 全教室データ生成
DEFAULT_CLASSROOMS_DATA = []
for i, pref_name in enumerate(PREFECTURES):
    DEFAULT_CLASSROOMS_DATA.append({"id": PREFECTURE_CLASSROOM_IDS[i], "location": pref_name})

ALL_CLASSROOM_IDS_COMBINED = PREFECTURE_CLASSROOM_IDS # 拠点という概念をなくし、都道府県教室のみとする

# 講師データ生成 (100人)
DEFAULT_LECTURERS_DATA = []
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
TIMES = ["AM", "PM"]
ALL_SLOTS = [(day, time) for day in DAYS for time in TIMES]
# AGE_CATEGORIES = ["low", "middle", "high"] # 実年齢を使用するため廃止
QUALIFICATION_RANKS = [1, 2, 3]
# FREQ_CATEGORIES = ["low", "middle", "high"] # 実際の割り当て回数を使用するため廃止

# 過去の割り当て日付生成用
TODAY = datetime.date.today()

for i in range(1, 301): # 講師数を100人から300人に変更
    num_available_slots = random.randint(3, 7)
    availability = random.sample(ALL_SLOTS, num_available_slots)

    # 過去の割り当て履歴を生成 (約10件)
    num_past_assignments = random.randint(8, 12) # 8から12件の間でランダム
    past_assignments = []
    for _ in range(num_past_assignments):
        days_ago = random.randint(1, 730) # 過去2年以内のランダムな日付
        assignment_date = TODAY - datetime.timedelta(days=days_ago)
        past_assignments.append({
            "classroom_id": random.choice(ALL_CLASSROOM_IDS_COMBINED),
            "date": assignment_date.strftime("%Y-%m-%d")
        })
    # 日付で降順ソート (最新が先頭)
    past_assignments.sort(key=lambda x: x["date"], reverse=True)

    DEFAULT_LECTURERS_DATA.append({
        "id": f"L{i}",
        "name": f"講師{i:03d}",
        "age": random.randint(22, 65), # 実年齢を追加 (例: 22歳から65歳)
        "home_classroom_id": random.choice(PREFECTURE_CLASSROOM_IDS), # 本拠地は都道府県のいずれか
        # "age_category": random.choice(AGE_CATEGORIES), # 廃止
        "qualification_rank": random.choice(QUALIFICATION_RANKS),
        "availability": availability,
        # "assignment_frequency_category": random.choice(FREQ_CATEGORIES), # 実際の割り当て回数を使用するため廃止
        "past_assignments": past_assignments # 過去の割り当て履歴
    })

# 講座データ生成 (47都道府県 × 7種類の講座)
# 講座種別を3種類に変更
BASE_COURSE_DEFINITIONS = [
    {"id_suffix": "C1", "name": "初級コース", "required_rank": 3, "schedule": ("Mon", "AM")}, # ランク3,2,1が担当可能
    {"id_suffix": "C2", "name": "中級コース", "required_rank": 2, "schedule": ("Tue", "PM")}, # ランク2,1が担当可能
    {"id_suffix": "C3", "name": "上級コース", "required_rank": 1, "schedule": ("Wed", "AM")}, # ランク1のみ担当可能
    # スケジュールのバリエーションを増やすため、同じコース種別で異なる時間帯も定義可能
    {"id_suffix": "C4", "name": "初級コース", "required_rank": 3, "schedule": ("Thu", "PM")},
    {"id_suffix": "C5", "name": "中級コース", "required_rank": 2, "schedule": ("Fri", "AM")},
]

DEFAULT_COURSES_DATA = []
for i, pref_classroom_id in enumerate(PREFECTURE_CLASSROOM_IDS):
    pref_name = PREFECTURES[i] # PREFECTURE_CLASSROOM_IDS と PREFECTURES のインデックスは対応
    for base_course in BASE_COURSE_DEFINITIONS:
        DEFAULT_COURSES_DATA.append({
            "id": f"{pref_classroom_id}-{base_course['id_suffix']}", # 例: P1-S1
            "name": f"{pref_name} {base_course['name']}", # 例: 北海道 初級プログラミング
            "classroom_id": pref_classroom_id,
            "required_rank": base_course["required_rank"],
            "schedule": base_course["schedule"]
        })

# --- 移動コスト生成のための地域定義 ---
REGIONS = {
    "Hokkaido": ["北海道"],
    "Tohoku": ["青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県"],
    "Kanto": ["茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県"],
    "Chubu": ["新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県", "静岡県", "愛知県"],
    "Kinki": ["三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県"],
    "Chugoku": ["鳥取県", "島根県", "岡山県", "広島県", "山口県"],
    "Shikoku": ["徳島県", "香川県", "愛媛県", "高知県"],
    "Kyushu_Okinawa": ["福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"]
}

PREFECTURE_TO_REGION = {pref: region for region, prefs in REGIONS.items() for pref in prefs}

REGION_GRAPH = { # 地域間の隣接関係グラフ (ホップ数計算用)
    "Hokkaido": {"Tohoku"},
    "Tohoku": {"Hokkaido", "Kanto", "Chubu"},
    "Kanto": {"Tohoku", "Chubu"},
    "Chubu": {"Tohoku", "Kanto", "Kinki"},
    "Kinki": {"Chubu", "Chugoku", "Shikoku"},
    "Chugoku": {"Kinki", "Shikoku", "Kyushu_Okinawa"},
    "Shikoku": {"Kinki", "Chugoku", "Kyushu_Okinawa"},
    "Kyushu_Okinawa": {"Chugoku", "Shikoku"}
}

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

# 教室IDから都道府県名へのマッピングを作成
CLASSROOM_ID_TO_PREF_NAME = {item["id"]: item["location"] for item in DEFAULT_CLASSROOMS_DATA}

# 移動コスト行列生成 (全教室間)
DEFAULT_TRAVEL_COSTS_MATRIX = {}

for c_from in ALL_CLASSROOM_IDS_COMBINED:
    for c_to in ALL_CLASSROOM_IDS_COMBINED:
        if c_from == c_to:
            base_cost = 0
        else:
            pref_from = CLASSROOM_ID_TO_PREF_NAME[c_from]
            pref_to = CLASSROOM_ID_TO_PREF_NAME[c_to]
            
            region_from = PREFECTURE_TO_REGION[pref_from]
            region_to = PREFECTURE_TO_REGION[pref_to]

            # 沖縄県と本土間の特別処理
            is_okinawa_involved = (pref_from == "沖縄県" and pref_to != "沖縄県") or \
                                  (pref_to == "沖縄県" and pref_from != "沖縄県")

            if is_okinawa_involved:
                base_cost = random.randint(80000, 120000) # 沖縄は高コスト帯 (円単位のイメージ)
            elif region_from == region_to: # 同一地域内
                base_cost = random.randint(5000, 15000)   # 低コスト帯
            else:
                hops = get_region_hops(region_from, region_to, REGION_GRAPH)
                if hops == 1: # 隣接地域
                    base_cost = random.randint(15000, 30000) # 中コスト帯
                elif hops == 2: # 1つ地域を挟む
                    base_cost = random.randint(35000, 60000) # やや高コスト帯
                else: # 2つ以上地域を挟む、または到達不能(get_region_hopsがinfの場合)
                    base_cost = random.randint(70000, 100000) # 高コスト帯
            
        DEFAULT_TRAVEL_COSTS_MATRIX[(c_from, c_to)] = base_cost

# DEFAULT_AGE_PRIORITY_COSTS は実年齢を使用するため廃止

# DEFAULT_FREQUENCY_PRIORITY_COSTS は実際の割り当て回数を使用するため廃止

# スケジュール違反に対する固定ペナルティ (floatで定義し、他の重み付けコストと合算)
# この値は、スケジュール違反を許容する場合に、他のコスト要因よりも優先度が低くなるように十分に大きく設定します。
BASE_PENALTY_SCHEDULE_VIOLATION = 1000000.0  # 例: 100万

# 過去の割り当てがない、または日付パース不能な場合に設定するデフォルトの経過日数 (ペナルティ計算上、十分に大きい値)
# サマリー表示でも使用するためグローバルスコープに移動
DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT = 100000

# --- 2. OR-Tools 最適化ロジック ---
class SolverOutput(TypedDict): # 提案: 戻り値を構造化するための型定義
    solution_status_str: str
    objective_value: Optional[float]
    assignments: List[dict]
    all_courses: List[dict]
    all_lecturers: List[dict]
    solver_raw_status_code: int
    full_application_and_solver_log: str # All logs including detailed app logs for UI's explained_log_text

def solve_assignment(lecturers_data, courses_data, classrooms_data,
                     travel_costs_matrix, # frequency_priority_costs を削除
                     weight_past_assignment_recency, weight_qualification, 
                     ignore_schedule_constraint: bool, # スケジュール制約を無視するかのフラグ
                     weight_travel, weight_age, weight_frequency) -> SolverOutput:
    model = cp_model.CpModel()
    # solve_assignment 内の print 文も解説対象に含めるために、
    # ここで stdout のキャプチャを開始する
    full_log_stream = io.StringIO()

    # アプリケーションログを full_log_stream に直接書き込むように変更
    def log_to_stream(message):
        print(message, file=full_log_stream)
        # print(message) # ターミナルにも表示（デバッグ用） - 大量ログの場合、パフォーマンスに影響する可能性

    # --- Main logic for model building and solving ---
    possible_assignments = []
    potential_assignment_count = 0
    log_to_stream(f"Initial lecturers: {len(lecturers_data)}, Initial courses: {len(courses_data)}")

    for lecturer in lecturers_data:
        for course in courses_data:
            lecturer_id = lecturer["id"]
            course_id = course["id"]

            # 資格ランクチェック: 講師のランクが講座の要求ランクより低い(数値が大きい)場合は除外
            if lecturer["qualification_rank"] > course["required_rank"]:
                log_to_stream(f"  - Filtered out: {lecturer_id} for {course_id} (Rank insufficient: Lecturer_rank={lecturer['qualification_rank']} (higher number is lower rank), Course_required_rank={course['required_rank']} (lecturer rank must be <= this number))")
                continue
            
            # スケジュールチェック
            schedule_available = course["schedule"] in lecturer["availability"]
            actual_schedule_incompatibility_occurred = not schedule_available # 最終的な結果表示用
            schedule_violation_penalty = 0.0

            if not schedule_available: # スケジュールが合わない場合
                if ignore_schedule_constraint: # スケジュール制約を無視する設定の場合
                    schedule_violation_penalty = BASE_PENALTY_SCHEDULE_VIOLATION
                    log_to_stream(f"  - Schedule incompatible (constraint ignored, penalty {schedule_violation_penalty} applied): {lecturer_id} for {course_id} (Course_schedule={course['schedule']}, Lecturer_avail={lecturer['availability']})")
                else: # スケジュール制約を無視しない設定の場合 -> 割り当て不可
                    log_to_stream(f"  - Filtered out: {lecturer_id} for {course_id} (Schedule unavailable and constraint NOT ignored: Course_schedule={course['schedule']}, Lecturer_avail={lecturer['availability']})")
                    continue
            # else: スケジュールが合う場合はペナルティなし

            potential_assignment_count += 1
            log_to_stream(f"  + Potential assignment: {lecturer_id} to {course_id}")
            var = model.NewBoolVar(f'x_{lecturer_id}_{course_id}')
            
            travel_cost = travel_costs_matrix.get((lecturer["home_classroom_id"], course["classroom_id"]), 999)
            age_cost = lecturer.get("age", 99) # 実年齢をコストとして使用。未設定の場合は大きな値。
            # 実際の過去の総割り当て回数を頻度コストとする (少ないほど良い)
            frequency_cost = len(lecturer.get("past_assignments", []))
            qualification_cost = lecturer["qualification_rank"] # ランク値が小さいほど高資格

            # 過去割り当ての近さによるコスト計算
            past_assignment_recency_cost = 0
            days_since_last_assignment_to_classroom = DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT

            if lecturer.get("past_assignments"):
                relevant_past_assignments_to_this_classroom = [
                    pa for pa in lecturer["past_assignments"]
                    if pa["classroom_id"] == course["classroom_id"]
                ]
                if relevant_past_assignments_to_this_classroom:
                    # past_assignments は日付降順ソート済みなので、リストの最初のものが最新の割り当て
                    latest_assignment_date_str = relevant_past_assignments_to_this_classroom[0]["date"]
                    try:
                        latest_assignment_date = datetime.datetime.strptime(latest_assignment_date_str, "%Y-%m-%d").date()
                        days_since_last_assignment_to_classroom = (TODAY - latest_assignment_date).days

                        # コスト計算: 経過日数が少ないほど高いコスト
                        # (DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT - 経過日数)
                        # 経過日数が DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT の場合、コストは0
                        raw_recency_cost = DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT - days_since_last_assignment_to_classroom
                        past_assignment_recency_cost = raw_recency_cost
                    except ValueError:
                        log_to_stream(f"    Warning: Could not parse date '{latest_assignment_date_str}' for {lecturer_id} and classroom {course['classroom_id']}")
                        days_since_last_assignment_to_classroom = DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT # パース失敗時
            
            total_weighted_cost_float = (weight_travel * travel_cost +
                                         weight_age * age_cost +
                                         weight_frequency * frequency_cost +
                                         weight_qualification * qualification_cost + # 資格コストを追加
                                         weight_past_assignment_recency * past_assignment_recency_cost
                                        ) + schedule_violation_penalty # スケジュール違反ペナルティを加算
            total_weighted_cost_int = int(total_weighted_cost_float * 100) # コストを整数にスケーリング
            log_to_stream(f"    Cost for {lecturer_id} to {course_id}: travel={travel_cost}, age={age_cost}, freq={frequency_cost}, qual={qualification_cost}, sched_viol_penalty={schedule_violation_penalty}, recency_cost_raw={past_assignment_recency_cost} (days_since_last_on_this_classroom={days_since_last_assignment_to_classroom}), total_weighted_int={total_weighted_cost_int}")
            possible_assignments.append({
                "lecturer_id": lecturer_id, "course_id": course_id,
                "variable": var, "cost": total_weighted_cost_int,
                "qualification_cost_raw": qualification_cost, "is_schedule_incompatible": actual_schedule_incompatibility_occurred, # キー名を is_schedule_violation から変更
                "debug_past_assignment_recency_cost": past_assignment_recency_cost, # デバッグ/結果表示用
                "debug_days_since_last_assignment": days_since_last_assignment_to_classroom
            })

    log_to_stream(f"Total potential assignments after filtering: {potential_assignment_count}")
    log_to_stream(f"Length of possible_assignments list (with variables): {len(possible_assignments)}")

    if not possible_assignments:
        log_to_stream("No possible assignments found after filtering. Optimization will likely result in no assignments.")
        all_captured_logs = full_log_stream.getvalue()
        return SolverOutput(
            solution_status_str="前提条件エラー (割り当て候補なし)",
            objective_value=None,
            assignments=[],
            all_courses=courses_data,
            all_lecturers=lecturers_data,
            solver_raw_status_code=cp_model.UNKNOWN, 
            full_application_and_solver_log=all_captured_logs
        )

    for course_item in courses_data:
        course_id = course_item["id"]
        # 各講座は、担当可能な講師候補が存在する場合に限り、必ず1名割り当てる。
        # 資格ランクなどのハード制約により候補がいない場合は、この強制割り当ての対象外とする。
        possible_assignments_for_course = [pa["variable"] for pa in possible_assignments if pa["course_id"] == course_id]
        if possible_assignments_for_course: # 担当可能な講師候補がいる場合のみ制約を追加
            model.Add(sum(possible_assignments_for_course) == 1)
            
    # 各講師は、割り当て期間全体を通して最大1つの講座のみ担当可能
    for lecturer_item in lecturers_data:
        lecturer_id = lecturer_item["id"]
        assignments_for_lecturer = [pa["variable"] for pa in possible_assignments if pa["lecturer_id"] == lecturer_id]
        if assignments_for_lecturer:
            model.Add(sum(assignments_for_lecturer) <= 1)

    assignment_costs = [pa["variable"] * pa["cost"] for pa in possible_assignments]
    # 未割り当てペナルティ (penalty_terms) を削除
    objective_terms = assignment_costs
    if objective_terms:
        model.Minimize(sum(objective_terms))
    else:
        log_to_stream("Objective terms list is empty. No assignments to optimize.")
        all_captured_logs = full_log_stream.getvalue()
        return SolverOutput(
            solution_status_str="目的関数エラー (最適化対象なし)",
            objective_value=None,
            assignments=[],
            all_courses=courses_data,
            all_lecturers=lecturers_data,
            solver_raw_status_code=cp_model.MODEL_INVALID,
            full_application_and_solver_log=all_captured_logs
        )

    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True

    log_to_stream("--- Solver Log (Captured by app.py) ---")
    
    status_code = cp_model.UNKNOWN # Initialize status_code
    with contextlib.redirect_stdout(full_log_stream):
        status_code = solver.Solve(model)
    
    log_to_stream("--- End Solver Log (Captured by app.py) ---")

    full_captured_logs = full_log_stream.getvalue()
    
    # DEBUG: キャプチャされた全ログの内容をターミナルに出力して確認
    print("\n--- BEGIN all_captured_logs (for debugging filter) ---")
    print(full_captured_logs)
    print("--- END all_captured_logs (for debugging filter) ---\n")

    status_name = solver.StatusName(status_code) # Get the status name
    results = []
    objective_value = None
    solution_status_str = "解なし"

    if status_code == cp_model.OPTIMAL or status_code == cp_model.FEASIBLE:
        solution_status_str = "最適解" if status_code == cp_model.OPTIMAL else "実行可能解"
        objective_value = solver.ObjectiveValue() / 100 # スケーリングを戻す
        
        for pa in possible_assignments:
            if solver.Value(pa["variable"]) == 1:
                lecturer = next(l for l in lecturers_data if l["id"] == pa["lecturer_id"])
                course = next(c for c in courses_data if c["id"] == pa["course_id"])
                results.append({
                    "講師ID": lecturer["id"],
                    "講師名": lecturer["name"],
                    "講座ID": course["id"],
                    "講座名": course["name"],
                    "教室ID": course["classroom_id"],
                    "スケジュール": f"{course['schedule'][0]} {course['schedule'][1]}",
                    "算出コスト(x100)": pa["cost"], # pa["cost"] は重み付け後の整数コスト
                    "移動コスト(元)": travel_costs_matrix.get((lecturer["home_classroom_id"], course["classroom_id"]), 999),
                    "年齢コスト(元)": lecturer.get("age", 99),
                    "頻度コスト(元)": len(lecturer.get("past_assignments", [])), # 実際の総割り当て回数
                    "スケジュール状況": "不適合" if pa.get("is_schedule_incompatible") else "適合", # "不適合" に変更
                    "資格コスト(元)": pa.get("qualification_cost_raw"), # 講師の資格ランク
                    "当該教室最終割当日からの日数": pa.get("debug_days_since_last_assignment") # "該当なし" のフォールバックを削除し、格納された値を直接使用
                })
    elif status_code == cp_model.INFEASIBLE:
        solution_status_str = "実行不可能 (制約を満たす解なし)"
    else:
        solution_status_str = f"解探索失敗 (ステータス: {status_name} [{status_code}])" # Include name and code
        
    return SolverOutput(
        solution_status_str=solution_status_str,
        objective_value=objective_value,
        assignments=results,
        all_courses=courses_data,
        all_lecturers=lecturers_data,
        solver_raw_status_code=status_code,
        full_application_and_solver_log=full_captured_logs # 全ログ
    )

# --- 3. Streamlit UI ---
def main():
    st.set_page_config(page_title="講師割り当てシステムデモ", layout="wide")

    # --- OIDC認証設定 ---
    GOOGLE_CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = st.secrets.get("GOOGLE_CLIENT_SECRET")
    REDIRECT_URI = st.secrets.get("REDIRECT_URI")
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

    AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
    # REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke" # 必要に応じて

    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REDIRECT_URI]):
        st.error("Google OAuth の設定が不完全です。管理者にお問い合わせください。(.streamlit/secrets.toml を確認してください)")
        st.stop()

    oauth2 = OAuth2Component(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        authorize_endpoint=AUTHORIZE_ENDPOINT,
        token_endpoint=TOKEN_ENDPOINT,
        refresh_token_endpoint=None,
        revoke_token_endpoint=None, # REVOKE_ENDPOINT,
    )

    # --- 認証とメインコンテンツの表示制御 ---
    if 'token' not in st.session_state:
        st.session_state.token = None # 初期化
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None # 初期化

    if not st.session_state.token:
        # --- 未認証の場合: ログインページ表示 ---
        st.title("講師割り当てシステムへようこそ")
        st.write("続行するにはGoogleアカウントでログインしてください。")
        result = oauth2.authorize_button(
            name="Googleでログイン",
            icon="https://www.google.com/favicon.ico",
            redirect_uri=REDIRECT_URI,
            scope="email profile openid", # openid を追加してユーザー情報を取得しやすくする
            key="google_login_main", # 他のボタンとキーが衝突しないように変更
            extras_params={"access_type": "offline"} # "prompt": "consent" を削除
        )
        if result and "token" in result:
            st.session_state.token = result.get("token")
            try:
                # IDトークンを取得して検証
                id_token_str = st.session_state.token.get("id_token")
                if id_token_str:
                    id_info = id_token.verify_oauth2_token(
                        id_token_str,
                        google_requests.Request(),
                        GOOGLE_CLIENT_ID
                    )
                    st.session_state.user_info = {
                        "email": id_info.get("email"),
                        "name": id_info.get("name")
                    }
                else:
                    st.error("IDトークンが取得できませんでした。")
                    st.session_state.user_info = {"email": "error@example.com", "name": "Unknown User"}
            except Exception as e:
                st.error(f"ユーザー情報の取得/設定中にエラー: {e}")
                st.session_state.user_info = {"email": "error@example.com"}
            st.rerun()
        return # 未認証の場合はここで処理を終了し、メインUIは表示しない

    # --- 認証済みの場合: メインアプリケーションUI表示 ---
    # このブロックは st.session_state.token が存在する場合のみ実行されます

    # st.sidebar.header("最適化設定") # より詳細な構成に変更

    # --- セッション状態の初期化 (表示モード管理用) ---
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "sample_data"  # デフォルトはサンプルデータ表示
    if "solution_executed" not in st.session_state:
        st.session_state.solution_executed = False

    # --- メイン画面上部にナビゲーションボタンを配置 ---
    nav_cols = st.columns([1, 1, 5]) # ボタンの幅を調整するための比率 (最後の列はスペーサー)
    with nav_cols[0]:
        if st.button("サンプルデータ", key="nav_sample_data_button", use_container_width=True):
            st.session_state.view_mode = "sample_data"
            # 最適化関連のキャッシュをクリアするが、solution_executed は変更しない
            # これにより、「最適化結果」ボタンが表示されたままになる
            keys_to_clear_for_sample_view = [
                "solver_result_cache",
                "raw_log_on_server",
                "gemini_explanation",
                "gemini_api_requested",
                "gemini_api_error",
                # "solution_executed" # ここから削除
            ]
            for key_to_clear in keys_to_clear_for_sample_view:
                if key_to_clear in st.session_state:
                    del st.session_state[key_to_clear]
            st.rerun()

    with nav_cols[1]:
        if st.session_state.get("solution_executed", False): # 最適化実行後に表示
            if st.button("最適化結果", key="nav_optimization_result_button", use_container_width=True):
                st.session_state.view_mode = "optimization_result"
                st.rerun()
        else:
            # 最適化実行前はボタンを非表示にするか、無効化する (ここでは無効化)
            st.button("最適化結果", key="nav_optimization_result_disabled", disabled=True, use_container_width=True)

    st.sidebar.markdown(
        "【制約】と【目的】を設定すれば、数理モデル最適化手法により自動的に最適な講師割り当てを実行します。"
        "また目的に重み付けすることでチューニングすることができます。"
    )
    
    # 「最適割り当てを実行」ボタンを説明文の直下に移動
    if st.sidebar.button("最適割り当てを実行", type="primary", key="execute_optimization_main_button"):
        # 既存のセッション変数をクリア (特に計算結果キャッシュ)
        if "solver_result_cache" in st.session_state: del st.session_state.solver_result_cache
        if "raw_log_on_server" in st.session_state: del st.session_state.raw_log_on_server
        if "gemini_explanation" in st.session_state: del st.session_state.gemini_explanation
        if "gemini_api_requested" in st.session_state: del st.session_state.gemini_api_requested # クリア
        if "gemini_api_error" in st.session_state: del st.session_state.gemini_api_error # クリア
        
        st.session_state.solution_executed = True # 実行フラグを立てる
        st.session_state.view_mode = "optimization_result" # 表示モードを最適化結果に
        st.rerun() # 再実行してメインエリアで処理と表示を行う

    # サイドバーの「サンプルデータ」「最適化結果」ボタンは削除 (上部に移動したため)

    st.sidebar.markdown("---")
    with st.sidebar.expander("【制約】", expanded=False):
        st.markdown("**ハード制約（絶対固定）**")
        st.markdown("- 講師は資格ランクに応じた講座しか割り当てできません。")
        st.markdown("- 講師は（今回の割り当てでは）1つの講座しか担当できません。")

        st.markdown("**ソフト制約（割り当てできない場合に許容できる）**")
        st.markdown(
            "講師の空きスケジュールに応じた講座しか割り当てできないことが原則ですが、"
            "完全割り当てのために以下のことを許容できます。"
        )
        ignore_schedule_constraint_checkbox = st.checkbox( # st.sidebar.checkbox から st.checkbox に変更
            "講師の空きスケジュールを無視する", 
            value=True, 
            help="チェックを外すと割り当て結果が得られないことがあります。その場合は、講師の空きスケジュールが合わない場合が想定されます。"
        )

    with st.sidebar.expander("【目的】", expanded=False):
        st.caption( # st.sidebar.caption から st.caption に変更
            "【重み】不要な目的はゼロにしてください。（目的から除外されます）"
            "また、相対的な値なので、全部0.1と全部1.0は同じ結果となります。"
        )
        st.markdown("**移動コストが低い人を優先**")
        weight_travel = st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど移動コストが低い人を重視します。", key="weight_travel_exp")
        st.markdown("**年齢の若い人を優先**")
        weight_age = st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど年齢が若い人を重視します。", key="weight_age_exp")
        st.markdown("**割り当て頻度の少ない人を優先**")
        weight_frequency = st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど全講座割当回数が少ない人を重視します。", key="weight_frequency_exp")
        st.markdown("**講師資格が高い人を優先**")
        weight_qualification_slider = st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど講師資格ランクが高い人が重視されます。", key="weight_qualification_exp")
        st.markdown("**同教室への割り当て実績が無い人を優先**")
        weight_past_assignment_recency_slider = st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど同教室への割り当て実績が無い人、或いは最後に割り当てられた日からの経過日数が長い人が重視されます。", key="weight_past_assignment_exp")

    # ログインユーザー情報とログアウトボタン
    user_email = st.session_state.user_info.get('email', '不明なユーザー') if st.session_state.user_info else '不明なユーザー'
    st.sidebar.markdown("---")
    st.sidebar.write(f"ログイン中: {user_email}")
    if st.sidebar.button("ログアウト"):
        st.session_state.token = None
        st.session_state.user_info = None
        # 関連するセッションステートもクリア (最適化結果キャッシュも含む)
        keys_to_clear = [
            "gemini_explanation", 
            "solution_executed", 
            "solver_result_cache",
            "raw_log_on_server",    # サーバー側で保持する生ログ
            "gemini_api_requested", # Gemini API実行フラグ
            "gemini_api_error"      # Gemini APIエラーメッセージ
        ]
        for key_to_clear in keys_to_clear:
            if key_to_clear in st.session_state:
                del st.session_state[key_to_clear]
        st.rerun()

    st.title("講師割り当てシステム デモ (OR-Tools) - ログ解説付き")

    # --- メインエリアの表示制御 ---
    if st.session_state.view_mode == "sample_data":
        st.header("入力データ")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("講師データ (サンプル)")
            # past_assignments を表示用に整形
            df_lecturers = pd.DataFrame(DEFAULT_LECTURERS_DATA)
            if 'past_assignments' in df_lecturers.columns:
                df_lecturers['past_assignments'] = df_lecturers['past_assignments'].apply(
                    lambda assignments: ", ".join([f"{a['classroom_id']} ({a['date']})" for a in assignments]) if isinstance(assignments, list) and assignments else "履歴なし"
                )
            st.dataframe(df_lecturers, height=200)
        with col2:
            st.subheader("講座データ (サンプル)")
            st.dataframe(pd.DataFrame(DEFAULT_COURSES_DATA), height=200)
        
        st.subheader("教室データと移動コスト (サンプル)")
        col3, col4 = st.columns(2)
        with col3:
            st.dataframe(pd.DataFrame(DEFAULT_CLASSROOMS_DATA))
        with col4:
            # travel_costs_matrix を表示用に整形
            df_travel_costs = pd.DataFrame([
                {"出発教室": k[0], "到着教室": k[1], "コスト": v}
                for k, v in DEFAULT_TRAVEL_COSTS_MATRIX.items()
            ])
            st.dataframe(df_travel_costs)

    elif st.session_state.view_mode == "optimization_result":
        # 最適化実行フラグに基づいて結果を表示
        if st.session_state.get("solution_executed", False):
            # # サイドバーに「最適化結果を表示中」の旨を表示 -> ボタンに変更したためコメントアウト
            # st.sidebar.markdown("表示中: **最適化結果**")

            st.header("最適化結果") # ヘッダーは計算前に表示

            # 計算結果がキャッシュにない場合のみ計算を実行
            if "solver_result_cache" not in st.session_state:
                with st.spinner("最適化計算を実行中..."): 
                    solver_output = solve_assignment(
                        DEFAULT_LECTURERS_DATA, DEFAULT_COURSES_DATA, DEFAULT_CLASSROOMS_DATA,
                        DEFAULT_TRAVEL_COSTS_MATRIX,
                        weight_past_assignment_recency_slider, weight_qualification_slider, 
                        ignore_schedule_constraint_checkbox,
                        weight_travel, weight_age, weight_frequency
                    )
                    # 生ログは別のセッションステートに保存し、結果キャッシュには含めない
                    st.session_state.raw_log_on_server = solver_output["full_application_and_solver_log"]
                    
                    # 結果キャッシュにはログ以外の情報を格納
                    st.session_state.solver_result_cache = {
                        "solution_status_str": solver_output["solution_status_str"],
                        "objective_value": solver_output["objective_value"],
                        "assignments": solver_output["assignments"],
                        "all_courses": solver_output["all_courses"],
                        "all_lecturers": solver_output["all_lecturers"],
                        "solver_raw_status_code": solver_output["solver_raw_status_code"],
                    }

            # キャッシュされた結果（または計算直後の結果）を取得
            solver_result = st.session_state.solver_result_cache
            st.subheader(f"求解ステータス: {solver_result['solution_status_str']}")
            if solver_result['objective_value'] is not None:
                st.metric("総コスト (目的値)", f"{solver_result['objective_value']:.2f}")

            if solver_result['assignments']:
                results_df = pd.DataFrame(solver_result['assignments']) # このdfはローカルでOK
                st.subheader("割り当て結果サマリー")
                
                summary_data = []
                schedule_compatible_count = results_df[results_df["スケジュール状況"] == "適合"].shape[0]
                schedule_incompatible_count = results_df[results_df["スケジュール状況"] == "不適合"].shape[0]
                summary_data.append(("**スケジュール**", ""))
                summary_data.append(("　適合", f"{schedule_compatible_count}人"))
                summary_data.append(("　不適合（講師の空きスケジュールに不適合）", f"{schedule_incompatible_count}人"))
                total_travel_cost = results_df["移動コスト(元)"].sum()
                summary_data.append(("**移動コストの合計値**", f"{total_travel_cost} 円"))
                assigned_lecturer_ids = results_df["講師ID"].unique()
                temp_assigned_lecturers = [l for l in DEFAULT_LECTURERS_DATA if l["id"] in assigned_lecturer_ids]
                if temp_assigned_lecturers:
                    avg_age = sum(l.get("age", 0) for l in temp_assigned_lecturers) / len(temp_assigned_lecturers)
                    summary_data.append(("**平均年齢**", f"{avg_age:.1f}才"))
                    avg_frequency = sum(len(l.get("past_assignments", [])) for l in temp_assigned_lecturers) / len(temp_assigned_lecturers)
                    summary_data.append(("**平均頻度**", f"{avg_frequency:.1f}回"))
                    lecturer_rank_total_counts = {1: 0, 2: 0, 3: 0}
                    for lecturer in DEFAULT_LECTURERS_DATA:
                        rank = lecturer.get("qualification_rank")
                        if rank in lecturer_rank_total_counts:
                            lecturer_rank_total_counts[rank] += 1
                    summary_data.append(("**資格別割り当て状況**", ""))
                    assigned_rank_counts = {1: 0, 2: 0, 3: 0}
                    for l_assigned in temp_assigned_lecturers:
                        rank = l_assigned.get("qualification_rank")
                        if rank in assigned_rank_counts:
                            assigned_rank_counts[rank] += 1
                    for rank_num in [1, 2, 3]:
                        summary_data.append((f"　ランク{rank_num}", f"{assigned_rank_counts.get(rank_num, 0)}人 / {lecturer_rank_total_counts.get(rank_num, 0)}人中"))
                past_assignment_new_count = results_df[results_df["当該教室最終割当日からの日数"] == DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT].shape[0]
                past_assignment_existing_count = results_df.shape[0] - past_assignment_new_count
                summary_data.append(("**同教室への過去の割り当て**", ""))
                summary_data.append(("　新規", f"{past_assignment_new_count}人"))
                summary_data.append(("　割当て実績あり", f"{past_assignment_existing_count}人"))
                markdown_table = "| 項目 | 値 |\n| :---- | :---- |\n"
                for item, value in summary_data:
                    markdown_table += f"| {item} | {value} |\n"
                st.markdown(markdown_table)
                st.markdown("---")

            if solver_result['solver_raw_status_code'] == cp_model.OPTIMAL or solver_result['solver_raw_status_code'] == cp_model.FEASIBLE:
                if solver_result['assignments']: # 'assignments' が空でないことを確認
                    results_df_display = pd.DataFrame(solver_result['assignments']) # 表示用に再度DataFrame作成
                    st.subheader("割り当て結果")
                    st.dataframe(results_df_display)
                    assigned_course_ids = {res["講座ID"] for res in solver_result['assignments']}
                    unassigned_courses = [c for c in solver_result['all_courses'] if c["id"] not in assigned_course_ids]
                    if unassigned_courses:
                        st.subheader("割り当てられなかった講座")
                        st.dataframe(pd.DataFrame(unassigned_courses))
                        st.caption("上記の講座は、スケジュール違反を許容しても、他の制約（資格ランクなど）により割り当て可能な講師が見つからなかったか、または他の割り当てと比較してコストが高すぎると判断された可能性があります。")
                    else:
                        st.success("全ての講座が割り当てられました。")
                else: # assignments が空の場合 (OPTIMAL/FEASIBLEだが割り当てなし)
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
            else: # UNKNOWN, MODEL_INVALID など
                st.error(solver_result['solution_status_str'])

            # 「Gemini API によるログ解説を実行」ボタン
            if GEMINI_API_KEY and "raw_log_on_server" in st.session_state and st.session_state.raw_log_on_server is not None:
                if st.button("Gemini API によるログ解説を実行", key="run_gemini_explanation_button"):
                    st.session_state.gemini_api_requested = True # 実行フラグ
                    # 既存の解説があればクリア
                    if "gemini_explanation" in st.session_state: del st.session_state.gemini_explanation
                    if "gemini_api_error" in st.session_state: del st.session_state.gemini_api_error
                    st.rerun() # ボタン押下で再実行し、下のブロックでAPI呼び出しと表示

                # 生ログダウンロードボタン (Gemini APIボタンの下)
                st.download_button(
                    label="ログのダウンロード",
                    data=st.session_state.raw_log_on_server,
                    file_name="assignment_log.txt",
                    mime="text/plain",
                    key="download_raw_log_button"
                )
            elif st.session_state.get("solution_executed"): # ボタンが表示されない場合のヒント (最適化実行後)
                if not GEMINI_API_KEY:
                    st.info("Gemini APIキーが設定されていません。ログ関連機能を利用するには設定が必要です。")
                elif "raw_log_on_server" not in st.session_state or st.session_state.raw_log_on_server is None:
                    st.warning("ログデータが利用できないため、ログ関連機能は表示されません。最適化処理が完了していないか、ログ取得に失敗した可能性があります。")

            # Gemini API 呼び出しと結果表示 (ボタン押下後に実行される)
            if st.session_state.get("gemini_api_requested") and \
               "gemini_explanation" not in st.session_state and \
               "gemini_api_error" not in st.session_state:
                    with st.spinner("Gemini API でログを解説中..."):
                        full_log_to_filter = st.session_state.raw_log_on_server # サーバー側の生ログを使用
                        filtered_log_for_gemini = filter_log_for_gemini(full_log_to_filter)

                        # solver_result_cache から必要な情報を取得
                        solver_cache = st.session_state.solver_result_cache
                        solver_status = solver_cache["solution_status_str"]
                        objective_value = solver_cache["objective_value"]
                        assignments_list = solver_cache.get("assignments", [])
                        assignments_summary_df = pd.DataFrame(assignments_list) if assignments_list else None

                        gemini_explanation_text = get_gemini_explanation(
                            filtered_log_for_gemini, GEMINI_API_KEY,
                            solver_status, objective_value, assignments_summary_df
                        )

                        if gemini_explanation_text.startswith("Gemini APIエラー:"):
                            st.session_state.gemini_api_error = gemini_explanation_text
                        else:
                            st.session_state.gemini_explanation = gemini_explanation_text
                            if "gemini_api_error" in st.session_state: del st.session_state.gemini_api_error
                        st.session_state.gemini_api_requested = False # 処理完了したのでフラグをリセット
                        st.rerun() # 結果を表示するために再実行

            if "gemini_api_error" in st.session_state and st.session_state.gemini_api_error: # エラーがあれば表示
                st.error(st.session_state.gemini_api_error)
            elif "gemini_explanation" in st.session_state and st.session_state.gemini_explanation: # 解説があれば表示
                with st.expander("Gemini API によるログ解説", expanded=True):
                    st.markdown(st.session_state.gemini_explanation)
        else:
            # この状態は、例えば「サンプルデータ」表示後に view_mode が "optimization_result" になったが、
            # solution_executed が False の場合など (通常は「最適割り当てを実行」で True になる)
            st.info("サイドバーの「最適割り当てを実行」ボタンを押して最適化を実行してください。")
    else: # view_mode が予期せぬ値の場合 (フォールバック)
        st.info("サイドバーから表示するデータを選択してください。")

if __name__ == "__main__":
    main()
