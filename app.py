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
# (get_gemini_explanation 関数は変更なしのため省略)
# ... (get_gemini_explanation の元のコード) ...

# --- データ生成関数群 ---
# これらの関数は initialize_app_data 内で呼び出され、結果を st.session_state に格納する
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

def generate_lecturers_data(prefecture_classroom_ids, today_date):
    lecturers_data = []
    DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    TIMES = ["AM", "PM"]
    ALL_SLOTS = [(day, time) for day in DAYS for time in TIMES]

    for i in range(1, 301): # 講師数を100人から300人に変更
        num_available_slots = random.randint(3, 7)
        availability = random.sample(ALL_SLOTS, num_available_slots)
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

def generate_courses_data(prefectures, prefecture_classroom_ids):
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

    # 各都道府県で生成する講座の種類とスケジュールを定義
    # (例として、一般講座3つ、特別講座2つを異なる時間帯で生成)
    COURSE_SCHEDULE_TEMPLATES = [
        {"type": "general", "level_idx": 0, "schedule": ("Mon", "AM"), "id_suffix_prefix": "GC"}, # 一般講座 ランク5 (初心)
        {"type": "general", "level_idx": 2, "schedule": ("Tue", "PM"), "id_suffix_prefix": "GC"}, # 一般講座 ランク3 (中級)
        {"type": "general", "level_idx": 4, "schedule": ("Wed", "AM"), "id_suffix_prefix": "GC"}, # 一般講座 ランク1 (プロ)
        {"type": "special", "level_idx": 1, "schedule": ("Thu", "PM"), "id_suffix_prefix": "SC"}, # 特別講座 ランク4 (初級)
        {"type": "special", "level_idx": 3, "schedule": ("Fri", "AM"), "id_suffix_prefix": "SC"}, # 特別講座 ランク2 (上級)
    ]

    courses_data = []
    course_counter = 1
    for i, pref_classroom_id in enumerate(prefecture_classroom_ids):
        pref_name = prefectures[i]
        for template in COURSE_SCHEDULE_TEMPLATES:
            level_info = None
            course_type_name = ""
            if template["type"] == "general":
                level_info = GENERAL_COURSE_LEVELS[template["level_idx"]]
                course_type_name = "一般講座"
            else: # special
                level_info = SPECIAL_COURSE_LEVELS[template["level_idx"]]
                course_type_name = "特別講座"

            courses_data.append({
                "id": f"{pref_classroom_id}-{template['id_suffix_prefix']}{course_counter}",
                "name": f"{pref_name} {course_type_name} {level_info['name_suffix']}",
                "classroom_id": pref_classroom_id,
                "course_type": template["type"],
                "rank": level_info['rank'],
                "schedule": template["schedule"]
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

# スケジュール違反に対する固定ペナルティ (floatで定義し、他の重み付けコストと合算)
# この値は、スケジュール違反を許容する場合に、他のコスト要因よりも優先度が低くなるように十分に大きく設定します。
BASE_PENALTY_SCHEDULE_VIOLATION = 1000000.0  # 例: 100万

# 過去の割り当てがない、または日付パース不能な場合に設定するデフォルトの経過日数 (ペナルティ計算上、十分に大きい値)
# これは initialize_app_data で st.session_state.DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT に設定される

# --- 2. OR-Tools 最適化ロジック ---
class SolverOutput(TypedDict): # 提案: 戻り値を構造化するための型定義
    solution_status_str: str
    objective_value: Optional[float]
    assignments: List[dict]
    all_courses: List[dict]
    all_lecturers: List[dict]
    solver_raw_status_code: int
    full_application_and_solver_log: str # All logs including detailed app logs for UI's explained_log_text

def solve_assignment(lecturers_data, courses_data, classrooms_data, # classrooms_data は現在未使用だが、将来のために残す
                     travel_costs_matrix, # frequency_priority_costs を削除
                     weight_past_assignment_recency, weight_qualification, 
                     ignore_schedule_constraint: bool, # スケジュール制約を無視するかのフラグ
                     weight_travel, weight_age, weight_frequency, # 既存の重み
                     weight_lecturer_concentration, # 新しい重み: 講師の割り当て集中度
                     today_date, default_days_no_past_assignment) -> SolverOutput: # 引数追加
    model = cp_model.CpModel()
    # solve_assignment 内の print 文も解説対象に含めるために、
    # ここで stdout のキャプチャを開始する
    full_log_stream = io.StringIO()

    # アプリケーションログを full_log_stream に直接書き込むように変更
    def log_to_stream(message):
        print(message, file=full_log_stream)
        # print(message) # ターミナルにも表示（デバッグ用） - 大量ログの場合、パフォーマンスに影響する可能性

    # 講師の追加割り当てに対するペナルティの基準値 (スケーリング前)
    # 他のコスト要素（移動コストなど）の代表的な値に基づいて設定
    PENALTY_PER_EXTRA_ASSIGNMENT_RAW = 5000 
    # --- Main logic for model building and solving ---
    possible_assignments = []
    potential_assignment_count = 0
    log_to_stream(f"Initial lecturers: {len(lecturers_data)}, Initial courses: {len(courses_data)}")

    for lecturer in lecturers_data:
        for course in courses_data:
            lecturer_id = lecturer["id"]
            course_id = course["id"]

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
            qualification_cost = qualification_cost_for_this_assignment # 上で計算した、この割り当てにおける資格コスト

            # 過去割り当ての近さによるコスト計算
            past_assignment_recency_cost = 0
            days_since_last_assignment_to_classroom = default_days_no_past_assignment # 引数から取得

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
                        days_since_last_assignment_to_classroom = (today_date - latest_assignment_date).days # 引数から取得
                        
                        # コスト計算: 経過日数が少ないほど高いコスト
                        # (DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT - 経過日数)
                        # 経過日数が DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT の場合、コストは0
                        raw_recency_cost = default_days_no_past_assignment - days_since_last_assignment_to_classroom
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
            total_weighted_cost_int = int(total_weighted_cost_float * 100) # コストを整数にスケーリング # type: ignore
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
            
    # 【変更点】各講師の担当上限制約 (<=1) を削除
    # for lecturer_item in lecturers_data:
    #     lecturer_id = lecturer_item["id"]
    #     assignments_for_lecturer = [pa["variable"] for pa in possible_assignments if pa["lecturer_id"] == lecturer_id]
    #     if assignments_for_lecturer:
    #         model.Add(sum(assignments_for_lecturer) <= 1)

    # 【新規】講師の割り当て集中度に対するペナルティ項を目的関数に追加
    concentration_penalty_terms = []
    for lecturer_item in lecturers_data:
        lecturer_id = lecturer_item["id"]
        assignments_for_lecturer_vars = [pa["variable"] for pa in possible_assignments if pa["lecturer_id"] == lecturer_id]
        if not assignments_for_lecturer_vars:
            continue

        num_assignments_l = model.NewIntVar(0, len(courses_data), f'num_assignments_{lecturer_id}')
        model.Add(num_assignments_l == sum(assignments_for_lecturer_vars))

        # extra_assignments_l = max(0, num_assignments_l - 1)
        extra_assignments_l = model.NewIntVar(0, len(courses_data), f'extra_assignments_{lecturer_id}')
        model.Add(extra_assignments_l >= num_assignments_l - 1)
        # IntVarの下限が0なので model.Add(extra_assignments_l >= 0) は不要

        cost_value_for_extra_assignment = int(weight_lecturer_concentration * PENALTY_PER_EXTRA_ASSIGNMENT_RAW * 100)
        if cost_value_for_extra_assignment > 0: # 重みが0の場合はペナルティなし
            concentration_penalty_terms.append(extra_assignments_l * cost_value_for_extra_assignment)
            log_to_stream(f"  - Lecturer Concentration Penalty: For {lecturer_id}, cost per extra assignment (after 1st) = {cost_value_for_extra_assignment} (raw penalty base: {PENALTY_PER_EXTRA_ASSIGNMENT_RAW}, weight: {weight_lecturer_concentration})")

    assignment_costs = [pa["variable"] * pa["cost"] for pa in possible_assignments]
    # 未割り当てペナルティ (penalty_terms) を削除
    objective_terms = assignment_costs
    if objective_terms:
        model.Minimize(sum(objective_terms))
    else:
        if concentration_penalty_terms: # 割り当てコストがなくても集中ペナルティだけはある場合
            model.Minimize(sum(concentration_penalty_terms))
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

    # 目的関数に集中度ペナルティ項を追加 (assignment_costs が空でもこちらが評価されるように修正)
    if concentration_penalty_terms:
        objective_terms.extend(concentration_penalty_terms)
    model.Minimize(sum(objective_terms)) # 常に Minimize を呼び出す (objective_termsが空なら0を最小化)
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True

    log_to_stream("--- Solver Log (Captured by app.py) ---")
    
    status_code = cp_model.UNKNOWN # Initialize status_code
    with contextlib.redirect_stdout(full_log_stream):
        status_code = solver.Solve(model)
    
    log_to_stream("--- End Solver Log (Captured by app.py) ---")

    full_captured_logs = full_log_stream.getvalue()

    status_name = solver.StatusName(status_code) # Get the status name
    results = []
    objective_value = None
    solution_status_str = "解なし"

    if status_code == cp_model.OPTIMAL or status_code == cp_model.FEASIBLE:
        solution_status_str = "最適解" if status_code == cp_model.OPTIMAL else "実行可能解"
        objective_value = solver.ObjectiveValue() / 100 # スケーリングを戻す # type: ignore
        
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
                    "資格コスト(元)": pa.get("qualification_cost_raw"), # この割り当てで評価された資格コスト
                    "当該教室最終割当日からの日数": pa.get("debug_days_since_last_assignment"),
                    "講師一般ランク": lecturer.get("qualification_general_rank"),
                    "講師特別ランク": lecturer.get("qualification_special_rank", "なし"),
                    "講座タイプ": course.get("course_type"),
                    "講座ランク": course.get("rank")
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
def initialize_app_data():
    """アプリケーションの初期データを生成し、セッション状態に保存する。"""
    if "app_data_initialized" not in st.session_state:
        st.session_state.TODAY = datetime.date.today()
        st.session_state.DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT = 100000

        PREFECTURES_val, PREFECTURE_CLASSROOM_IDS_val = generate_prefectures_data()
        st.session_state.PREFECTURES = PREFECTURES_val
        st.session_state.PREFECTURE_CLASSROOM_IDS = PREFECTURE_CLASSROOM_IDS_val

        st.session_state.DEFAULT_CLASSROOMS_DATA = generate_classrooms_data(
            st.session_state.PREFECTURES, st.session_state.PREFECTURE_CLASSROOM_IDS
        )
        st.session_state.ALL_CLASSROOM_IDS_COMBINED = st.session_state.PREFECTURE_CLASSROOM_IDS

        st.session_state.DEFAULT_LECTURERS_DATA = generate_lecturers_data(
            st.session_state.PREFECTURE_CLASSROOM_IDS, st.session_state.TODAY
        )
        st.session_state.DEFAULT_COURSES_DATA = generate_courses_data(
            st.session_state.PREFECTURES, st.session_state.PREFECTURE_CLASSROOM_IDS
        )

        REGIONS = {
            "Hokkaido": ["北海道"], "Tohoku": ["青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県"],
            "Kanto": ["茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県"],
            "Chubu": ["新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県", "静岡県", "愛知県"],
            "Kinki": ["三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県"],
            "Chugoku": ["鳥取県", "島根県", "岡山県", "広島県", "山口県"],
            "Shikoku": ["徳島県", "香川県", "愛媛県", "高知県"],
            "Kyushu_Okinawa": ["福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"]
        }
        st.session_state.PREFECTURE_TO_REGION = {pref: region for region, prefs in REGIONS.items() for pref in prefs}
        st.session_state.REGION_GRAPH = {
            "Hokkaido": {"Tohoku"}, "Tohoku": {"Hokkaido", "Kanto", "Chubu"},
            "Kanto": {"Tohoku", "Chubu"}, "Chubu": {"Tohoku", "Kanto", "Kinki"},
            "Kinki": {"Chubu", "Chugoku", "Shikoku"}, "Chugoku": {"Kinki", "Shikoku", "Kyushu_Okinawa"},
            "Shikoku": {"Kinki", "Chugoku", "Kyushu_Okinawa"}, "Kyushu_Okinawa": {"Chugoku", "Shikoku"}
        }
        st.session_state.CLASSROOM_ID_TO_PREF_NAME = {
            item["id"]: item["location"] for item in st.session_state.DEFAULT_CLASSROOMS_DATA
        }
        st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX = generate_travel_costs_matrix(
            st.session_state.ALL_CLASSROOM_IDS_COMBINED,
            st.session_state.CLASSROOM_ID_TO_PREF_NAME,
            st.session_state.PREFECTURE_TO_REGION,
            st.session_state.REGION_GRAPH
        )
        st.session_state.app_data_initialized = True

def main():
    st.set_page_config(page_title="講師割り当てシステムデモ", layout="wide")
    initialize_app_data() # アプリケーションデータの初期化

    # --- OIDC認証設定 ---
    GOOGLE_CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = st.secrets.get("GOOGLE_CLIENT_SECRET")
    REDIRECT_URI = st.secrets.get("REDIRECT_URI")
    ALLOWED_EMAIL = "asaumi.ctc@gmail.com" # 許可するメールアドレス
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

    AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
    # REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke" # 必要に応じて

    if not all([GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REDIRECT_URI]): # type: ignore
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
        if result and "token" in result: # type: ignore
            st.session_state.token = result.get("token")
            try:
                # IDトークンを取得して検証
                id_token_str = st.session_state.token.get("id_token")
                if id_token_str:
                    id_info = id_token.verify_oauth2_token(
                        id_token_str,
                        google_requests.Request(),
                        GOOGLE_CLIENT_ID # type: ignore
                    )
                    user_email = id_info.get("email")
                    if user_email == ALLOWED_EMAIL:
                        st.session_state.user_info = {
                            "email": user_email,
                            "name": id_info.get("name")
                        }
                    else:
                        st.error(f"このアプリケーションへのアクセスは許可されていません。({user_email})")
                        st.session_state.token = None # トークンをクリアしてログイン状態を解除
                        st.session_state.user_info = None
                else:
                    st.error("IDトークンが取得できませんでした。")
                    st.session_state.token = None # トークンをクリア
                    st.session_state.user_info = {"email": "error@example.com", "name": "Unknown User"}
            except Exception as e:
                st.error(f"ユーザー情報の取得/設定中にエラー: {e}")
                st.session_state.token = None # トークンをクリア
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
    # ボタン幅を広げ、文字列が改行されないように比率を調整 (例: [1,1,5] -> [2,2,3])
    nav_cols = st.columns([2, 2, 2, 1])  # ボタン数を3つに合わせ、比率を調整
    with nav_cols[0]:
        # サンプルデータボタン
        button_type_sample = "primary" if st.session_state.view_mode == "sample_data" else "secondary"
        if st.button("サンプルデータ", key="nav_sample_data_button", use_container_width=True, type=button_type_sample):
            st.session_state.view_mode = "sample_data"
            # サンプルデータ表示時、最適化結果の主要キャッシュ(solver_result_cache, raw_log_on_server)は保持する。
            # Gemini API関連のキャッシュのみクリアする。
            keys_to_clear_for_sample_view = [
                "gemini_explanation",
                "gemini_api_requested",
                "gemini_api_error",
            ]
            for key_to_clear in keys_to_clear_for_sample_view:
                if key_to_clear in st.session_state:
                    del st.session_state[key_to_clear]
            st.rerun()

    with nav_cols[1]:
        # 目的関数の数式ボタン
        button_type_objective = "primary" if st.session_state.view_mode == "objective_function" else "secondary"
        if st.button("ソルバーとmodelオブジェクト", key="nav_solver_model_object_button", use_container_width=True, type=button_type_objective): # ボタン名を変更
            st.session_state.view_mode = "objective_function"
            st.rerun()

    with nav_cols[2]:
        # 「最適化結果」ボタンは、最適化が一度でも実行された後にのみ表示
        if st.session_state.get("solution_executed", False): 
            button_type_result = "primary" if st.session_state.view_mode == "optimization_result" else "secondary"
            if st.button("最適化結果", key="nav_optimization_result_button", use_container_width=True, type=button_type_result):
                st.session_state.view_mode = "optimization_result"
                st.rerun()
        # else: # solution_executed が False の場合 (初期状態など) は「最適化結果」ボタンをレンダリングしない

    st.sidebar.markdown(
        "【制約】と【目的】を設定すれば、数理モデル最適化手法により自動的に最適な講師割り当てを実行します。"
        "また目的に重み付けすることでチューニングすることができます。"
    )
    
    # 「最適割り当てを実行」ボタンを説明文の直下に移動
    if st.sidebar.button("最適割り当てを実行", type="primary", key="execute_optimization_main_button"):
        # 既存の計算結果関連のセッション変数をクリア
        keys_to_clear_on_execute = [
            "solver_result_cache", "raw_log_on_server", "gemini_explanation",
            "gemini_api_requested", "gemini_api_error"
        ]
        for key in keys_to_clear_on_execute:
            if key in st.session_state:
                del st.session_state[key]
        
        try:
            # ここで最適化計算を実行し、結果をキャッシュに保存する
            with st.spinner("最適化計算を実行中..."):
                solver_output = solve_assignment(
                    st.session_state.DEFAULT_LECTURERS_DATA, st.session_state.DEFAULT_COURSES_DATA,
                    st.session_state.DEFAULT_CLASSROOMS_DATA, st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX,
                    st.session_state.get("weight_past_assignment_exp", 0.5), # スライダーのキー名で取得
                    st.session_state.get("weight_qualification_exp", 0.5),  # スライダーのキー名で取得
                    st.session_state.get("ignore_schedule_constraint_checkbox_value", True), # チェックボックスの値を取得
                    st.session_state.get("weight_travel_exp", 0.5),         # スライダーのキー名で取得
                    st.session_state.get("weight_age_exp", 0.5),            # スライダーのキー名で取得
                    st.session_state.get("weight_frequency_exp", 0.5),       # スライダーのキー名で取得
                    st.session_state.get("weight_lecturer_concentration_exp", 0.5), # 新しい重み
                    st.session_state.TODAY, # 追加
                    st.session_state.DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT # 追加
                )
            
            # solver_output の検証を追加
            if not isinstance(solver_output, dict):
                st.error(f"最適化関数の戻り値が不正です (辞書ではありません)。型: {type(solver_output)}")
                st.session_state.solution_executed = True
                st.session_state.view_mode = "optimization_result"
                st.rerun()
                return # tryブロックを抜ける (st.rerunがあるので実際には不要だが念のため)

            required_keys = ["full_application_and_solver_log", "solution_status_str", 
                             "objective_value", "assignments", "all_courses", 
                             "all_lecturers", "solver_raw_status_code"]
            missing_keys = [key for key in required_keys if key not in solver_output]
            if missing_keys:
                st.error(f"最適化関数の戻り値に必要なキーが不足しています。不足キー: {missing_keys}。取得キー: {list(solver_output.keys())}")
                st.session_state.solution_executed = True
                st.session_state.view_mode = "optimization_result"
                st.rerun()
                return

            # 検証が通れば、結果を保存
            st.session_state.raw_log_on_server = solver_output["full_application_and_solver_log"]
            # solver_result_cache には、SolverOutput のキーから full_application_and_solver_log を除いたものを格納
            st.session_state.solver_result_cache = {
                k: solver_output[k] for k in required_keys if k != "full_application_and_solver_log" # type: ignore
            }
            st.session_state.solution_executed = True # 実行フラグを立てる
            st.session_state.view_mode = "optimization_result" # 表示モードを最適化結果に

        except Exception as e:
            st.error(f"最適化処理中または結果の格納中に予期せぬエラーが発生しました: {e}")
            # エラーが発生しても、実行試行済みとし、結果表示画面へ遷移させる
            # solver_result_cache は設定されないため、結果表示画面で「データなし」の警告が表示される
            # デバッグ用にスタックトレースを表示することも検討できます:
            # import traceback
            # st.text_area("エラー詳細:", traceback.format_exc(), height=300)
            st.session_state.solution_executed = True 
            st.session_state.view_mode = "optimization_result"

        st.rerun() # 再実行してメインエリアで処理と表示を行う

    st.sidebar.markdown("---")
    with st.sidebar.expander("【制約】", expanded=False):
        st.markdown("**ハード制約（絶対固定）**")
        st.markdown("- 講師は資格ランクに応じた講座しか割り当てできません。")
        # st.markdown("- 講師は（今回の割り当てでは）1つの講座しか担当できません。") # この行を削除

        st.markdown("**ソフト制約（割り当てできない場合に許容できる）**")
        st.markdown(
            "講師の空きスケジュールに応じた講座しか割り当てできないことが原則ですが、"
            "完全割り当てのために以下のことを許容できます。"
        )
        st.session_state.ignore_schedule_constraint_checkbox_value = st.checkbox( # 値をセッションステートに保存
            "講師の空きスケジュールを無視する", 
            value=True, 
            help="チェックを外すと割り当て結果が得られないことがあります。その場合は、講師の空きスケジュールが合わない場合が想定されます。"
        )

    with st.sidebar.expander("【目的】", expanded=False):
        st.caption(
            "【重み】不要な目的はゼロにしてください。（目的から除外されます）"
            "また、相対的な値なので、全部0.1と全部1.0は同じ結果となります。"
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
        st.markdown("**講師の割り当て集中度を低くする**")
        st.slider("重み", 0.0, 1.0, 0.5, 0.1, format="%.1f", help="高いほど、一人の講師が複数の講座を担当することへのペナルティが大きくなります。値が高いほど、各講師は1つの講座に近づきます。", key="weight_lecturer_concentration_exp")

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
            "gemini_api_error",     # Gemini APIエラーメッセージ
            "app_data_initialized"  # アプリデータ初期化フラグもクリア
        ]
        for key_to_clear in keys_to_clear:
            if key_to_clear in st.session_state:
                del st.session_state[key_to_clear]
        st.rerun()

    st.title("講師割り当てシステム(OR-Tools)-プロトタイプ")

    # --- メインエリアの表示制御 ---
    if st.session_state.view_mode == "sample_data":
        st.header("入力データ")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("講師データ (サンプル)")
            # st.session_state からデータを取得
            df_lecturers_display = pd.DataFrame(st.session_state.DEFAULT_LECTURERS_DATA)
            # 表示用に 'qualification_special_rank' が None の場合は "なし" に変換
            if 'qualification_special_rank' in df_lecturers_display.columns:
                df_lecturers_display['qualification_special_rank'] = df_lecturers_display['qualification_special_rank'].apply(lambda x: "なし" if x is None else x)
            if 'past_assignments' in df_lecturers_display.columns:
                df_lecturers_display['past_assignments'] = df_lecturers_display['past_assignments'].apply(
                    lambda assignments: ", ".join([f"{a['classroom_id']} ({a['date']})" for a in assignments]) if isinstance(assignments, list) and assignments else "履歴なし"
                )
            # 表示するカラムを調整
            lecturer_display_columns = ["id", "name", "age", "home_classroom_id", "qualification_general_rank", "qualification_special_rank", "availability", "past_assignments"]
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

    elif st.session_state.view_mode == "objective_function":
        st.header("ソルバーとmodelオブジェクト") # ヘッダー名を変更

        st.markdown(
            r"""
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
            st.markdown(
                r"""
                CP-SATソルバーがこの講師割り当て問題に適している主な理由は以下の通りです。

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

                これらの特性と他のソルバーとの比較から、CP-SATソルバーは本問題に対して効率的かつ効果的な解を提供するための強力な選択肢となります。
                """
            )
        st.markdown("---")
        st.subheader("model オブジェクトの構成要素")

        st.subheader("1. 決定変数 (Decision Variables)")
        st.markdown(
            r"""
            決定変数は、ソルバーが最適解を見つけるために値を決定する変数です。
            この講師割り当て問題では、各講師 $l$ が各講座 $c$ に割り当てられるかどうかを示すバイナリ変数 $x_{l,c}$ を使用します。

            $$
            x_{l,c} \in \{0, 1\} \quad (\forall l \in L, \forall c \in C)
            $$

            ここで、
            - $L$ は講師の集合
            - $C$ は講座の集合
            - $x_{l,c} = 1$ ならば、講師 $l$ は講座 $c$ に割り当てられます。
            - $x_{l,c} = 0$ ならば、講師 $l$ は講座 $c$ に割り当てられません。
            """
        )
        st.markdown("**対応するPythonコード (抜粋):**")
        st.code(
            """
# ... ループ内で各講師と講座の組み合わせに対して ...
var = model.NewBoolVar(f'x_{lecturer_id}_{course_id}')
possible_assignments.append({"variable": var, ...})
            """, language="python"
        )
        with st.expander("コード解説", expanded=False):
            st.markdown(
                """
                - `model.NewBoolVar(f'x_{lecturer_id}_{course_id}')` は、各講師 (`lecturer_id`) と各講座 (`course_id`) の組み合わせに対して、0または1の値を取るブール変数（決定変数）を作成します。
                - 変数名は、デバッグしやすいように講師IDと講座IDを含む一意な文字列 (`x_L1_C1` など）としています。
                - 作成された変数は、他の情報（講師ID、講座ID、後で計算されるコストなど）と共に `possible_assignments` リストに辞書として格納され、後で制約や目的関数の定義に使用されます。
                """
            )

        st.subheader("2. 制約 (Constraints)")
        st.markdown(
            r"""
            制約は、決定変数が取りうる値の範囲や、変数間の関係を定義する条件です。
            これにより、実行可能な解（許容される割り当てパターン）の範囲が定まります。

            **主な制約:**
            - **各講座への割り当て制約:** 各講座 $c$ には、担当可能な講師候補が存在する場合、必ず1人の講師が割り当てられます。
              $$
              \sum_{l \in L_c} x_{l,c} = 1 \quad (\forall c \in C \text{ s.t. } L_c \neq \emptyset)
              $$
              ここで、$L_c$ は講座 $c$ を担当可能な講師の集合です（資格ランクやスケジュール（無視しない場合）を考慮）。

            - **各講師の担当上限制約:** 各講師 $l$ は、最大で1つの講座しか担当できません。
              $$
              \sum_{c \in C} x_{l,c} \leq 1 \quad (\forall l \in L)
              $$
            
            - **暗黙的な制約:** ソースコード上では、講師の資格ランクが講座の要求ランクを満たさない場合や、スケジュールが適合しない（かつスケジュール制約を無視する設定でない）組み合わせは、そもそも上記の決定変数 $x_{l,c}$ が生成される前の段階で除外されます。これは、それらの組み合わせに対する $x_{l,c}$ が実質的に 0 に固定される制約と見なせます。
            """
        )
        st.markdown("**対応するPythonコード (抜粋):**")
        st.code(
            """
# 各講座への割り当て制約
for course_item in courses_data:
    possible_assignments_for_course = [pa["variable"] for pa in possible_assignments if pa["course_id"] == course_item["id"]]
    if possible_assignments_for_course:
        model.Add(sum(possible_assignments_for_course) == 1)

# 各講師の担当上限制約
for lecturer_item in lecturers_data:
    assignments_for_lecturer = [pa["variable"] for pa in possible_assignments if pa["lecturer_id"] == lecturer_item["id"]]
    if assignments_for_lecturer:
        model.Add(sum(assignments_for_lecturer) <= 1)
            """, language="python"
        )
        with st.expander("コード解説", expanded=False):
            st.markdown(
                """
                **各講座への割り当て制約:**
                - `for course_item in courses_data:`: 全ての講座に対してループ処理を行います。
                - `possible_assignments_for_course = [...]`: 特定の講座 `course_item["id"]` に割り当て可能な全ての決定変数をリストアップします。
                - `if possible_assignments_for_course:`: その講座に割り当て可能な講師候補が1人以上いる場合のみ、制約を追加します。
                - `model.Add(sum(possible_assignments_for_course) == 1)`: その講座に割り当てられる講師の総数がちょうど1人になるように制約を設定します。つまり、必ず1人の講師が割り当てられます（候補がいる場合）。

                **各講師の担当上限制約:**
                - `for lecturer_item in lecturers_data:`: 全ての講師に対してループ処理を行います。
                - `assignments_for_lecturer = [...]`: 特定の講師 `lecturer_item["id"]` が担当する可能性のある全ての決定変数をリストアップします。
                - `model.Add(sum(assignments_for_lecturer) <= 1)`: その講師が担当する講座の総数が1つ以下になるように制約を設定します。つまり、各講師は最大で1つの講座しか担当できません。
                """
            )

        st.subheader("3. 目的関数 (Objective Function)")
        st.markdown(
            r"""
            目的関数は、最適化の目標を定義する数式です。この問題では、割り当ての総コストを最小化することが目的です。

            $$
            \text{Minimize} \quad Z = \sum_{l \in L} \sum_{c \in C} x_{l,c} \cdot \text{Cost}_{l,c}
            $$

            ここで、$\text{Cost}_{l,c}$ は講師 $l$ が講座 $c$ に割り当てられた場合の個別のコストで、以下のように計算されます。

            $$
            \text{Cost}_{l,c} = w_{\text{travel}} \cdot \text{TravelCost}_{l,c} + w_{\text{age}} \cdot \text{AgeCost}_l + w_{\text{frequency}} \cdot \text{FrequencyCost}_l + w_{\text{qualification}} \cdot \text{QualificationCost}_l + w_{\text{recency}} \cdot \text{RecencyCost}_{l,c}  + \text{ScheduleViolationPenalty}_{l,c}
            $$

            各記号の意味:
            - $w_{\text{...}}$: 各コスト要素に対する重み（サイドバーで設定可能）
            - $\text{TravelCost}_{l,c}$: 講師 $l$ と講座 $c$ の教室間の移動コスト
            - $\text{AgeCost}_l$: 講師 $l$ の年齢に基づくコスト
            - $\text{FrequencyCost}_l$: 講師 $l$ の過去の総割り当て頻度に基づくコスト
            - $\text{QualificationCost}_l$: 講師 $l$ の資格ランクに基づくコスト
            - $\text{RecencyCost}_{l,c}$: 講師 $l$ が講座 $c$ の教室に最後に割り当てられてからの経過日数に基づくコスト
            - $\text{ScheduleViolationPenalty}_{l,c}$: 講師 $l$ と講座 $c$ のスケジュールが不適合な場合に発生するペナルティ（スケジュール制約を無視する場合）
            """
        )
        st.markdown("**対応するPythonコード (抜粋):**")
        st.code(
            """
# 各割り当て候補のコスト計算 (total_weighted_cost_int)
# total_weighted_cost_float = (weight_travel * travel_cost + ...) + schedule_violation_penalty
# total_weighted_cost_int = int(total_weighted_cost_float * 100)
# possible_assignments.append({..., "cost": total_weighted_cost_int, ...})

# 目的関数の設定
assignment_costs = [pa["variable"] * pa["cost"] for pa in possible_assignments]
objective_terms = assignment_costs
if objective_terms:
    model.Minimize(sum(objective_terms))
            """, language="python"
        )
        with st.expander("コード解説", expanded=False):
            st.markdown(
                """
                - **各割り当て候補のコスト計算**:
                    - `total_weighted_cost_float = ...`: 各コスト要素（移動、年齢、頻度など）に、ユーザーが設定した重みを掛け合わせ、それらを合計して、その割り当ての総フロートコストを計算します。スケジュール違反がある場合は、ここでペナルティも加算されます。
                    - `total_weighted_cost_int = int(total_weighted_cost_float * 100)`: CP-SATソルバーは整数演算を基本とするため、計算されたフロートコストを100倍して整数に変換（スケーリング）します。
                    - `possible_assignments.append({..., "cost": total_weighted_cost_int, ...})`: 計算された整数コストを、対応する決定変数などと一緒に `possible_assignments` リストに格納します。
                - **目的関数の設定**:
                    - `assignment_costs = [pa["variable"] * pa["cost"] for pa in possible_assignments]`: `possible_assignments` リスト内の各割り当て候補について、その割り当てが選択された場合（`pa["variable"]` が1の場合）のコスト (`pa["cost"]`) を計算し、リスト `assignment_costs` を作成します。
                    - `objective_terms = assignment_costs`: この問題では、割り当てコストの合計が目的関数の全てです。
                    - `model.Minimize(sum(objective_terms))`: ソルバーに対して、`objective_terms` の合計値（つまり、選択された全ての割り当ての総コスト）を最小化するように指示します。
                """
            )

    elif st.session_state.view_mode == "optimization_result":
        st.header("最適化結果") # ヘッダーは最初に表示

        if not st.session_state.get("solution_executed", False):
            st.info("サイドバーの「最適割り当てを実行」ボタンを押して最適化を実行してください。")
        else: # solution_executed is True
            if "solver_result_cache" not in st.session_state:
                # solution_executed is True, but no cache (e.g., after viewing sample data, which clears the cache)
                # この場合、再計算は行わず、キャッシュがない旨をユーザーに伝える (メッセージを簡潔に)
                st.warning(
                    "最適化結果のデータは現在ありません。\n"
                    "再度結果を表示するには、サイドバーの「最適割り当てを実行」ボタンを押してください。"
                )
            else: # solution_executed is True and solver_result_cache exists
                # キャッシュされた結果（または計算直後の結果）を取得
                # このブロックは、以前のコードで再計算ブロックの後から始まっていた部分に相当
                solver_result = st.session_state.solver_result_cache
                st.subheader(f"求解ステータス: {solver_result['solution_status_str']}")
                if solver_result['objective_value'] is not None:
                    st.metric("総コスト (目的値)", f"{solver_result['objective_value']:.2f}")

                # 「全ての講座が割り当てられました」メッセージをここに移動
                if solver_result['solver_raw_status_code'] == cp_model.OPTIMAL or solver_result['solver_raw_status_code'] == cp_model.FEASIBLE:
                    if solver_result['assignments']:
                        assigned_course_ids_for_message = {res["講座ID"] for res in solver_result['assignments']}
                        unassigned_courses_for_message = [c for c in solver_result['all_courses'] if c["id"] not in assigned_course_ids_for_message]
                        if not unassigned_courses_for_message:
                            st.success("全ての講座が割り当てられました。")
                    # assignments が空の場合のメッセージは後続のロジックで表示されるため、ここでは不要

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
                    # サマリー計算時も st.session_state のデータを使用
                    temp_assigned_lecturers = [l for l in st.session_state.DEFAULT_LECTURERS_DATA if l["id"] in assigned_lecturer_ids]
                    if temp_assigned_lecturers:
                        avg_age = sum(l.get("age", 0) for l in temp_assigned_lecturers) / len(temp_assigned_lecturers)
                        summary_data.append(("**平均年齢**", f"{avg_age:.1f}才"))
                        avg_frequency = sum(len(l.get("past_assignments", [])) for l in temp_assigned_lecturers) / len(temp_assigned_lecturers)
                        summary_data.append(("**平均頻度**", f"{avg_frequency:.1f}回"))

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

                    past_assignment_new_count = results_df[results_df["当該教室最終割当日からの日数"] == st.session_state.DEFAULT_DAYS_FOR_NO_OR_INVALID_PAST_ASSIGNMENT].shape[0] # st.session_state から取得
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
                        st.dataframe(results_df_display) # type: ignore
                        assigned_course_ids = {res["講座ID"] for res in solver_result['assignments']}
                        unassigned_courses = [c for c in solver_result['all_courses'] if c["id"] not in assigned_course_ids]
                        if unassigned_courses:
                            st.subheader("割り当てられなかった講座")
                            st.dataframe(pd.DataFrame(unassigned_courses))
                            st.caption("上記の講座は、スケジュール違反を許容しても、他の制約（資格ランクなど）により割り当て可能な講師が見つからなかったか、または他の割り当てと比較してコストが高すぎると判断された可能性があります。")
                        # else: # 「全ての講座が割り当てられました」メッセージは上で表示済のため削除
                            # st.success("全ての講座が割り当てられました。")
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
                elif st.session_state.get("solution_executed"): # ボタンが表示されない場合のヒント (最適化実行後だが、raw_log_on_server がない場合など)
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
                            solver_cache = st.session_state.solver_result_cache # この時点では solver_result と同じ
                            solver_status = solver_cache["solution_status_str"]
                            objective_value = solver_cache["objective_value"]
                            assignments_list = solver_cache.get("assignments", [])
                            assignments_summary_df = pd.DataFrame(assignments_list) if assignments_list else None # type: ignore

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

    else: # view_mode が予期せぬ値の場合 (フォールバック)
        st.info("サイドバーから表示するデータを選択してください。")

if __name__ == "__main__":
    main()
