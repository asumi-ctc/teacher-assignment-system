import streamlit as st
from ortools.sat.python import cp_model
import pandas as pd
import io
import contextlib
import re # 正規表現モジュール
import datetime # 日付処理用に追加
import google.generativeai as genai # Gemini API 用
from streamlit_oauth import OAuth2Component # OIDC認証用
import random # データ生成用

# アプリケーションバージョン
APP_VERSION = "1.1.3" # 例: バージョン番号を定義

# --- 1. データ定義 ---

# ログパターンと解説の対応辞書 (log_explainer.py から移動)
LOG_EXPLANATIONS = {
    # 例: "Error Code 404: File not found" のようなログに対応
    r"Error Code (\d+): File not found": "指定されたファイルが見つかりませんでした。(エラーコード: \\1)",
    # 例: "Warning: Low disk space" のようなログに対応
    r"Warning: Low disk space": "ディスクの空き容量が少なくなっています。不要なファイルを削除してください。",
    # 例: "INFO: User 'admin' logged in" のようなログに対応
    r"INFO: User '(\w+)' logged in": "ユーザー「\\1」がログインしました。",
    # 例: "DEBUG: Connection established to example.com:8080" のようなログに対応
    r"DEBUG: Connection established to ([\w.:-]+)": "サーバー「\\1」への接続が確立されました。",
    r"Failed to process task ID (\w+-\w+)": "タスクID「\\1」の処理に失敗しました。詳細を確認してください。",
    r"Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.": "Streamlit が利用状況統計を収集しています。無効化するには browser.gatherUsageStats を false に設定します。",
    r"You can now view your Streamlit app in your browser.": "Streamlit アプリがブラウザで表示可能になりました。",
    r"Local URL: (http://localhost:\d+)": "ローカルURL: \\1 (開発用マシンからアクセス)",
    r"Network URL: (http://[\d\.]+:\d+)": "ネットワークURL: \\1 (同一ネットワーク内の他のデバイスからアクセス可能)",
    r"External URL: (http://[\d\.]+:\d+)": "外部URL: \\1 (インターネット経由でアクセス可能 - 注意して使用)",
    r"Initial lecturers: (\d+), Initial courses: (\d+)": "初期講師数: \\1, 初期コース数: \\2",
    r"Length of possible_assignments list \(with variables\): (\d+)": "変数を含む割り当て候補リストの長さ: \\1",
    r"^\s*\+ Potential assignment: (\w+) to (\w+)": "講師 \\1 をコース \\2 へ割り当てる可能性があります。",
    r"^\s*Cost for (\w+) to (\w+):.*total_weighted_int=(\d+)": "講師 \\1 からコース \\2 への割り当てコスト (重み付け合計): \\3",
    r"^\s*- Filtered out: (\w+) for (\w+) \(Last classroom failed: L_last_class=(\w+), C_class=(\w+)\)": "講師 \\1 のコース \\2 への割り当ては除外されました (理由: 最終教室の重複 L_last_class=\\3, C_class=\\4)。",
    r"^\s*- Filtered out: (\w+) for (\w+) \(Schedule failed: Course_schedule=\(.*\), Lecturer_avail=\[.*\]\)": "講師 \\1 のコース \\2 への割り当ては除外されました (理由: スケジュール不一致)。",
    r"Total potential assignments after filtering: (\d+)": "フィルタリング後の潜在的な割り当て総数: \\1",
    r"Parameters: (.*)": "CP-SAT ソルバーパラメータ: \\1",
    r"Setting number of workers to (\d+)": "CP-SAT ワーカースレッド数を \\1 に設定。",
    r"Initial optimization model .*: \(model_fingerprint: ([\w\d]+)\)": "初期最適化モデル (フィンガープリント: \\1)",
    r"#Variables: (\d+) \(#bools: (\d+) in objective\)": "変数総数: \\1 (うち目的関数内のブール変数: \\2)",
    r"Starting presolve at ([\d\.]+)s": "Presolve (前処理) を \\1 秒で開始。",
    r"Starting CP-SAT solver v([\d\.]+)": "CP-SAT ソルバー (バージョン \\1) を開始します。",
    r"Presolved optimization model .*: \(model_fingerprint: ([\w\d]+)\)": "Presolve後の最適化モデル (フィンガープリント: \\1)",
    r"Preloading model.": "モデルをプリロード中。",
    r"#Bound\s+[\d\.]+s\s+best:([\w\.]+)\s+next:\[([^\]]+)\]\s+(\w+)": "境界値情報: 最良値=\\1, 次候補=\\2, 状態=\\3",
    r"The solution hint is complete and is feasible. Its objective value is ([\d\.]+).": "提供された解ヒントは完全かつ実行可能です。目的値: \\1",
    r"Starting search at ([\d\.]+)s with (\d+) workers.": "\\2 個のワーカーで \\1 秒に探索を開始。",
    r"#(\d+)\s+[\d\.]+s\s+best:([\w\.]+)\s+next:\[([^\]]*)\]\s+(\w+)": "探索ステップ #\\1: 最良値=\\2, 次候補=\\3, ソルバー=\\4",
    r"#Done\s+[\d\.]+s\s+(\w+)": "探索完了: ソルバー=\\1",
    r"CpSolverResponse summary:": "CP-SAT ソルバー応答サマリー:",
    r"status: (\w+)": "ソルバーの最終ステータス: \\1",
    r"objective: ([\d\.]+)": "ソルバーの目的値: \\1",
    r"best_bound: ([\d\.]+)": "最適解の下界 (Best Bound): \\1",
    r"Presolve summary:": "Presolve (前処理) の要約:",
    r"^\s*- rule '([^']*)' was applied (\d+) time": "Presolve ルール「\\1」が \\2 回適用されました。",
    r"walltime: ([\d\.]+)": "実行時間 (Wall Time): \\1 秒",
    r"usertime: ([\d\.]+)": "ユーザー時間 (User Time): \\1 秒",
    r"deterministic_time: ([\d\.e\+\-]+)": "決定論的時間 (Deterministic Time): \\1",
    r"--- Solver Log \(Captured by app.py\) ---": "--- ソルバーログ (キャプチャ区間開始) ---", # app.py でキャプチャすることを示すように変更
    r"--- End Solver Log \(Captured by app.py\) ---": "--- ソルバーログ (キャプチャ区間終了) ---", # app.py でキャプチャすることを示すように変更
}

def _get_log_explanation(log_line: str) -> str:
    """
    ログ行をチェックし、定義されたパターンに一致する場合、解説文字列を返します。
    一致しない場合は、デフォルトのメッセージを返します。
    (log_explainer.py の get_log_explanation 関数を移動)
    """
    stripped_log_line = log_line.strip()
    for pattern, explanation_template in LOG_EXPLANATIONS.items():
        match = re.search(pattern, stripped_log_line)
        if match:
            explanation = explanation_template
            for i, group_val in enumerate(match.groups()):
                explanation = explanation.replace(f"\\{i+1}", str(group_val))
            return explanation
    return "(このログメッセージに対する定義済みの解説はありません)"

# --- 大規模データ生成 ---
# (変更なし)

# --- Gemini API 連携 ---
def get_gemini_explanation(log_text: str, api_key: str) -> str:
    """
    指定されたログテキストを Gemini API に送信し、解説を取得します。
    """
    if not api_key:
        return "エラー: Gemini API キーが設定されていません。"
    if not log_text:
        return "解説対象のログがありません。"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro') 
        prompt = f"""以下のシステムログについて、IT専門家でない人にも分かりやすく解説してください。
ログの各部分が何を示しているのか、全体としてどのような処理が行われているのかを説明してください。
特に重要な情報、警告、エラーがあれば指摘し、考えられる原因や対処法についても言及してください。

ログ本文:
```text
{log_text}
```

解説:
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API での解説中にエラーが発生しました: {e}")
        return f"Gemini API での解説中にエラーが発生しました: {str(e)[:500]}..." # エラーメッセージを短縮して表示

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

from typing import TypedDict, List, Optional, Any, Tuple # 追加

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
    raw_solver_log: str
    explained_log_text: str # Detailed log with line-by-line explanation for UI
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
    explained_log_output = [] # 解説付きログを格納するリスト

    # アプリケーションログを full_log_stream に直接書き込むように変更
    def log_to_stream(message):
        print(message, file=full_log_stream)
        print(message) # ターミナルにも表示（デバッグ用）

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
            total_weighted_cost_int = int(total_weighted_cost_float * 100)
            log_to_stream(f"    Cost for {lecturer_id} to {course_id}: travel={travel_cost}, age={age_cost}, freq={frequency_cost}, qual={qualification_cost}, sched_viol_penalty={schedule_violation_penalty}, recency_cost_raw={past_assignment_recency_cost} (days_since_last_on_this_classroom={'N/A' if days_since_last_assignment_to_classroom == float('inf') else days_since_last_assignment_to_classroom}), total_weighted_int={total_weighted_cost_int}")
            # 上記ログの days_since_last_assignment_to_classroom の表示を修正
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
        if all_captured_logs:
            for line in all_captured_logs.splitlines():
                explanation = _get_log_explanation(line)
                explained_log_output.append(f"ログ: {line.strip()}")
                explained_log_output.append(f"解説: {explanation}")
                explained_log_output.append("-" * 20)
        explained_log_text = "\n".join(explained_log_output)
        return SolverOutput(
            solution_status_str="前提条件エラー (割り当て候補なし)",
            objective_value=None,
            assignments=[],
            all_courses=courses_data,
            all_lecturers=lecturers_data,
            solver_raw_status_code=cp_model.UNKNOWN, 
            raw_solver_log=all_captured_logs,
            explained_log_text=explained_log_text,
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
        if all_captured_logs:
            for line in all_captured_logs.splitlines():
                explanation = _get_log_explanation(line)
                explained_log_output.append(f"ログ: {line.strip()}")
                explained_log_output.append(f"解説: {explanation}")
                explained_log_output.append("-" * 20)
        explained_log_text = "\n".join(explained_log_output)
        return SolverOutput(
            solution_status_str="目的関数エラー (最適化対象なし)",
            objective_value=None,
            assignments=[],
            all_courses=courses_data,
            all_lecturers=lecturers_data,
            solver_raw_status_code=cp_model.MODEL_INVALID,
            raw_solver_log=all_captured_logs,
            explained_log_text=explained_log_text,
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

    # --- Gemini API送信用ログのフィルタリング ---
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
            
            if solver_log_end_marker in line: # solver_log_end_marker が先に来ることはないはずだが念のため
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

        # 詳細アプリケーションログの最大表示行数を調整
        MAX_APP_DETAIL_FIRST_N_LINES = 3 # 先頭から表示する最大行数をさらに削減
        MAX_APP_DETAIL_LAST_N_LINES = 3  # 末尾から表示する最大行数をさらに削減
        if len(app_detailed_lines_collected) > (MAX_APP_DETAIL_FIRST_N_LINES + MAX_APP_DETAIL_LAST_N_LINES):
            gemini_log_lines_final.extend(app_detailed_lines_collected[:MAX_APP_DETAIL_FIRST_N_LINES])
            omitted_count = len(app_detailed_lines_collected) - (MAX_APP_DETAIL_FIRST_N_LINES + MAX_APP_DETAIL_LAST_N_LINES)
            gemini_log_lines_final.append(f"\n[... {omitted_count} 件の詳細なアプリケーションログ（個々の割り当てチェック等）は簡潔さのため省略されました ...]\n")
            gemini_log_lines_final.extend(app_detailed_lines_collected[-MAX_APP_DETAIL_LAST_N_LINES:])
        else:
            gemini_log_lines_final.extend(app_detailed_lines_collected)

        # ソルバーログの最大表示行数を調整
        MAX_SOLVER_LOG_FIRST_N_LINES = 30 # ソルバーログの先頭から表示する最大行数をさらに削減
        MAX_SOLVER_LOG_LAST_N_LINES = 30  # ソルバーログの末尾から表示する最大行数をさらに削減
        if len(solver_log_block) > (MAX_SOLVER_LOG_FIRST_N_LINES + MAX_SOLVER_LOG_LAST_N_LINES):
            truncated_solver_log = solver_log_block[:MAX_SOLVER_LOG_FIRST_N_LINES]
            omitted_solver_lines = len(solver_log_block) - (MAX_SOLVER_LOG_FIRST_N_LINES + MAX_SOLVER_LOG_LAST_N_LINES)
            truncated_solver_log.append(f"\n[... {omitted_solver_lines} 件のソルバーログ中間行は簡潔さのため省略されました ...]\n")
            truncated_solver_log.extend(solver_log_block[-MAX_SOLVER_LOG_LAST_N_LINES:])
            gemini_log_lines_final.extend(truncated_solver_log)
        else:
            gemini_log_lines_final.extend(solver_log_block)
        
        return "\n".join(gemini_log_lines_final)

    gemini_api_log = filter_log_for_gemini(full_captured_logs)

    # --- UI表示用の解説付きログ生成 (全ログを使用) ---
    # explained_log_output は既に solve_assignment の冒頭で初期化されている
    if full_captured_logs:
        for line in full_captured_logs.splitlines():
            explanation = _get_log_explanation(line) # このファイル内の関数を使用
            explained_log_output.append(f"ログ: {line.strip()}")
            explained_log_output.append(f"解説: {explanation}")
            explained_log_output.append("-" * 20) # 区切り線
    
    explained_log_text = "\n".join(explained_log_output)

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
        raw_solver_log=gemini_api_log, # Gemini API に送る削減版ログ
        explained_log_text=explained_log_text, # UI表示用の解説付き全ログ
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
            # トークンからユーザー情報を取得する (streamlit-oauth は直接ユーザー情報を返さない場合がある)
            # ここでは簡略化のため、email をユーザー情報として扱う例を示します。
            # 実際には、トークンを使ってGoogleのユーザー情報エンドポイントに問い合わせる必要があります。
            # もし oauth2.get_user_info() のようなメソッドがあればそれを使います。
            # ここでは仮にトークン自体にemailが含まれていると仮定します（実際はIDトークンをデコード）。
            # 簡単な例として、ログイン成功時に固定のユーザー情報をセットします。
            # 実際のアプリケーションでは、IDトークンを検証し、そこからemailやnameを取得すべきです。
            # (例: from google.oauth2 import id_token; from google.auth.transport import requests;
            #      id_info = id_token.verify_oauth2_token(st.session_state.token['id_token'], requests.Request(), GOOGLE_CLIENT_ID)
            #      st.session_state.user_info = {"email": id_info.get("email"), "name": id_info.get("name")} )
            try:
                # ここでは簡略化のため、固定のユーザー情報を設定します。
                # 実際にはIDトークンからユーザー情報を抽出・検証してください。
                st.session_state.user_info = {"email": "user@example.com", "name": "Test User"}
            except Exception as e:
                st.error(f"ユーザー情報の取得/設定中にエラー: {e}")
                st.session_state.user_info = {"email": "error@example.com"}
            st.rerun()
        return # 未認証の場合はここで処理を終了し、メインUIは表示しない

    # --- 認証済みの場合: メインアプリケーションUI表示 ---
    # このブロックは st.session_state.token が存在する場合のみ実行されます

    # st.sidebar.header("最適化設定") # より詳細な構成に変更

    st.sidebar.markdown(
        "以下の制約の中で、目的に対する最適化を実行します。"
        "目的に重み付けすることでシミュレーションしながら最適解を探索することができます。"
    )
    st.sidebar.markdown("---")

    st.sidebar.subheader("【制約】")
    st.sidebar.markdown("**ハード制約（絶対固定）**")
    st.sidebar.markdown("- 講師は資格ランクに応じた講座しか割り当てできません。")
    st.sidebar.markdown("- 講師は（今回の割り当てでは）1つの講座しか担当できません。")

    st.sidebar.markdown("**ソフト制約（割り当てできない場合に許容できる）**")
    st.sidebar.markdown(
        "講師の空きスケジュールに応じた講座しか割り当てできないことが原則ですが、"
        "完全割り当てのために以下のことを許容できます。"
    )
    ignore_schedule_constraint_checkbox = st.sidebar.checkbox(
        "講師の空きスケジュールを無視する", 
        value=True, 
        help="チェックを外すと割り当て結果が得られないことがあります。その場合は、講師の空きスケジュールが合わない場合が想定されます。"
    )
    st.sidebar.markdown("---")

    st.sidebar.subheader("【最適化目的と重み付け】")
    st.sidebar.caption(
        "【重要度について】不要な場合は、重要度をゼロしてください。（最適化要素に加味されなくなります。）"
        "また、各々の重要度は相対的なものなので、全部0.1と全部1.00は同じ結果となります。"
    )
    st.sidebar.markdown("**移動コストが低い人を優先**")
    weight_travel = st.sidebar.slider("重要度", 0.0, 1.0, 0.5, 0.05, help="高いほど移動コストが低い人を重視します。", key="weight_travel")
    st.sidebar.markdown("**年齢の若い人を優先**")
    weight_age = st.sidebar.slider("重要度", 0.0, 1.0, 0.3, 0.05, help="高いほど年齢が若い人を重視します。", key="weight_age")
    st.sidebar.markdown("**割り当て頻度の少ない人を優先**")
    weight_frequency = st.sidebar.slider("重要度", 0.0, 1.0, 0.2, 0.05, help="高いほど全講座割当回数が少ない人を重視します。", key="weight_frequency")
    st.sidebar.markdown("**講師資格が高い人を優先**")
    weight_qualification_slider = st.sidebar.slider("重要度", 0.0, 1.0, 0.25, 0.05, help="高いほど講師資格ランクが高い人が重視されます。", key="weight_qualification")
    st.sidebar.markdown("**同教室への割り当てが過去に無い人を優先**")
    weight_past_assignment_recency_slider = st.sidebar.slider("重要度", 0.0, 1.0, 0.4, 0.05, help="高いほど同教室への割り当て実績人が無い人或いは最後に割り当てられた日からの経過日数が長い人が重視されます。", key="weight_past_assignment")

    # ログインユーザー情報とログアウトボタン
    user_email = st.session_state.user_info.get('email', '不明なユーザー') if st.session_state.user_info else '不明なユーザー'
    st.sidebar.markdown("---")
    st.sidebar.write(f"ログイン中: {user_email}")
    if st.sidebar.button("ログアウト"):
        st.session_state.token = None
        st.session_state.user_info = None
        # 関連するセッションステートもクリア (最適化結果キャッシュも含む)
        keys_to_clear = [
            "raw_solver_log_for_gca", 
            "gemini_explanation", 
            "solution_executed", 
            "solver_result_cache" # 追加
        ]
        for key_to_clear in keys_to_clear:
            if key_to_clear in st.session_state:
                del st.session_state[key_to_clear]
        st.rerun()

    # 「最適割り当てを実行」ボタンをサイドバーに移動
    if st.sidebar.button("最適割り当てを実行", type="primary"):
        # 既存のセッション変数をクリア (特に計算結果キャッシュ)
        if "solver_result_cache" in st.session_state: del st.session_state.solver_result_cache
        if "raw_solver_log_for_gca" in st.session_state: del st.session_state.raw_solver_log_for_gca
        if "gemini_explanation" in st.session_state: del st.session_state.gemini_explanation
        st.session_state.solution_executed = True # 実行フラグを立てる
        st.rerun() # 再実行してメインエリアで処理と表示を行う

    # アプリケーションバージョンをサイドバーに表示
    st.sidebar.markdown("---")
    st.sidebar.info(f"アプリバージョン: {APP_VERSION}")
    st.title("講師割り当てシステム デモ (OR-Tools) - ログ解説付き")
    # --- メインコンテンツ (認証済みの場合のみ表示) ---
    st.header("入力データ")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("講師データ")
        # past_assignments を表示用に整形
        df_lecturers = pd.DataFrame(DEFAULT_LECTURERS_DATA)
        if 'past_assignments' in df_lecturers.columns:
            df_lecturers['past_assignments'] = df_lecturers['past_assignments'].apply(
                lambda assignments: ", ".join([f"{a['classroom_id']} ({a['date']})" for a in assignments]) if isinstance(assignments, list) and assignments else "履歴なし"
            )
        st.dataframe(df_lecturers, height=200)
    with col2:
        st.subheader("講座データ")
        st.dataframe(pd.DataFrame(DEFAULT_COURSES_DATA), height=200)
    
    st.subheader("教室データと移動コスト")
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

    # 「基本コスト設定」セクションを削除

    # 最適化実行フラグに基づいて結果を表示
    if st.session_state.get("solution_executed", False):
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
                st.session_state.solver_result_cache = solver_output # 結果をキャッシュ
                st.session_state.raw_solver_log_for_gca = solver_output["raw_solver_log"]

                log_for_gemini_api = solver_output["raw_solver_log"]
                if log_for_gemini_api and GEMINI_API_KEY:
                    with st.spinner("Gemini API でログを解説中..."):
                        gemini_explanation_text = get_gemini_explanation(log_for_gemini_api, GEMINI_API_KEY)
                        st.session_state.gemini_explanation = gemini_explanation_text
                elif not GEMINI_API_KEY:
                    st.session_state.gemini_explanation = "Gemini API キーが設定されていません。ログ解説はスキップされました。"
        
        # キャッシュされた結果（または計算直後の結果）を取得
        solver_result = st.session_state.solver_result_cache

        # --- 以降、solver_result を使った表示ロジック ---
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

        if "gemini_explanation" in st.session_state and st.session_state.gemini_explanation:
            with st.expander("Gemini API によるログ解説", expanded=True):
                st.markdown(st.session_state.gemini_explanation)
        if solver_result.get('explained_log_text'): # キーが存在するか確認
            with st.expander("処理ログ詳細 (解説付き)"):
                st.text_area("Explained Log Output", solver_result['explained_log_text'], height=400)
        if solver_result.get('raw_solver_log'): # キーが存在するか確認 (Gemini API送信ログ)
            with st.expander("生ログ詳細 (最適化処理の全出力 - Gemini APIへ送信されたログ)"):
                st.text_area("Raw Solver Log (Sent to Gemini API)", solver_result['raw_solver_log'], height=300)

if __name__ == "__main__":
    main()
