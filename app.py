import streamlit as st
from ortools.sat.python import cp_model
import pandas as pd
import io
import contextlib
import re # 正規表現モジュール
import google.generativeai as genai # Gemini API 用
from streamlit_oauth import OAuth2Component # OIDC認証用
import random # データ生成用

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
    r"^\s*- Filtered out: (\w+) for (\w+) \(Rank failed: L_rank=(\d+), C_req_rank=(\d+)\)": "講師 \\1 のコース \\2 への割り当ては除外されました (理由: ランク不一致 L_rank=\\3, C_req_rank=\\4)。",
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
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # または 'gemini-pro' など適切なモデル
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

# 既存の教室ID
EXISTING_CLASSROOM_IDS = ["C1", "C2", "C3"]
EXISTING_CLASSROOM_LOCATIONS = {"C1": "東京拠点", "C2": "大阪拠点", "C3": "名古屋拠点"}

# 都道府県教室ID (P1 - P47)
PREFECTURE_CLASSROOM_IDS = [f"P{i+1}" for i in range(len(PREFECTURES))]

# 全教室データ生成
DEFAULT_CLASSROOMS_DATA = []
for cid in EXISTING_CLASSROOM_IDS:
    DEFAULT_CLASSROOMS_DATA.append({"id": cid, "location": EXISTING_CLASSROOM_LOCATIONS[cid]})
for i, pref_name in enumerate(PREFECTURES):
    DEFAULT_CLASSROOMS_DATA.append({"id": PREFECTURE_CLASSROOM_IDS[i], "location": pref_name})

ALL_CLASSROOM_IDS_COMBINED = EXISTING_CLASSROOM_IDS + PREFECTURE_CLASSROOM_IDS

# 講師データ生成 (100人)
DEFAULT_LECTURERS_DATA = []
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
TIMES = ["AM", "PM"]
ALL_SLOTS = [(day, time) for day in DAYS for time in TIMES]
AGE_CATEGORIES = ["low", "middle", "high"]
QUALIFICATION_RANKS = [1, 2, 3]
FREQ_CATEGORIES = ["low", "middle", "high"]

for i in range(1, 101):
    num_available_slots = random.randint(3, 7)
    availability = random.sample(ALL_SLOTS, num_available_slots)
    DEFAULT_LECTURERS_DATA.append({
        "id": f"L{i}",
        "name": f"講師{i:03d}",
        "home_classroom_id": random.choice(PREFECTURE_CLASSROOM_IDS), # 本拠地は都道府県のいずれか
        "age_category": random.choice(AGE_CATEGORIES),
        "qualification_rank": random.choice(QUALIFICATION_RANKS),
        "availability": availability,
        "assignment_frequency_category": random.choice(FREQ_CATEGORIES),
        "last_assigned_classroom_id": random.choice(ALL_CLASSROOM_IDS_COMBINED + [None]) # 前回割り当ては全教室またはNone
    })

# 講座データ生成 (47都道府県 × 7種類の講座)
BASE_COURSE_DEFINITIONS = [
    {"id_suffix": "S1", "name": "初級プログラミング", "required_rank": 1, "schedule": ("Mon", "AM")},
    {"id_suffix": "S2", "name": "中級データ分析", "required_rank": 2, "schedule": ("Tue", "PM")},
    {"id_suffix": "S3", "name": "上級機械学習", "required_rank": 3, "schedule": ("Mon", "PM")},
    {"id_suffix": "S4", "name": "初級ウェブデザイン", "required_rank": 1, "schedule": ("Wed", "AM")},
    {"id_suffix": "S5", "name": "中級プログラミング", "required_rank": 2, "schedule": ("Thu", "AM")},
    {"id_suffix": "S6", "name": "Python入門", "required_rank": 1, "schedule": ("Fri", "PM")},
    {"id_suffix": "S7", "name": "データベース基礎", "required_rank": 2, "schedule": ("Tue", "AM")}
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

# 移動コスト行列生成 (全教室間)
DEFAULT_TRAVEL_COSTS_MATRIX = {}

# 既存のC1,C2,C3間のコスト
EXISTING_C_TRAVEL_COSTS = {
    ("C1", "C1"): 0, ("C1", "C2"): 20, ("C1", "C3"): 30,
    ("C2", "C1"): 20, ("C2", "C2"): 0, ("C2", "C3"): 25,
    ("C3", "C1"): 30, ("C3", "C2"): 25, ("C3", "C3"): 0,
}

for c_from in ALL_CLASSROOM_IDS_COMBINED:
    for c_to in ALL_CLASSROOM_IDS_COMBINED:
        if c_from == c_to:
            DEFAULT_TRAVEL_COSTS_MATRIX[(c_from, c_to)] = 0
        elif c_from in EXISTING_CLASSROOM_IDS and c_to in EXISTING_CLASSROOM_IDS:
            DEFAULT_TRAVEL_COSTS_MATRIX[(c_from, c_to)] = EXISTING_C_TRAVEL_COSTS.get((c_from, c_to), 50) # フォールバック
        else:
            # 都道府県間、または都道府県と既存拠点C1-C3間はランダムコスト (例: 10-150)
            # ここでは簡略化のため、異なる都道府県/拠点間は比較的高めのコストとする
            if c_from.startswith("P") and c_to.startswith("P"): # 都道府県間
                cost = random.randint(10, 80) if c_from != c_to else 0
            elif (c_from.startswith("P") and c_to.startswith("C")) or \
                 (c_from.startswith("C") and c_to.startswith("P")): # 都道府県とC拠点間
                cost = random.randint(30, 150)
            else: # 予期せぬケース (基本的には上記でカバーされるはず)
                cost = 100 
            DEFAULT_TRAVEL_COSTS_MATRIX[(c_from, c_to)] = cost

# 年齢カテゴリ別基本コスト (若いほど低コスト = 優先度高)
DEFAULT_AGE_PRIORITY_COSTS = {
    "low": 1,     # 若手
    "middle": 5,  # 中堅
    "high": 10    # ベテラン
}

# 割り当て頻度カテゴリ別基本コスト (頻度が低いほど低コスト = 優先度高)
DEFAULT_FREQUENCY_PRIORITY_COSTS = {
    "low": 1,
    "middle": 5,
    "high": 10
}

from typing import TypedDict, List, Optional, Any, Tuple # 追加

# --- 2. OR-Tools 最適化ロジック ---
class SolverOutput(TypedDict): # 提案: 戻り値を構造化するための型定義
    solution_status_str: str
    objective_value: Optional[float]
    assignments: List[dict]
    all_courses: List[dict]
    all_lecturers: List[dict]
    solver_raw_status_code: int
    raw_solver_log: str
    explained_log_text: str
    filtered_log_for_gemini: str # Gemini API送信用にフィルタリングされたログ

def solve_assignment(lecturers_data, courses_data, classrooms_data,
                     travel_costs_matrix, age_priority_costs, frequency_priority_costs,
                     weight_travel, weight_age, weight_frequency, unassigned_course_penalty_value,
                     option_avoid_last_classroom) -> SolverOutput: # 戻り値の型ヒントを変更
    model = cp_model.CpModel()
    # solve_assignment 内の print 文も解説対象に含めるために、
    # ここで stdout のキャプチャを開始する
    full_log_stream = io.StringIO()
    explained_log_output = [] # 解説付きログを格納するリスト

    with contextlib.redirect_stdout(full_log_stream): # この関数全体の標準出力をキャプチャ
        print(f"Initial lecturers: {len(lecturers_data)}, Initial courses: {len(courses_data)}")
        potential_assignment_count = 0
        possible_assignments = []

        for lecturer in lecturers_data:
            for course in courses_data:
                lecturer_id = lecturer["id"]
                course_id = course["id"]

                if lecturer["qualification_rank"] < course["required_rank"]:
                    print(f"  - Filtered out: {lecturer_id} for {course_id} (Rank failed: L_rank={lecturer['qualification_rank']}, C_req_rank={course['required_rank']})")
                    continue
                if course["schedule"] not in lecturer["availability"]:
                    print(f"  - Filtered out: {lecturer_id} for {course_id} (Schedule failed: Course_schedule={course['schedule']} (type: {type(course['schedule'])}), Lecturer_avail={lecturer['availability']} (type: {type(lecturer['availability'])}, element_type: {type(lecturer['availability'][0]) if lecturer['availability'] else 'N/A'}))")
                    continue
                if option_avoid_last_classroom and \
                   lecturer["last_assigned_classroom_id"] == course["classroom_id"]:
                    print(f"  - Filtered out: {lecturer_id} for {course_id} (Last classroom failed: L_last_class={lecturer['last_assigned_classroom_id']}, C_class={course['classroom_id']})")
                    continue
                
                potential_assignment_count += 1
                print(f"  + Potential assignment: {lecturer_id} to {course_id}")

                var = model.NewBoolVar(f'x_{lecturer_id}_{course_id}')
                
                travel_cost = travel_costs_matrix.get((lecturer["home_classroom_id"], course["classroom_id"]), 999)
                age_cost = age_priority_costs.get(lecturer["age_category"], 999)
                frequency_cost = frequency_priority_costs.get(lecturer["assignment_frequency_category"], 999)
                total_weighted_cost_float = (weight_travel * travel_cost +
                                             weight_age * age_cost +
                                             weight_frequency * frequency_cost)
                total_weighted_cost_int = int(total_weighted_cost_float * 100)
                print(f"    Cost for {lecturer_id} to {course_id}: travel={travel_cost}, age={age_cost}, freq={frequency_cost}, total_weighted_int={total_weighted_cost_int}")

                possible_assignments.append({
                    "lecturer_id": lecturer_id, "course_id": course_id,
                    "variable": var, "cost": total_weighted_cost_int
                })

        print(f"Total potential assignments after filtering: {potential_assignment_count}")
        print(f"Length of possible_assignments list (with variables): {len(possible_assignments)}")

        if not possible_assignments:
            print("No possible assignments found after filtering. Optimization will likely result in no assignments.")
            # 早期リターンのための準備 (現状のコードでは早期リターンはないが、もし追加する場合)
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
                solver_raw_status_code=cp_model.UNKNOWN, # 適切なステータスコード
                raw_solver_log=all_captured_logs,
                explained_log_text=explained_log_text
            )

        for course_item in courses_data:
            course_id = course_item["id"]
            model.Add(sum(pa["variable"] for pa in possible_assignments if pa["course_id"] == course_id) <= 1)

        courses_dict = {c["id"]: c for c in courses_data}
        for lecturer_item in lecturers_data:
            lecturer_id = lecturer_item["id"]
            lecturer_assigned_schedules = {}
            for pa in possible_assignments:
                if pa["lecturer_id"] == lecturer_id:
                    c_schedule = courses_dict[pa["course_id"]]["schedule"]
                    if c_schedule not in lecturer_assigned_schedules:
                        lecturer_assigned_schedules[c_schedule] = []
                    lecturer_assigned_schedules[c_schedule].append(pa["variable"])
            for schedule_vars in lecturer_assigned_schedules.values():
                if len(schedule_vars) > 1:
                    model.Add(sum(schedule_vars) <= 1)

        assignment_costs = [pa["variable"] * pa["cost"] for pa in possible_assignments]
        penalty_terms = []
        for course_data_item in courses_data:
            course_id = course_data_item["id"]
            possible_assignments_for_course = [pa["variable"] for pa in possible_assignments if pa["course_id"] == course_id]
            if possible_assignments_for_course:
                penalty_terms.append((1 - sum(possible_assignments_for_course)) * unassigned_course_penalty_value)
            else:
                penalty_terms.append(unassigned_course_penalty_value)
        
        objective_terms = assignment_costs + penalty_terms
        if objective_terms:
            model.Minimize(sum(objective_terms))
        else:
            print("Objective terms list is empty. No assignments to optimize.")
            # 目的項がない場合も早期リターンと同様の処理
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
                solver_raw_status_code=cp_model.MODEL_INVALID, # 適切なステータスコード
                raw_solver_log=all_captured_logs,
                explained_log_text=explained_log_text
            )

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True

        # ログコールバック関数を定義
        # この関数はソルバーからのメッセージを full_log_stream に書き込みます。
        def cp_sat_log_callback(message):
            full_log_stream.write(message) # メッセージをそのままストリームに書き込む

        # ソルバーにログコールバックを設定
        solver.SetLogCallback(cp_sat_log_callback)

        print("--- Solver Log (Captured by app.py via LogCallback) ---") # CP-SATのログ開始を示すマーカー (コールバック経由であることを明示)
        status_code = solver.Solve(model) # status_code を保持
        print("--- End Solver Log (Captured by app.py via LogCallback) ---") # CP-SATのログ終了を示すマーカー

    # キャプチャ終了
    all_captured_logs = full_log_stream.getvalue()
    
    # Geminiに渡すログをフィルタリングする例 (より関心のある情報に絞る)
    # ソルバーログのセクションのみを対象とする
    filtered_log_for_gemini_lines = []
    in_solver_log_section = False

    for line in all_captured_logs.splitlines():
        stripped_line = line.strip()

        if stripped_line.startswith("--- Solver Log ("):
            in_solver_log_section = True
            filtered_log_for_gemini_lines.append(line) # 開始マーカーは含める
            continue

        if stripped_line.startswith("--- End Solver Log ("):
            in_solver_log_section = False
            filtered_log_for_gemini_lines.append(line) # 終了マーカーは含める
            continue
        
        if in_solver_log_section:
            # ソルバーログセクション内のログのみをフィルタリング対象とする
            # ソルバーが何を選んだか (最終ステータス、目的値、応答サマリー)
            if stripped_line.startswith("CpSolverResponse summary:") or \
               stripped_line.startswith("status:") or \
               stripped_line.startswith("objective:") or \
               # ソルバー内部で使われたツールや選択
               stripped_line.startswith("Presolve summary:") or \ # Presolve処理の概要のみ
               re.search(r"Parameters:.*(linear_programming_relaxation|use_lp|log_search_progress|num_search_workers|max_time_in_seconds)", stripped_line, re.IGNORECASE) or \
               "LP statistics" in stripped_line or \
               re.search(r"Using relaxation:.*linear_programming", stripped_line, re.IGNORECASE) or \
               re.search(r"Starting presolve", stripped_line, re.IGNORECASE) or \
               re.search(r"Starting search", stripped_line, re.IGNORECASE) or \
               # 探索ステップのログは、"Done" または "Optimal" が含まれる最終的なものに近いログ、
               # または "objective" が更新されたことを示すログに絞り込むことを試みる
               (re.search(r"#\d+\s+[\d\.]+s\s+best:[\w\.]+", stripped_line) and ("Done" in stripped_line or "Optimal" in stripped_line or "objective" in stripped_line.lower())) or \
               # 主要な探索戦略の開始を示すログ (例: LNS worker)
               re.search(r"Worker \d+ starting.*(LNS|Core|FeasibilityPump|Probing)", stripped_line, re.IGNORECASE):
                filtered_log_for_gemini_lines.append(line)
        # else:
            # ソルバーログセクション外のアプリケーションログは、Geminiへの入力からは除外
            # (UI表示用の explained_log_text には含まれる)
            pass

    filtered_log_str_for_gemini = "\n".join(filtered_log_for_gemini_lines)
    # あまりに短い場合は元のログを使うなどの調整 (ただし、トークン数上限に注意)
    if not filtered_log_str_for_gemini and all_captured_logs:
        # filtered_log_str_for_gemini = all_captured_logs[:1000000] # トークン上限に近い文字数で切り詰める例
        pass # 一旦、フィルタリング結果が空なら空のまま渡す（UI側で全ログを使うため）

    if all_captured_logs:
        for line in all_captured_logs.splitlines():
            explanation = _get_log_explanation(line) # このファイル内の関数を使用
            explained_log_output.append(f"ログ: {line.strip()}") # .strip() は元のコードにもあったので維持
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
                    # デバッグ用に個別のコストも追加してみる (重み付け前)
                    "移動コスト(元)": travel_costs_matrix.get((lecturer["home_classroom_id"], course["classroom_id"]), 999),
                    "年齢コスト(元)": age_priority_costs.get(lecturer["age_category"], 999),
                    "頻度コスト(元)": frequency_priority_costs.get(lecturer["assignment_frequency_category"], 999)

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
        # Geminiにはフィルタリングしたログを、生ログとしては全体を渡すように変更も可能
        raw_solver_log=all_captured_logs, # UI表示用には全ログ
        explained_log_text=explained_log_text,
        filtered_log_for_gemini=filtered_log_str_for_gemini
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

    if "token" not in st.session_state:
        result = oauth2.authorize_button(
            name="Google でログイン",
            icon="https://www.google.com/favicon.ico",
            redirect_uri=REDIRECT_URI,
            scope="openid email profile",
            key="google_login",
            use_container_width=True,
        )
        if result:
            st.session_state.token = result.get("token")
            st.rerun()
        return # 認証が完了するまでメインコンテンツは表示しない
    else:
        # token = st.session_state.token # 必要であればトークン情報を使用
        if st.sidebar.button("ログアウト"):
            del st.session_state.token
            st.rerun()

    st.title("講師割り当てシステム デモ (OR-Tools) - ログ解説付き")

    # --- サイドバー: 設定 ---
    st.sidebar.header("最適化設定")

    st.sidebar.subheader("目的関数の重み")
    weight_travel = st.sidebar.slider("移動コストの重要度", 0.0, 1.0, 0.5, 0.05, help="高いほど移動コストを重視します。")
    weight_age = st.sidebar.slider("年齢の若さの重要度 (若い人を優先)", 0.0, 1.0, 0.3, 0.05, help="高いほど若い講師の割り当てを優先します。")
    weight_frequency = st.sidebar.slider("割り当て頻度の低さの重要度 (頻度少を優先)", 0.0, 1.0, 0.2, 0.05, help="高いほど割り当て頻度の低い講師を優先します。")

    st.sidebar.subheader("オプション制約")
    option_avoid_last_classroom = st.sidebar.checkbox("前回割り当てた教室には割り当てない", value=True)

    st.sidebar.subheader("未割り当て講座ペナルティ")
    unassigned_penalty_slider = st.sidebar.slider("未割り当て講座1件あたりのペナルティの大きさ", 0, 200000, 100000, 1000, help="値を大きくするほど、全ての講座を割り当てることを強く優先します。0にするとペナルティなし。")

    # --- メインコンテンツ ---
    st.header("入力データ")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("講師データ")
        st.dataframe(pd.DataFrame(DEFAULT_LECTURERS_DATA), height=200)
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

    st.subheader("基本コスト設定")
    col5, col6 = st.columns(2)
    with col5:
        st.write("年齢優先基本コスト (若いほど低コスト)")
        st.json(DEFAULT_AGE_PRIORITY_COSTS)
    with col6:
        st.write("頻度優先基本コスト (頻度低いほど低コスト)")
        st.json(DEFAULT_FREQUENCY_PRIORITY_COSTS)


    if st.button("最適割り当てを実行", type="primary"):
        if "raw_solver_log_for_gca" in st.session_state: # 関連するセッション変数もクリア
            del st.session_state.raw_solver_log_for_gca
        if "gemini_explanation" in st.session_state:
            del st.session_state.gemini_explanation
        if "solution_executed" in st.session_state: # 関連するセッション変数もクリア
            del st.session_state.solution_executed

        st.header("最適化結果")
        with st.spinner("最適化計算を実行中..."):
            # 既存の solve_assignment 関数呼び出し
            solver_result = solve_assignment( # 変更: 構造化された結果を受け取る
                DEFAULT_LECTURERS_DATA, DEFAULT_COURSES_DATA, DEFAULT_CLASSROOMS_DATA,
                DEFAULT_TRAVEL_COSTS_MATRIX, DEFAULT_AGE_PRIORITY_COSTS, DEFAULT_FREQUENCY_PRIORITY_COSTS,
                weight_travel, weight_age, weight_frequency, unassigned_penalty_slider,
                option_avoid_last_classroom
            )
            # 結果をセッション状態に保存してGCA解説ボタンで使えるようにする
            st.session_state.raw_solver_log_for_gca = solver_result["raw_solver_log"]
            st.session_state.solution_executed = True # 実行済みフラグ

            # Geminiに送信するログを準備 (フィルタリングされたもの、または全体)
            log_for_gemini_api = solver_result["filtered_log_for_gemini"] if solver_result["filtered_log_for_gemini"] else solver_result["raw_solver_log"]

            # Gemini API で解説を取得
            if log_for_gemini_api and GEMINI_API_KEY:
                with st.spinner("Gemini API でログを解説中..."):
                    gemini_explanation_text = get_gemini_explanation(log_for_gemini_api, GEMINI_API_KEY)
                    st.session_state.gemini_explanation = gemini_explanation_text
            elif not GEMINI_API_KEY:
                st.session_state.gemini_explanation = "Gemini API キーが設定されていません。ログ解説はスキップされました。"

        st.subheader(f"求解ステータス: {solver_result['solution_status_str']}") # 変更
        if solver_result['objective_value'] is not None: # 変更
            st.metric("総コスト (目的値)", f"{solver_result['objective_value']:.2f}") # 変更

        # 変更: solver_result から各値を取得するように修正
        if solver_result['solver_raw_status_code'] == cp_model.OPTIMAL or solver_result['solver_raw_status_code'] == cp_model.FEASIBLE:
            # results_df_data が実際に割り当て結果を含んでいるかを確認
            actual_assignments_made = bool(solver_result['assignments']) 

            if actual_assignments_made:
                st.subheader("割り当て結果")
                results_df = pd.DataFrame(solver_result['assignments'])
                st.dataframe(results_df)

                assigned_course_ids = {res["講座ID"] for res in solver_result['assignments']}
                unassigned_courses = [c for c in solver_result['all_courses'] if c["id"] not in assigned_course_ids]

                if unassigned_courses:
                    st.subheader("割り当てられなかった講座")
                    st.dataframe(pd.DataFrame(unassigned_courses))
                else:
                    st.success("全ての講座が割り当てられました。") 
                # 講師ごとの割り当て状況（オプション）
                st.subheader("講師ごとの割り当て状況")
                lecturer_assignments = {}
                for l_data in solver_result['all_lecturers']: # Iterate over actual lecturer data
                    lecturer_assignments[l_data["name"]] = []
                for res in solver_result['assignments']:
                    lecturer_name = res.get("講師名", "不明な講師") # 講師名がない場合のフォールバック
                    if lecturer_name not in lecturer_assignments: # まれに講師データにない名前が結果に含まれる場合への対処
                        lecturer_assignments[lecturer_name] = []
                    lecturer_assignments[lecturer_name].append(f"{res['講座名']} ({res['教室ID']}, {res['スケジュール']})")

                for name, assigned_list in lecturer_assignments.items():
                    if assigned_list:
                        st.markdown(f"**{name}**: " + ", ".join(assigned_list))
                    else:
                        st.markdown(f"**{name}**: 担当なし")
            else: # 最適解または実行可能解が見つかったが、実際の割り当ては行われなかった場合
                st.error("最適解または実行可能解と判定されましたが、実際の割り当ては行われませんでした。")
                st.warning(
                    "考えられる原因:\n"
                    "- 導入されたペナルティを考慮しても、全ての講座を割り当てない方が総コストが低いと判断された。\n"
                    "- または、割り当て可能なペアが元々存在しない (制約が厳しすぎる、データ不適合)。\n"
                    "**結果として、総コスト 0.00 (何も割り当てない) が最適と判断された可能性があります。**\n"
                    "コンソールのデバッグ出力を確認してください。\n"
                    f"  - `Total potential assignments after filtering`: 制約フィルタリング後の割り当て候補数\n"
                    f"  - `Length of possible_assignments list (with variables)`: OR-Toolsが実際に最適化を試みた候補数\n"
                    f"  - 各候補のコスト計算結果、およびフィルタリングで除外された理由"
                )
                st.subheader("全ての講座が割り当てられませんでした")
                st.dataframe(pd.DataFrame(solver_result['all_courses']))
        elif solver_result['solver_raw_status_code'] == cp_model.INFEASIBLE: # "実行不可能" の場合
            st.warning("指定された条件では、実行可能な割り当てが見つかりませんでした。制約やデータを見直してください。")
        else:
            st.error(solver_result['solution_status_str'])

        # Gemini API による解説結果の表示
        if "gemini_explanation" in st.session_state and st.session_state.gemini_explanation:
            with st.expander("Gemini API によるログ解説", expanded=True):
                st.markdown(st.session_state.gemini_explanation)

        # 解説付きログの表示
        if solver_result['explained_log_text']: # explained_log があれば表示
            with st.expander("処理ログ詳細 (解説付き)"):
                st.text_area("Explained Log Output", solver_result['explained_log_text'], height=400)

        if solver_result['raw_solver_log']: # raw_solver_log があれば表示
            with st.expander("生ログ詳細 (最適化処理の全出力)"):
                st.text_area("Raw Solver Log Output", solver_result['raw_solver_log'], height=300)

if __name__ == "__main__":
    main()
