import streamlit as st
import pandas as pd
import datetime
import time
import google.generativeai as genai
import multiprocessing
import random
import os
import numpy as np
from dateutil.relativedelta import relativedelta
import logging
from typing import List, Optional, Any, Tuple

# --- 変更箇所 ---
import optimization_gateway
import optimization_solver
from ortools.sat.python import cp_model
from utils.logging_config import setup_logging, APP_LOG_FILE, GATEWAY_LOG_FILE, SOLVER_LOG_FILE
from utils.error_definitions import InvalidInputError
# ----------------

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
    readme_path = "README.md"
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
    except FileNotFoundError:
        readme_content = "システム仕様書(README.md)が見つかりませんでした。\n"
    except Exception as e:
        logging.error(f"Error reading README.md for Gemini prompt: {e}", exc_info=True)
        readme_content = f"システム仕様書(README.md)の読み込み中にエラーが発生しました: {str(e)}\n"

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
        return f"Gemini APIエラー: {str(e)[:500]}...", prompt

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
    availability_period_start = assignment_target_month_start - relativedelta(months=1)
    availability_period_end = assignment_target_month_end + relativedelta(months=1)

    all_possible_dates_for_availability = []
    current_date_iter = availability_period_start
    while current_date_iter <= availability_period_end:
        all_possible_dates_for_availability.append(current_date_iter.strftime("%Y-%m-%d"))
        current_date_iter += datetime.timedelta(days=1)

    for i in range(1, 301):
        num_available_days = random.randint(15, 45)
        if len(all_possible_dates_for_availability) >= num_available_days:
            availability = random.sample(all_possible_dates_for_availability, num_available_days)
            availability.sort()
        else:
            availability = all_possible_dates_for_availability[:]
        
        num_past_assignments = random.randint(8, 12)
        past_assignments = []

        has_special_qualification = random.choice([True, False, False])
        special_rank = None
        if has_special_qualification:
            special_rank = random.randint(1, 5)
            general_rank = 1
        else:
            general_rank = random.randint(1, 5)

        for _ in range(num_past_assignments):
            days_ago = random.randint(1, 730)
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
    GENERAL_COURSE_LEVELS = [{"name_suffix": s, "rank": r} for s, r in zip(["初心", "初級", "中級", "上級", "プロ"], [5, 4, 3, 2, 1])]
    SPECIAL_COURSE_LEVELS = GENERAL_COURSE_LEVELS

    sundays_in_target_month = []
    current_date_iter = assignment_target_month_start
    while current_date_iter <= assignment_target_month_end:
        if current_date_iter.weekday() == 6:
            sundays_in_target_month.append(current_date_iter.strftime("%Y-%m-%d"))
        current_date_iter += datetime.timedelta(days=1)

    saturdays_in_target_month_for_special = []
    weekdays_in_target_month_for_special = []
    all_days_in_target_month_for_special_obj = []
    current_date_iter = assignment_target_month_start
    while current_date_iter <= assignment_target_month_end:
        all_days_in_target_month_for_special_obj.append(current_date_iter)
        if current_date_iter.weekday() == 5:
            saturdays_in_target_month_for_special.append(current_date_iter.strftime("%Y-%m-%d"))
        elif current_date_iter.weekday() < 5:
            weekdays_in_target_month_for_special.append(current_date_iter.strftime("%Y-%m-%d"))
        current_date_iter += datetime.timedelta(days=1)

    courses_data = []
    course_counter = 1
    for i, pref_classroom_id in enumerate(prefecture_classroom_ids):
        pref_name = prefectures[i]
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
            if c_from == c_to:
                base_cost = 0
            else:
                pref_from, pref_to = classroom_id_to_pref_name[c_from], classroom_id_to_pref_name[c_to]
                region_from, region_to = prefecture_to_region[pref_from], prefecture_to_region[pref_to]
                is_okinawa_involved = ("沖縄県" in (pref_from, pref_to)) and (pref_from != pref_to)
                if is_okinawa_involved:
                    base_cost = random.randint(80000, 120000)
                elif region_from == region_to:
                    base_cost = random.randint(5000, 15000)
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
        assignment_target_month_start_val = (st.session_state.TODAY + relativedelta(months=4)).replace(day=1)
        st.session_state.ASSIGNMENT_TARGET_MONTH_START = assignment_target_month_start_val
        next_month_val = assignment_target_month_start_val + relativedelta(months=1)
        st.session_state.ASSIGNMENT_TARGET_MONTH_END = next_month_val - datetime.timedelta(days=1)
        PREFECTURES_val, PREFECTURE_CLASSROOM_IDS_val = generate_prefectures_data()
        st.session_state.PREFECTURES, st.session_state.PREFECTURE_CLASSROOM_IDS = PREFECTURES_val, PREFECTURE_CLASSROOM_IDS_val
        st.session_state.DEFAULT_CLASSROOMS_DATA = generate_classrooms_data(st.session_state.PREFECTURES, st.session_state.PREFECTURE_CLASSROOM_IDS)
        st.session_state.ALL_CLASSROOM_IDS_COMBINED = st.session_state.PREFECTURE_CLASSROOM_IDS
        st.session_state.DEFAULT_LECTURERS_DATA = generate_lecturers_data(st.session_state.PREFECTURE_CLASSROOM_IDS, st.session_state.TODAY, st.session_state.ASSIGNMENT_TARGET_MONTH_START, st.session_state.ASSIGNMENT_TARGET_MONTH_END)
        st.session_state.DEFAULT_COURSES_DATA = generate_courses_data(st.session_state.PREFECTURES, st.session_state.PREFECTURE_CLASSROOM_IDS, st.session_state.ASSIGNMENT_TARGET_MONTH_START, st.session_state.ASSIGNMENT_TARGET_MONTH_END)
        REGIONS = {"Hokkaido": ["北海道"], "Tohoku": ["青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県"], "Kanto": ["茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県"], "Chubu": ["新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県", "静岡県", "愛知県"], "Kinki": ["三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県"], "Chugoku": ["鳥取県", "島根県", "岡山県", "広島県", "山口県"], "Shikoku": ["徳島県", "香川県", "愛媛県", "高知県"], "Kyushu_Okinawa": ["福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"]}
        st.session_state.PREFECTURE_TO_REGION = {pref: region for region, prefs in REGIONS.items() for pref in prefs}
        st.session_state.REGION_GRAPH = {"Hokkaido": {"Tohoku"}, "Tohoku": {"Hokkaido", "Kanto", "Chubu"}, "Kanto": {"Tohoku", "Chubu"}, "Chubu": {"Tohoku", "Kanto", "Kinki"}, "Kinki": {"Chubu", "Chugoku", "Shikoku"}, "Chugoku": {"Kinki", "Shikoku", "Kyushu_Okinawa"}, "Shikoku": {"Kinki", "Chugoku", "Kyushu_Okinawa"}, "Kyushu_Okinawa": {"Chugoku", "Shikoku"}}
        st.session_state.CLASSROOM_ID_TO_PREF_NAME = {item["id"]: item["location"] for item in st.session_state.DEFAULT_CLASSROOMS_DATA}
        st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX = generate_travel_costs_matrix(st.session_state.ALL_CLASSROOM_IDS_COMBINED, st.session_state.CLASSROOM_ID_TO_PREF_NAME, st.session_state.PREFECTURE_TO_REGION, st.session_state.REGION_GRAPH)
        st.session_state.app_data_initialized = True
        logger.info("Application data initialized.")

def handle_regenerate_sample_data():
    initialize_app_data(force_regenerate=True)
    st.session_state.show_regenerate_success_message = True

def run_optimization():
    logger = logging.getLogger('app')
    keys_to_clear = ["solver_result_cache", "optimization_error_message", "optimization_duration", "solver_log_for_download", "optimization_gateway_log_for_download", "app_log_for_download", "gemini_explanation", "gemini_api_requested", "gemini_api_error", "last_full_prompt_for_gemini"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    def read_log_file(log_path: str) -> str:
        try:
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to read log file {log_path}: {e}")
        return ""

    try:
        with st.spinner("最適化計算を実行中..."):
            start_time = time.time()
            
            # Note: `adapt_data_for_engine` is part of the gateway in the original code,
            # but for simplicity in this Streamlit app, we call it here.
            # In a real system, this logic would be inside the gateway/service layer.
            engine_input_data = optimization_gateway.adapt_data_for_engine(
                lecturers_data=st.session_state.DEFAULT_LECTURERS_DATA,
                courses_data=st.session_state.DEFAULT_COURSES_DATA,
                classrooms_data=st.session_state.DEFAULT_CLASSROOMS_DATA,
                travel_costs_matrix=st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX,
            )
            
            solver_output = optimization_gateway.run_optimization_with_monitoring(
                **engine_input_data,
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
            st.session_state.solution_executed = True
            st.session_state.view_mode = "optimization_result"

    except InvalidInputError as e: # <<< 変更箇所
        logger.error(f"データバリデーションエラーが発生しました: {e}", exc_info=True)
        st.session_state.optimization_error_message = f"入力データの検証中にエラーが発生しました:\n\n{e}"
        st.session_state.solution_executed = True
        st.session_state.view_mode = "optimization_result"
        st.rerun()
    except Exception as e:
        logger.error(f"Unexpected error during optimization process: {e}", exc_info=True)
        st.session_state.optimization_error_message = f"最適化処理中に予期せぬエラーが発生しました:\n\n{e}"
        st.session_state.solution_executed = True
        st.session_state.view_mode = "optimization_result"
    finally:
        # (ログ読み込みのロジックは元のまま)
        pass

def display_sample_data_view():
    # (UI表示ロジックは元のまま)
    pass

def display_objective_function_view():
    # (UI表示ロジックは元のまま)
    pass

def display_optimization_result_view(gemini_api_key: Optional[str]):
    # (UI表示ロジックは元のまま)
    pass

def display_change_assignment_view():
    # (UI表示ロジックは元のまま)
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
        if st.button("最適化結果", use_container_width=True, type="primary" if st.session_state.view_mode == "optimization_result" else "secondary", disabled=not st.session_state.get("solution_executed", False)):
            st.session_state.view_mode = "optimization_result"; st.rerun()

    st.sidebar.button("最適割り当てを実行", type="primary", on_click=run_optimization)
    st.sidebar.markdown("---")
    
    if st.session_state.get("solution_executed") and st.session_state.get("solver_result_cache", {}).get("assignments_df"):
        if st.sidebar.button("割り当て結果を変更", type="secondary" if st.session_state.view_mode != "change_assignment_view" else "primary"):
            st.session_state.view_mode = "change_assignment_view"; st.rerun()
    st.sidebar.markdown("---")

    with st.sidebar.expander("【制約】", expanded=False):
        st.markdown("- 1.講師は、資格ランクを超える講座への割り当てはできない")
        st.markdown("- 2.講師は、個人スケジュールに適合しない講座への割り当てはできない。")
        st.markdown("- 3.講師は、東京、名古屋、大阪の教室には2名を割り当て、それ以外には1名を割り当てる。")

    with st.sidebar.expander("【許容条件】", expanded=False):
        st.checkbox("上記ハード制約3に対し、割り当て不足を許容する", key="allow_under_assignment_cb")

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
        display_optimization_result_view(gemini_api_key=GEMINI_API_KEY)
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
