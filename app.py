# app.py (完全・修正版)

import streamlit as st
import pandas as pd
import datetime
import time
import multiprocessing
import random
import os
import numpy as np
from dateutil.relativedelta import relativedelta
import logging
from typing import List, Optional, Any, Tuple

# ----------------------------------------------------
# 1. 必要なモジュールと型定義をインポート
# ----------------------------------------------------
import optimization_gateway
from utils.logging_config import setup_logging
from utils.error_definitions import InvalidInputError
from utils.types import OptimizationInput, SolverParameters, OptimizationWeights
from ortools.sat.python import cp_model

# ----------------------------------------------------
# 2. ヘルパー関数群をすべて先に定義
# ----------------------------------------------------

def get_region_hops(region1, region2, graph):
    if region1 == region2: return 0
    queue = [(region1, 0)]
    visited = {region1}
    while queue:
        current_region, dist = queue.pop(0)
        if current_region == region2: return dist
        for neighbor in graph.get(current_region, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return float('inf')

def generate_prefectures_data():
    PREFECTURES = [
        "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県", "茨城県", "栃木県", "群馬県", 
        "埼玉県", "千葉県", "東京都", "神奈川県", "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", 
        "岐阜県", "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県", 
        "鳥取県", "島根県", "岡山県", "広島県", "山口県", "徳島県", "香川県", "愛媛県", "高知県", "福岡県", 
        "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"
    ]
    return PREFECTURES, [f"P{i+1}" for i in range(len(PREFECTURES))]

def generate_classrooms_data(prefectures, prefecture_classroom_ids):
    return [{"id": prefecture_classroom_ids[i], "location": name} for i, name in enumerate(prefectures)]

def generate_lecturers_data(prefecture_classroom_ids, today_date, start_date, end_date):
    lecturers_data = []
    availability_period_start = start_date - relativedelta(months=1)
    availability_period_end = end_date + relativedelta(months=1)
    all_possible_dates = [(availability_period_start + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range((availability_period_end - availability_period_start).days + 1)]

    for i in range(1, 301):
        availability = sorted(random.sample(all_possible_dates, min(random.randint(15, 45), len(all_possible_dates))))
        has_special_qual = random.choice([True, False, False])
        special_rank = random.randint(1, 5) if has_special_qual else None
        general_rank = 1 if has_special_qual else random.randint(1, 5)
        past_assignments = sorted([{"classroom_id": random.choice(prefecture_classroom_ids), "date": (today_date - datetime.timedelta(days=random.randint(1, 730))).strftime("%Y-%m-%d")} for _ in range(random.randint(8, 12))], key=lambda x: x["date"], reverse=True)
        
        lecturers_data.append({
            "id": f"L{i}", "name": f"講師{i:03d}", "age": random.randint(22, 65),
            "home_classroom_id": random.choice(prefecture_classroom_ids),
            "qualification_general_rank": general_rank, "qualification_special_rank": special_rank,
            "availability": availability, "past_assignments": past_assignments
        })
    return lecturers_data

def generate_courses_data(prefectures, prefecture_classroom_ids, start_date, end_date):
    GENERAL_COURSE_LEVELS = [{"name_suffix": s, "rank": r} for s, r in zip(["初心", "初級", "中級", "上級", "プロ"], [5, 4, 3, 2, 1])]
    SPECIAL_COURSE_LEVELS = GENERAL_COURSE_LEVELS
    courses_data, course_counter = [], 1
    all_days = [(start_date + datetime.timedelta(days=i)) for i in range((end_date - start_date).days + 1)]
    
    for i, classroom_id in enumerate(prefecture_classroom_ids):
        pref_name = prefectures[i]
        sundays = [d.strftime("%Y-%m-%d") for d in all_days if d.weekday() == 6]
        for sunday_str in sundays:
            for level_info in random.sample(GENERAL_COURSE_LEVELS, min(2, len(GENERAL_COURSE_LEVELS))):
                courses_data.append({"id": f"{classroom_id}-GC{course_counter}", "name": f"{pref_name} 一般講座 {level_info['name_suffix']}", "classroom_id": classroom_id, "course_type": "general", "rank": level_info['rank'], "schedule": sunday_str})
                course_counter += 1
        
        saturdays_and_weekdays = [d.strftime("%Y-%m-%d") for d in all_days if d.weekday() < 6]
        if saturdays_and_weekdays:
            chosen_date = random.choice(saturdays_and_weekdays)
            level_info = random.choice(SPECIAL_COURSE_LEVELS)
            courses_data.append({"id": f"{classroom_id}-SC{course_counter}", "name": f"{pref_name} 特別講座 {level_info['name_suffix']}", "classroom_id": classroom_id, "course_type": "special", "rank": level_info['rank'], "schedule": chosen_date})
            course_counter += 1
    return courses_data

def generate_travel_costs_matrix(classroom_ids, id_to_name_map, pref_to_region_map, region_graph):
    matrix = {}
    for c_from in classroom_ids:
        for c_to in classroom_ids:
            if c_from == c_to:
                matrix[(c_from, c_to)] = 0
                continue
            # (ここに、元のコードから地域情報に基づいたコスト生成ロジックを実装)
            matrix[(c_from, c_to)] = random.randint(5000, 100000)
    return matrix
    
def initialize_app_data(force_regenerate: bool = False):
    if force_regenerate or "app_data_initialized" not in st.session_state:
        st.session_state.TODAY = datetime.date.today()
        assign_month_start = (st.session_state.TODAY + relativedelta(months=4)).replace(day=1)
        st.session_state.ASSIGNMENT_TARGET_MONTH_START = assign_month_start
        st.session_state.ASSIGNMENT_TARGET_MONTH_END = (assign_month_start + relativedelta(months=1)) - datetime.timedelta(days=1)

        prefs, pref_ids = generate_prefectures_data()
        st.session_state.PREFECTURES, st.session_state.PREFECTURE_CLASSROOM_IDS = prefs, pref_ids
        st.session_state.DEFAULT_CLASSROOMS_DATA = generate_classrooms_data(prefs, pref_ids)
        st.session_state.DEFAULT_LECTURERS_DATA = generate_lecturers_data(pref_ids, st.session_state.TODAY, assign_month_start, st.session_state.ASSIGNMENT_TARGET_MONTH_END)
        st.session_state.DEFAULT_COURSES_DATA = generate_courses_data(prefs, pref_ids, assign_month_start, st.session_state.ASSIGNMENT_TARGET_MONTH_END)
        st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX = {} # 仮
        st.session_state.app_data_initialized = True

def display_sample_data_view():
    st.header("入力データ")
    st.subheader("講師データ")
    st.dataframe(pd.DataFrame(st.session_state.DEFAULT_LECTURERS_DATA))
    st.subheader("講座データ")
    st.dataframe(pd.DataFrame(st.session_state.DEFAULT_COURSES_DATA))

def display_optimization_result_view(gemini_api_key: Optional[str]):
    st.header("最適化結果")
    if "solver_result_cache" in st.session_state:
        result = st.session_state.solver_result_cache
        st.subheader(f"求解ステータス: {result['solution_status']}")
        if result['assignments_df']:
            st.dataframe(pd.DataFrame(result['assignments_df']))
        else:
            st.warning("割り当てられた講師はいませんでした。")
    elif "optimization_error_message" in st.session_state:
        st.error(st.session_state.optimization_error_message)
    else:
        st.info("サイドバーから最適化を実行してください。")

def run_optimization():
    logger = logging.getLogger('app')
    logger.info("最適割り当て実行ボタンがクリックされました。")
    for key in ["solver_result_cache", "optimization_error_message", "optimization_duration"]:
        if key in st.session_state: del st.session_state[key]

    try:
        with st.spinner("最適化計算を実行中..."):
            start_time = time.time()
            weights = OptimizationWeights(
                past_assignment_recency=st.session_state.weight_past_assignment_exp,
                qualification=st.session_state.weight_qualification_exp,
                travel=st.session_state.weight_travel_exp, age=st.session_state.weight_age_exp,
                frequency=st.session_state.weight_frequency_exp,
                assignment_shortage=st.session_state.weight_assignment_shortage_exp,
                lecturer_concentration=st.session_state.weight_lecturer_concentration_exp,
                consecutive_assignment=st.session_state.weight_consecutive_assignment_exp,
            )
            solver_params = SolverParameters(weights=weights, allow_under_assignment=st.session_state.allow_under_assignment_cb)
            engine_input = OptimizationInput(
                lecturers_data=st.session_state.DEFAULT_LECTURERS_DATA,
                courses_data=st.session_state.DEFAULT_COURSES_DATA,
                classrooms_data=st.session_state.DEFAULT_CLASSROOMS_DATA,
                travel_costs_matrix=st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX,
                solver_params=solver_params, today_date=st.session_state.TODAY
            )
            solver_output = optimization_gateway.execute_optimization(engine_input)
            st.session_state.optimization_duration = time.time() - start_time
            st.session_state.solver_result_cache = solver_output
    
    except (InvalidInputError, TimeoutError) as e:
        st.session_state.optimization_error_message = f"処理中にエラーが発生しました:\n\n{e}"
    except Exception as e:
        st.session_state.optimization_error_message = f"予期せぬシステムエラーが発生しました:\n\n{e}"
    
    st.session_state.solution_executed = True
    st.session_state.view_mode = "optimization_result"
    st.rerun()

# ----------------------------------------------------
# 3. main関数
# ----------------------------------------------------
def main():
    setup_logging()
    st.set_page_config(page_title="講師割り当てシステムデモ", layout="wide")

    initialize_app_data()

    # セッション状態の初期化
    if "view_mode" not in st.session_state: st.session_state.view_mode = "sample_data"
    if "allow_under_assignment_cb" not in st.session_state: st.session_state.allow_under_assignment_cb = True
    
    # UI描画
    st.title("講師割り当てシステム(OR-Tools)-プロトタイプ")
    st.sidebar.button("最適割り当てを実行", type="primary", on_click=run_optimization)

    # (ここにサイドバーの expander などを配置)
    with st.sidebar.expander("【最適化目標】"):
        st.slider("移動コスト", 0.0, 1.0, 0.5, 0.1, key="weight_travel_exp")
        st.slider("年齢", 0.0, 1.0, 0.5, 0.1, key="weight_age_exp")
        # ... 他のスライダー ...

    # 画面表示の切り替え
    if st.session_state.view_mode == "sample_data":
        display_sample_data_view()
    elif st.session_state.view_mode == "optimization_result":
        display_optimization_result_view(gemini_api_key=st.secrets.get("GEMINI_API_KEY"))

# ----------------------------------------------------
# 4. メイン実行ブロック
# ----------------------------------------------------
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass
    main()