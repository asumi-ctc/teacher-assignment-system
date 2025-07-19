# ==============================================================================
# 2. app_data_utils.py (データ生成と初期化)
# ==============================================================================
import streamlit as st
import datetime
import random
from dateutil.relativedelta import relativedelta
import logging

# このファイルは外部モジュールに依存しないため、変更はありません。

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
        classrooms_data.append({
            "id": prefecture_classroom_ids[i],
            "location": pref_name
        })
    return classrooms_data

def generate_lecturers_data(prefecture_classroom_ids, today_date, assignment_target_month_start, assignment_target_month_end):
    lecturers_data = []
    availability_period_start = assignment_target_month_start - relativedelta(months=1)
    availability_period_end = assignment_target_month_end + relativedelta(months=1)

    all_possible_dates_for_availability = []
    current_date_iter = availability_period_start
    while current_date_iter <= availability_period_end:
        all_possible_dates_for_availability.append(current_date_iter)
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
                "date": assignment_date
            })
        past_assignments.sort(key=lambda x: x["date"], reverse=True)
        lecturers_data.append({
            "id": f"L{i}",
            "name": f"講師{i:03d}",
            "age": random.randint(22, 65),
            "home_classroom_id": "P13",
            "qualification_general_rank": general_rank,
            "qualification_special_rank": special_rank,
            "availability": availability,
            "past_assignments": past_assignments
        })
    return lecturers_data

def generate_courses_data(prefectures, prefecture_classroom_ids, assignment_target_month_start, assignment_target_month_end):
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

    sundays_in_target_month = []
    current_date_iter = assignment_target_month_start
    while current_date_iter <= assignment_target_month_end:
        if current_date_iter.weekday() == 6:
            sundays_in_target_month.append(current_date_iter)
        current_date_iter += datetime.timedelta(days=1)

    saturdays_in_target_month_for_special = []
    weekdays_in_target_month_for_special = []
    all_days_in_target_month_for_special_obj = []
    current_date_iter = assignment_target_month_start
    while current_date_iter <= assignment_target_month_end:
        all_days_in_target_month_for_special_obj.append(current_date_iter)
        if current_date_iter.weekday() == 5:
            saturdays_in_target_month_for_special.append(current_date_iter)
        elif current_date_iter.weekday() < 5:
            weekdays_in_target_month_for_special.append(current_date_iter)
        current_date_iter += datetime.timedelta(days=1)

    courses_data = []
    course_counter = 1
    for i, pref_classroom_id in enumerate(prefecture_classroom_ids):
        pref_name = prefectures[i]

        for sunday_date in sundays_in_target_month:
            selected_levels_for_general = random.sample(GENERAL_COURSE_LEVELS, min(2, len(GENERAL_COURSE_LEVELS)))
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

def initialize_app_data(force_regenerate: bool = False):
    logger = logging.getLogger('app')
    logger.info(f"Entering initialize_app_data(force_regenerate={force_regenerate})")
    
    if force_regenerate or not st.session_state.get("app_data_initialized"):
        logger.info("Regenerating data...")
        st.session_state.TODAY = datetime.date.today()
        assignment_target_month_start_val = (st.session_state.TODAY + relativedelta(months=4)).replace(day=1)
        st.session_state.ASSIGNMENT_TARGET_MONTH_START = assignment_target_month_start_val
        next_month_val = assignment_target_month_start_val + relativedelta(months=1)
        st.session_state.ASSIGNMENT_TARGET_MONTH_END = next_month_val - datetime.timedelta(days=1)
        
        PREFECTURES_val, PREFECTURE_CLASSROOM_IDS_val = generate_prefectures_data()
        st.session_state.PREFECTURES = PREFECTURES_val
        st.session_state.PREFECTURE_CLASSROOM_IDS = PREFECTURE_CLASSROOM_IDS_val
        
        st.session_state.DEFAULT_CLASSROOMS_DATA = generate_classrooms_data(st.session_state.PREFECTURES, st.session_state.PREFECTURE_CLASSROOM_IDS)
        st.session_state.DEFAULT_LECTURERS_DATA = generate_lecturers_data(st.session_state.PREFECTURE_CLASSROOM_IDS, st.session_state.TODAY, st.session_state.ASSIGNMENT_TARGET_MONTH_START, st.session_state.ASSIGNMENT_TARGET_MONTH_END)
        st.session_state.DEFAULT_COURSES_DATA = generate_courses_data(st.session_state.PREFECTURES, st.session_state.PREFECTURE_CLASSROOM_IDS, st.session_state.ASSIGNMENT_TARGET_MONTH_START, st.session_state.ASSIGNMENT_TARGET_MONTH_END)
        
        st.session_state.app_data_initialized = True
        logger.info("Data generation logic executed.")

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
        lecturers[0]['availability'][0] = "2025/01/01"
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

def generate_invalid_sample_data():
    initialize_app_data(force_regenerate=True)
    logger = logging.getLogger('app')
    logger.info("Generated a fresh set of valid data to be corrupted for testing.")
    corruption_functions = [
        _corrupt_duplicate_classroom_id, _corrupt_missing_classroom_location,
        _corrupt_lecturer_bad_age, _corrupt_lecturer_bad_availability_date,
        _corrupt_course_bad_rank, _corrupt_course_with_nonexistent_classroom,
    ]
    chosen_corruption = random.choice(corruption_functions)
    description = chosen_corruption()
    logger.info(f"Data corruption applied: {description}")
    return description
