# app.py の run_optimization 関数の変更後イメージ

from optimization_gateway import OptimizationInput, SolverParameters, OptimizationWeights

def run_optimization():
    # ... (ログやセッションのクリア) ...

    try:
        with st.spinner("最適化計算を実行中..."):
            start_time = time.time()
            
            # --- 1. 設定オブジェクトの組み立て ---
            weights = OptimizationWeights(
                past_assignment_recency=st.session_state.get("weight_past_assignment_exp", 0.5),
                qualification=st.session_state.get("weight_qualification_exp", 0.5),
                travel=st.session_state.get("weight_travel_exp", 0.5),
                age=st.session_state.get("weight_age_exp", 0.5),
                frequency=st.session_state.get("weight_frequency_exp", 0.5),
                assignment_shortage=st.session_state.get("weight_assignment_shortage_exp", 0.5),
                lecturer_concentration=st.session_state.get("weight_lecturer_concentration_exp", 0.5),
                consecutive_assignment=st.session_state.get("weight_consecutive_assignment_exp", 0.5),
            )
            solver_params = SolverParameters(
                weights=weights,
                allow_under_assignment=st.session_state.allow_under_assignment_cb,
                # max_search_seconds は gateway 側でデフォルト値が設定される
            )
            
            # --- 2. 入力オブジェクトの組み立て ---
            engine_input_data = OptimizationInput(
                lecturers_data=st.session_state.DEFAULT_LECTURERS_DATA,
                courses_data=st.session_state.DEFAULT_COURSES_DATA,
                classrooms_data=st.session_state.DEFAULT_CLASSROOMS_DATA,
                travel_costs_matrix=st.session_state.DEFAULT_TRAVEL_COSTS_MATRIX,
                solver_params=solver_params,
                today_date=st.session_state.TODAY,
                fixed_assignments=st.session_state.get("fixed_assignments_for_solver"),
                forced_unassignments=st.session_state.get("forced_unassignments_for_solver")
            )

            # --- 3. 新しいエントリーポイントを呼び出し ---
            solver_output = optimization_gateway.execute_optimization(engine_input_data)
            
            # ... (処理時間計測と結果の保存) ...

    except InvalidInputError as e:
        # ... (エラーハンドリング) ...
    except Exception as e:
        # ... (エラーハンドリング) ...
    finally:
        # ... (ログの読み込み) ...