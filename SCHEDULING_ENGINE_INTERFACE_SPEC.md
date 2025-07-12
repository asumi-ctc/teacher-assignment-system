 # スケジューリング最適化エンジン インターフェース仕様書
 
 ## 1. 概要
 
 本ドキュメントは、講師割り当てシステム（Djangoを想定）にスケジューリング最適化エンジン（以下、エンジン）を実装する上でのインターフェース仕様を解説します。
 エンジンは、講師、講座、教室のデータと、最適化の目標を定義するパラメータ群を入力として受け取り、最適な割り当て計画を出力するコンポーネントです。呼び出し側のシステムは、本仕様書に定義されたデータ構造に従ってエンジンにデータを渡し、エンジンから返される結果を受け取って後続処理（データベースへの保存やUIへの表示など）を行う必要があります。
 
 ### エンジンを構成する主要なソースコード
 
 | ファイル名                      | 役割                                                                                               |
 | :------------------------------ | :------------------------------------------------------------------------------------------------- |
 | `optimization_gateway.py`       | 最適化エンジンへのゲートウェイ。入力データの検証、プロセスのタイムアウト監視、ロギングを行い、ソルバーを呼び出す。 |
 | `optimization_solver.py`        | OR-Tools (CP-SAT) を使用したコア最適化ロジック。モデル、変数、制約、目的関数を定義する。               |
 | `utils/error_definitions.py`    | `InvalidInputError`などのカスタム例外クラスを定義する。                                            |
 | `utils/logging_config.py`       | アプリケーション全体のロギング設定を行う。                                                         |
 | `utils/types.py`                | `TypedDict`を用いて、講師や講座などのデータ構造の型を定義する。                                    |
 
 ## 2. エンジン呼び出し
 
 エンジンは、ゲートウェイモジュール `optimization_gateway` を通じて呼び出されます。ゲートウェイは入力データのバリデーション、プロセスのタイムアウト監視、ロギングなどの責務を持ちます。
 
 最適化処理は10秒以上かかる可能性があるため、Webアプリケーションの応答性を損なわないよう、非同期タスクキュー（Celery）を利用した実装が推奨されます。
 これにより、ユーザーは最適化の実行をリクエストした後、すぐに別の操作に移ることができ、処理が完了したら結果を確認できます。
 
 以下に、DjangoとCeleryを利用した非同期呼び出しのサンプル実装を示します。
 
 ### 2.1. Celeryタスクの定義
 
 最適化処理そのものをカプセル化するCeleryタスクを定義します。
 このタスクは、データベースから最新のデータを取得し、最適化エンジンを呼び出します。
 
 ```python
 # django_app/tasks.py
 
 from celery import shared_task
 import datetime
 import logging
 
 import optimization_gateway
 from utils.error_definitions import InvalidInputError, ProcessTimeoutError, ProcessExecutionError
 
 # データベースモデルからデータを取得し、仕様に合った形式に変換するヘルパー関数 (実装は省略)
 from .utils import prepare_lecturers_data, prepare_courses_data, prepare_classrooms_data, prepare_travel_costs
 
 logger = logging.getLogger(__name__)
 
 @shared_task(bind=True)
 def execute_optimization_task(self, optimization_params: dict):
     """
     Celeryタスクとして最適化を実行します。
     大量のデータを引数で渡すのではなく、タスク内でデータを準備します。
 
     Args:
         optimization_params (dict): 最適化の重みや対象月など、
                                     データ準備と最適化に必要なパラメータを含む辞書。
     """
     try:
         logger.info(f"Starting optimization task with params: {optimization_params}")
         self.update_state(state='PROGRESS', meta={'status': 'データの準備中...'})
 
         # --- 1. データの準備 ---
         # パラメータに基づき、DB等から最新のデータを取得・変換
         lecturers = prepare_lecturers_data()
         courses = prepare_courses_data(target_month=optimization_params.get("target_month"))
 
         # --- 2. エンジンの呼び出し ---
         self.update_state(state='PROGRESS', meta={'status': '最適化エンジンを実行中...'})
         
         # optimization_params から必要な引数を抽出してゲートウェイに渡す
         solver_input_args = {
             "lecturers_data": lecturers,
             "courses_data": courses,
             "today_date": datetime.date.today(),
             **optimization_params  # 重みやオプションなどを展開して渡す
         }
 
         solver_output = optimization_gateway.run_optimization_with_monitoring(**solver_input_args)
 
         logger.info(f"Optimization task completed with status: {solver_output.get('status')}")
         return solver_output
 
     except (InvalidInputError, ProcessTimeoutError, ProcessExecutionError) as e:
         logger.error(f"Handled error during optimization task: {e}", exc_info=True)
         # Celeryタスクを失敗としてマークし、エラー情報をmetaに格納
         self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
         # Celeryのデフォルトのエラーハンドリングに任せるために例外を再送出
         raise
     except Exception as e:
         logger.error(f"Unexpected error during optimization task: {e}", exc_info=True)
         self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': '予期せぬエラーが発生しました。'})
         raise
 ```
 
 ### 2.2. 最適化タスクの起動API
 
 ユーザーからのリクエストを受け付け、Celeryタスクを非同期で起動するAPIエンドポイントです。
 クライアントには即座にタスクIDを返します。
 
 ```python
 # django_app/views.py
 
 from django.http import JsonResponse
 from django.views.decorators.http import require_POST
 from django.views.decorators.csrf import csrf_exempt
 import json
 from .tasks import execute_optimization_task
 
 @csrf_exempt
 @require_POST
 def start_optimization_api(request):
     """最適化タスクを開始し、タスクIDを返すAPI"""
     try:
         # リクエストボディから最適化パラメータを取得
         params = json.loads(request.body)
         
         # Celeryタスクを非同期で実行
         # 大量のデータそのものではなく、パラメータのみを渡す
         task = execute_optimization_task.delay(optimization_params=params)
         
         # クライアントにタスクIDを返す
         # クライアントはこのIDを使って結果をポーリングする
         return JsonResponse({"task_id": task.id, "message": "最適化処理を開始しました。"}, status=202)
 
     except json.JSONDecodeError:
         return JsonResponse({"status": "error", "message": "リクエストボディのJSON形式が不正です。"}, status=400)
     except Exception as e:
         # logging.error(f"Failed to start optimization task: {e}", exc_info=True)
         return JsonResponse({"status": "error", "message": "最適化タスクの開始に失敗しました。"}, status=500)
 ```
 
 ### 2.3. 結果取得API
 
 クライアントがタスクIDを使って処理状況や最終的な結果を取得するためのAPIエンドポイントです。
 
 ```python
 # django_app/views.py
 
 from django.http import JsonResponse
 from django.views.decorators.http import require_GET
 from celery.result import AsyncResult
 
 @require_GET
 def get_optimization_result_api(request, task_id: str):
     """タスクIDに基づいて最適化の状況や結果を返すAPI"""
     task_result = AsyncResult(task_id)
 
     response_data = {
         'task_id': task_id,
         'status': task_result.state,
         'result': None
     }
 
     if task_result.state == 'SUCCESS':
         response_data['result'] = task_result.get()
         return JsonResponse(response_data, status=200)
     elif task_result.state == 'FAILURE':
         # task.info は update_state で設定した meta 情報を格納する
         response_data['result'] = task_result.info 
         return JsonResponse(response_data, status=500)
     else: # PENDING, PROGRESS, RETRY など
         # task.info には進捗メッセージなども格納できる
         response_data['result'] = task_result.info
         return JsonResponse(response_data, status=202) # Accepted (まだ処理中)
 ```

 ## 3. 最適化パラメータ定義の取得

Djangoアプリケーション側で、最適化目標の重み調整UI（スライダーなど）や許容条件の選択UI（チェックボックスなど）を動的に生成するために、エンジン側がパラメータの定義情報を提供します。
これにより、将来エンジン側で新しい最適化目標が追加された場合でも、UI側のコードを修正することなく対応が可能になります。

### 3.1. APIエンドポイント

- **URL**: `/api/optimization/definitions` (推奨)
- **HTTP Method**: `GET`

### 3.2. 成功レスポンス (200 OK)

パラメータの定義情報をJSON形式で返します。

- **Content-Type**: `application/json`

### 3.3. 失敗レスポンス (500 Internal Server Error)
サーバー内部でエラーが発生した場合に返されます。

 ## 4. 入力データ仕様
 
 ### 4.1. `lecturers_data`
 
 - **型**: `List[Dict[str, Any]]`
 - **説明**: 講師情報のリスト。各要素は一人の講師を表す辞書です。
 - **辞書のキー仕様**:
   | キー名 | 型 | 必須 | 説明 |
   | :--- | :--- | :--- | :--- |
   | `id` | `str` | ✔ | 講師の一意なID。例: `"L1"` |
   | `name` | `str` | ✔ | 講師名。例: `"講師001"` |
   | `age` | `int` | ✔ | 講師の年齢。 |
   | `home_classroom_id` | `str` | ✔ | 講師の自宅最寄り教室のID。移動コスト計算の始点となる。 |
   | `qualification_general_rank` | `int` | ✔ | 一般資格ランク (1:高 - 5:低)。 |
   | `qualification_special_rank` | `Optional[int]` | ✔ | 特別資格ランク (1:高 - 5:低)。保有しない場合は `None`。 |
   | `availability` | `List[datetime.date]` | ✔ | 講師が勤務可能な日付オブジェクトのリスト。 |
   | `past_assignments` | `List[Dict]` | ✔ | 過去の割り当て履歴のリスト。各要素は `{"classroom_id": str, "date": datetime.date}` の形式。 |
 
 ### 4.2. `courses_data`
 
 - **型**: `List[Dict[str, Any]]`
 - **説明**: 割り当て対象となる講座情報のリスト。
 - **辞書のキー仕様**:
   | キー名 | 型 | 必須 | 説明 |
   | :--- | :--- | :--- | :--- |
   | `id` | `str` | ✔ | 講座の一意なID。例: `"P1-GC1"` |
   | `name` | `str` | ✔ | 講座名。例: `"北海道 一般講座 初心"` |
   | `classroom_id` | `str` | ✔ | 開催される教室のID。 |
   | `course_type` | `str` | ✔ | 講座タイプ。`"general"` または `"special"`。 |
   | `rank` | `int` | ✔ | 講座の要求ランク (1:高 - 5:低)。 |
   | `schedule` | `datetime.date` | ✔ | 講座の開催日。 |
 
 ### 4.3. `classrooms_data`
 
 - **型**: `List[Dict[str, Any]]`
 - **説明**: 教室情報のリスト。
 - **辞書のキー仕様**:
   | キー名 | 型 | 必須 | 説明 |
   | :--- | :--- | :--- | :--- |
   | `id` | `str` | ✔ | 教室の一意なID。例: `"P1"` |
   | `location` | `str` | ✔ | 教室の場所（都道府県名）。例: `"北海道"` |
 
 ### 4.5. 最適化目標の重みパラメータ
 
 - **型**: `float`
 - **説明**: 各最適化目標の重要度を調整する重み。通常 `0.0` から `1.0` の範囲で指定します。`0.0` を指定すると、その最適化目標は考慮されません。
 - **パラメータ一覧**:
   - `weight_age`
   - `weight_frequency`
   - `weight_qualification`
   - `weight_past_assignment_recency`
   - `weight_lecturer_concentration`
   - `weight_consecutive_assignment`
 
 ### 4.6. その他のオプション
 
 - **`today_date`**:
   - **型**: `datetime.date`
   - **説明**: 処理実行日の日付オブジェクト。過去の割り当てからの経過日数計算などに使用されます。
 - **`fixed_assignments`**:
   - **型**: `Optional[List[Tuple[str, str]]]`
   - **説明**: `(講師ID, 講座ID)` のタプルのリスト。ここで指定されたペアは、コストに関わらず強制的に割り当てられます。
 - **`forced_unassignments`**:
   - **型**: `Optional[List[Tuple[str, str]]]`
   - **説明**: `(講師ID, 講座ID)` のタプルのリスト。ここで指定されたペアは、割り当て候補から除外されます。
 
 ## 5. 出力データ仕様
 
 エンジンは、処理結果を単一の辞書オブジェクトとして返します。
 
 - **型**: `Dict[str, Any]`
 - **キー仕様**:
   | キー名 | 型 | 説明 |
   | :--- | :--- | :--- |
   | `status` | `str` | 処理全体のステータス。 `"success"` または `"error"`。 |
   | `message` | `str` | 処理結果に関する人間可読なメッセージ。 |
   | `solution_status` | `str` | OR-Toolsソルバーの求解ステータス。例: `"OPTIMAL"`, `"FEASIBLE"`, `"INFEASIBLE"`。 |
   | `objective_value` | `Optional[float]` | 目的関数の最終値。解が見つからない場合は `None`。 |
   | `assignments_df` | `List[Dict]` | 割り当て結果のリスト。各要素が1つの割り当てを表す辞書。空リストの場合もあり。 |
   | `lecturer_course_counts` | `Dict[str, int]` | 各講師が何件の講座に割り当てられたかの集計。キーは講師ID。 |
   | `course_assignment_counts` | `Dict[str, int]` | 各講座に何人の講師が割り当てられたかの集計。キーは講座ID。 |
   | `course_remaining_capacity` | `Dict[str, int]` | 各講座の残りの割り当て可能枠数。 |
   | `raw_solver_status_code` | `int` | OR-Toolsソルバーが返す生のステータスコード。 |
 
 ### 5.1. `assignments_df` の要素の辞書構造
 
 `assignments_df` リスト内の各辞書は、以下のキーを持ちます。
 
 | キー名 | 型 | 説明 |
   | :--- | :--- | :--- |
   | `講師ID` | `str` | 割り当てられた講師のID。 |
   | `講師名` | `str` | 割り当てられた講師の名前。 |
   | `講座ID` | `str` | 割り当てられた講座のID。 |
   | `講座名` | `str` | 割り当てられた講座の名前。 |
   | `教室ID` | `str` | 講座が開催される教室のID。 |
   | `教室名` | `str` | 講座が開催される教室の場所。 |
   | `スケジュール` | `str` | 講座の開催日 (`YYYY-MM-DD`形式)。 |
   | `今回の割り当て回数` | `int` | この最適化で当該講師が割り当てられた総回数。 |
   | `当該教室最終割当日からの日数` | `int` | 当該教室への最終割り当て日からの経過日数。新規の場合は負の値。 |
   | ... | ... | その他、最適化の評価に使用されたコスト関連の列。 |
 
 ## 6. エラーハンドリング
 
 ゲートウェイモジュールは、エンジン実行前にバリデーションを行います。バリデーションエラーが発生した場合、エンジンは実行されずに例外が送出されます。
 
 - **`InvalidInputError`**: 入力データの型、構造、または整合性に問題がある場合に発生します。
 - **`ProcessTimeoutError`**: 最適化処理が規定のタイムアウト時間を超えた場合に発生します。
 - **`ProcessExecutionError`**: 最適化を実行するサブプロセスで予期せぬエラーが発生した場合に発生します。
 
 呼び出し側はこれらの例外を捕捉し、適切に処理する必要があります。
 
 ## 7. ロギング
 
 エンジンを構成する各モジュール (`optimization_gateway`, `optimization_solver`) は、Python標準の `logging` モジュールを利用して処理の進捗や内部状態に関するログを出力します。
 ロガー名は、それぞれのモジュール名 (`"optimization_gateway"`, `"optimization_solver"`) に対応しています。
 
 ### 7.1. Djangoへの統合
 
 エンジンのロギング設定は、Djangoの `settings.py` に容易に統合できるよう設計されています。
 `utils/logging_config.py` 内に定義されている `LOGGING_CONFIG` 辞書は、Djangoの `LOGGING` 設定と互換性のある形式です。
 
 以下に、Djangoプロジェクトの `settings.py` にロギング設定を組み込む際の推奨設定例を示します。
 この設定により、エンジンからのログをコンソールと、モジュールごとのログファイルの両方に出力できます。
 
 ```python
 # your_project/settings.py
 
 import os
 
 # プロジェクトのベースディレクトリを基準にログディレクトリを設定
 BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 LOG_DIR = os.path.join(BASE_DIR, 'logs')
 os.makedirs(LOG_DIR, exist_ok=True)
 
 LOGGING = {
     'version': 1,
     'disable_existing_loggers': False,
     'formatters': {
         'standard': {
             'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
         },
     },
     'handlers': {
         'console': {
             'class': 'logging.StreamHandler',
             'formatter': 'standard'
         },
         'gateway_file': {
             'class': 'logging.handlers.RotatingFileHandler',
             'filename': os.path.join(LOG_DIR, 'optimization_gateway.log'),
             'maxBytes': 1024 * 1024 * 5,  # 5 MB
             'backupCount': 5,
             'encoding': 'utf-8',
             'formatter': 'standard',
         },
         'solver_file': {
             'class': 'logging.handlers.RotatingFileHandler',
             'filename': os.path.join(LOG_DIR, 'optimization_solver.log'),
             'maxBytes': 1024 * 1024 * 10, # ソルバーログは大きくなる可能性があるため10MB
             'backupCount': 5,
             'encoding': 'utf-8',
             'formatter': 'standard',
         },
     },
     'loggers': {
         # Django自体のロガー設定
         'django': {
             'handlers': ['console'],
             'level': 'INFO',
             'propagate': False,
         },
         # 最適化ゲートウェイのロガー設定
         'optimization_gateway': {
             'handlers': ['console', 'gateway_file'],
             'level': 'INFO',
             'propagate': False, # ルートロガーへの伝播を防ぐ
         },
         # 最適化ソルバーのロガー設定
         'optimization_solver': {
             'handlers': ['console', 'solver_file'],
             'level': 'INFO',
             'propagate': False, # ルートロガーへの伝播を防ぐ
         },
     },
 }
 ```
 
 **設定のポイント:**
 
 - **`RotatingFileHandler`**: ログファイルが肥大化し続けるのを防ぐため、一定サイズに達すると新しいファイルに切り替える（ローテーションする）ハンドラーを使用します。
 - **`propagate: False`**: エンジン用のロガーで捕捉されたログが、上位のロガー（ルートロガーなど）に伝播して二重に出力されるのを防ぎます。
 - **ファイルパス**: `os.path.join(BASE_DIR, 'logs', ...)` のように、Djangoプロジェクトのベースディレクトリからの相対パスでログファイルの場所を指定するのが一般的です。
 
 この設定を `settings.py` に追加することで、CeleryワーカーやDjangoアプリケーションサーバーから実行されたエンジンのログが、指定されたファイルに記録されるようになります。