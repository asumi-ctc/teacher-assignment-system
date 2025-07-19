### 変更サマリー

**日時:** 2025-07-19
**要件:** `app.py`が巨大すぎるため、責務に基づいて5つのファイル (`app_main.py`, `app_data_utils.py`, `app_callbacks.py`, `app_ui_views.py`, `gemini_utils.py`) に分割する。
**変更概要:** - `app.py`を削除し、上記の5つの新しいファイルを作成。
-   各ファイルに、それぞれの責務（エントリーポイント、データ生成、コールバック、UI描画、Gemini連携）に応じた関数と`import`文を移動・整理。
-   `app_main.py`から他の`app_*.py`モジュールを呼び出すように修正。