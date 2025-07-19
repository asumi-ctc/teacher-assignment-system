# ==============================================================================
# 5. gemini_utils.py (Gemini API関連)
# ==============================================================================
import google.generativeai as genai
import logging
import pandas as pd
from typing import Optional, Tuple

# このファイルは外部モジュールに依存しないため、変更はありません。

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
