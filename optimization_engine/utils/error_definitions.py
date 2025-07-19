# ==============================================================================
# 4. utils/error_definitions.py (カスタム例外)
# ==============================================================================
"""
アプリケーション全体で使用されるカスタム例外を定義します。
"""

class BaseOptimizationError(Exception):
    """最適化関連エラーの基底クラス。"""
    pass

class InvalidInputError(BaseOptimizationError):
    """
    最適化エンジンへの入力データが無効な場合に送出される例外。
    データバリデーションの失敗など。
    """
    pass

class ProcessExecutionError(BaseOptimizationError):
    """
    最適化を実行する子プロセスで予期せぬエラーが発生した場合に送出される例外。
    """
    pass

class ProcessTimeoutError(BaseOptimizationError):
    """
    最適化を実行する子プロセスがタイムアウトした場合に送出される例外。
    """
    pass

class SolverError(BaseOptimizationError):
    """
    ソルバーが解を見つけられなかった、または予期せぬ状態で終了した場合に送出される例外。
    (例: INFEASIBLE, UNKNOWN)
    """
    pass
