class AegisException(Exception):
    """Base exception class for all errors in Project AEGIS"""
    pass

class DataLeakageError(AegisException):
    """Raised if future data leaks into the training pipeline"""
    pass

class RiskLimitBreachedError(AegisException):
    """Raised if execution breaches predefined drawdown or exposure limits"""
    pass

class ConfigError(AegisException):
    """Raised if invalid parameters are loaded at system startup"""
    pass
