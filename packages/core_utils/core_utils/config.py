import os

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class XGBoostConfig(BaseModel):
    n_estimators: int = Field(default=500, ge=10)
    max_depth: int = Field(default=3, ge=1, le=10)
    learning_rate: float = Field(default=0.01, gt=0.0, lt=1.0)
    subsample: float = Field(default=0.7, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=0.7, gt=0.0, le=1.0)
    reg_alpha: float = Field(default=0.5, ge=0.0)
    reg_lambda: float = Field(default=1.5, ge=0.0)
    eval_metric: str = "logloss"


class PPOConfig(BaseModel):
    learning_rate: float = Field(default=3e-4, gt=0.0, lt=1e-1)
    n_steps: int = Field(default=256, ge=16)
    batch_size: int = Field(default=64, ge=8)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, gt=0.0, le=1.0)
    clip_range: float = Field(default=0.2, gt=0.0, lt=1.0)
    ent_coef: float = Field(default=0.01, ge=0.0)
    total_timesteps: int = Field(default=100000, ge=100)


class StrategyParams(BaseModel):
    symbols: list[str] = ["RELIANCE.NS", "TCS.NS"]
    commission: float = Field(default=0.001, ge=0.0)
    slippage: float = Field(default=0.0005, ge=0.0)
    consensus_gate: str = "hard_agreement"


class RiskParams(BaseModel):
    max_drawdown_limit: float = Field(default=0.15, gt=0.0, lt=1.0)
    max_volatility_limit: float = Field(default=0.03, gt=0.0, lt=1.0)
    leverage_cap: float = Field(default=1.0, ge=1.0)


class AegisSettings(BaseSettings):
    environment: str = "development"
    database_uri: str = "sqlite:///data/sandbox.db"
    log_level: str = "INFO"
    use_mock_broker: bool = True

    strategy: StrategyParams = StrategyParams()
    xgboost: XGBoostConfig = XGBoostConfig()
    ppo: PPOConfig = PPOConfig()
    risk: RiskParams = RiskParams()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AEGIS_",
        extra="ignore",
    )


def load_config(config_path: str | None = None) -> AegisSettings:
    """Load configuration from YAML and environment overrides"""
    if config_path is None:
        config_path = os.environ.get("ENV_CONFIG_PATH", "configs/base.yaml")

    if not os.path.exists(config_path):
        return AegisSettings()

    with open(config_path) as f:
        yaml_data = yaml.safe_load(f) or {}

    return AegisSettings(**yaml_data)
