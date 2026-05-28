from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["BaseConfig"]


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )
