from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# ==================================================
# Project Paths
# ==================================================

# 현재 파일:
# team-b-database/app/config.py
#
# parent       -> app
# parent.parent -> team-b-database
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"


# ==================================================
# Settings
# ==================================================

class Settings(BaseSettings):

    # ==============================================
    # Database
    # ==============================================

    db_host: str
    db_port: int = 5432
    db_name: str = "team_b_ai"
    db_user: str = "postgres"
    db_password: str
    db_sslmode: str = "require"

    # ==============================================
    # arXiv
    # ==============================================

    arxiv_api_url: str = (
        "https://export.arxiv.org/api/query"
    )

    core_target_per_axis: int = 5

    arxiv_candidates_per_axis: int = 10

    request_delay: int = 5

    request_timeout: int = 60

    # ==============================================
    # Local Paths
    # ==============================================

    temp_dir: Path = Path("data/tmp")

    report_dir: Path = Path("data/reports")

    normalization_version: str = "v1"

    # ==============================================
    # Pydantic Settings
    # ==============================================

    model_config = SettingsConfigDict(
        # 핵심:
        # 현재 실행 위치가 어디든
        # 프로젝트 루트의 .env를 읽는다.
        env_file=ENV_FILE,

        env_file_encoding="utf-8",

        case_sensitive=False,

        extra="ignore",
    )

    # ==============================================
    # PostgreSQL DSN
    # ==============================================

    @property
    def dsn(self) -> str:
        return (
            f"host={self.db_host} "
            f"port={self.db_port} "
            f"dbname={self.db_name} "
            f"user={self.db_user} "
            f"password={self.db_password} "
            f"sslmode={self.db_sslmode}"
        )


# ==================================================
# Global Settings
# ==================================================

settings = Settings()