import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    OPENAI_API_KEY: str | None

    # SQL Server
    # 학습(train)은 DB가 필요 없으므로 optional 처리
    DB_SERVER: Optional[str] = None
    DB_NAME: Optional[str] = None
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_DRIVER: str = "ODBC Driver 17 for SQL Server"

    # S3 (음성 분석 API에서 오디오 다운로드)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "ap-northeast-2"
    S3_BUCKET_NAME: Optional[str] = None

    # 학습 모델 디렉터리(프로젝트 루트 기준 상대 경로 또는 절대 경로)
    MODEL_DIR: str = "delivery_learning_models"
    ANALYZER_VERSION: str = "delivery_learning_api_v1"

    # 설정 시 Health/Analyze 요청에 동일 헤더 필요 (브라우저에는 넣지 말 것)
    INTERNAL_API_SECRET: Optional[str] = None

    @property
    def db_connection_string(self) -> str:
        if not (self.DB_SERVER and self.DB_NAME and self.DB_USER and self.DB_PASSWORD):
            raise RuntimeError("DB 접속을 위해 DB_SERVER/DB_NAME/DB_USER/DB_PASSWORD 환경변수가 필요합니다.")
        return (
            f"mssql+pyodbc://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_SERVER}/{self.DB_NAME}"
            f"?driver={self.DB_DRIVER.replace(' ', '+')}"
        )


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"환경변수 {name} 가 필요합니다.")
    return v


settings = Settings(
    OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY"),
    DB_SERVER=os.environ.get("DB_SERVER"),
    DB_NAME=os.environ.get("DB_NAME"),
    DB_USER=os.environ.get("DB_USER"),
    DB_PASSWORD=os.environ.get("DB_PASSWORD"),
    AWS_ACCESS_KEY_ID=os.environ.get("AWS_ACCESS_KEY_ID"),
    AWS_SECRET_ACCESS_KEY=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    AWS_REGION=os.environ.get("AWS_REGION") or "ap-northeast-2",
    S3_BUCKET_NAME=os.environ.get("S3_BUCKET_NAME"),
    MODEL_DIR=os.environ.get("MODEL_DIR", "delivery_learning_models"),
    ANALYZER_VERSION=os.environ.get("ANALYZER_VERSION", "delivery_learning_api_v1"),
    INTERNAL_API_SECRET=os.environ.get("INTERNAL_API_SECRET"),
)

