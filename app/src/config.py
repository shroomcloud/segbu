import yaml
from typing import Optional, List, Dict, Final, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


# models for validating yaml config
class AppConfig(BaseModel):
    host: str
    port: int
    keep_alive: int


class ModelConfig(BaseModel):
    input_size: Final[Literal[384]]
    batch_size: int = Field(..., ge=1)
    tile_res: int = Literal[256, 384, 512, 768, 1024]
    to_resize: Optional[int] = None
    overlap: float = Field(..., ge=0.0, le=1.0)
    norm_mean: List[float]
    norm_std: List[float]


class InferenceConfig(BaseModel):
    execution_providers: List[str]
    enable_fallback: bool


class ServiceConfig(BaseModel):
    app: AppConfig
    models: Dict[str, ModelConfig]
    inference: InferenceConfig


# class representing session config
class Config:
    def __init__(self, path=None):
        path = path or os.getenv("API_CONFIG", "/mnt/config/config.yaml")

        with open(path, "r") as f:
            self.data = yaml.safe_load(f)
        self.config = ServiceConfig.model_validate(self.data)

        self.KEEP_ALIVE = self.config.app.keep_alive
        self.HOST = self.config.app.host
        self.PORT = self.config.app.port
        self.AVAILABLE_MODELS = list(self.config.models)
        self.MODEL_PATHS = dict()
        for model in self.AVAILABLE_MODELS:
            self.MODEL_PATHS[model] = os.getenv(model.upper())
        self.EXEC_PROVIDERS = self.config.inference.execution_providers
        self.ENABLE_FALLBACK = self.config.inference.enable_fallback
        self.MODEL_CONFIGS = {
            model: schema.model_dump() for model, schema in self.config.models.items()
        }

    def __getitem__(self, key):
        return self.data.get(key)


config = Config()
