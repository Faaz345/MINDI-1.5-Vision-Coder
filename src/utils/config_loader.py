"""
MINDI 1.5 Vision-Coder — Configuration Loader

Typed dataclasses for all YAML configuration files.
Provides validated, type-safe access to model, training, data, and search configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# ── Model Config Dataclasses ──

@dataclass
class BaseModelConfig:
    name: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    parameters: str = "16B"
    license: str = "Apache-2.0"
    context_length: int = 8192
    dtype: str = "bfloat16"


@dataclass
class VisionConfig:
    name: str = "openai/clip-vit-large-patch14"
    image_size: int = 224
    patch_size: int = 14
    hidden_size: int = 1024
    projection_dim: int = 768
    freeze_backbone: bool = True
    trainable_projection: bool = True


@dataclass
class LoraConfig:
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class OutputConfig:
    framework: str = "nextjs-14"
    styling: str = "tailwindcss"
    language: str = "typescript"
    template_format: str = "markdown-codeblock"


@dataclass
class HuggingFaceConfig:
    repo_id: str = "Mindigenous/MINDI-1.5-Vision-Coder"
    private: bool = False
    license: str = "apache-2.0"


@dataclass
class ModelConfig:
    name: str = "MINDI-1.5-Vision-Coder"
    version: str = "1.5.0"
    base: BaseModelConfig = field(default_factory=BaseModelConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)


# ── Training Config Dataclasses ──

@dataclass
class LocalOverrides:
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    optim: str = "adamw_8bit"


@dataclass
class WandbConfig:
    project: str = "mindi-1.5-vision-coder"
    entity: str = "mindigenous"
    tags: list[str] = field(default_factory=lambda: ["mindi-1.5", "lora", "vision-coder"])


@dataclass
class TrainingConfig:
    local_device: str = "cuda"
    cloud_device: str = "cuda"
    precision: str = "bf16"
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = 32
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    max_seq_length: int = 8192
    packing: bool = True
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 5
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    logging_steps: int = 10
    log_dir: str = "./logs/training"
    report_to: str = "wandb"
    eval_strategy: str = "steps"
    eval_steps: int = 250
    eval_samples: int = 1000
    local_overrides: LocalOverrides = field(default_factory=LocalOverrides)
    wandb: WandbConfig = field(default_factory=WandbConfig)


# ── Data Config Dataclasses ──

@dataclass
class DataSource:
    name: str = ""
    description: str = ""
    path: str = ""
    weight: float = 0.0


@dataclass
class DataProcessing:
    tokenizer: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    max_length: int = 8192
    min_length: int = 64
    dedup_strategy: str = "minhash"
    quality_filter: bool = True
    output_dir: str = "./data/processed/"


@dataclass
class DataSplits:
    train: float = 0.95
    validation: float = 0.05


@dataclass
class KnowledgeBase:
    path: str = "./data/knowledge_base/"
    sources: list[str] = field(default_factory=lambda: [
        "nextjs-14-docs", "tailwindcss-docs", "typescript-docs",
        "react-docs", "shadcn-ui-docs",
    ])
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass
class DataConfig:
    name: str = "mindi-1.5-training-data"
    target_size: int = 500000
    format: str = "jsonl"
    sources: list[DataSource] = field(default_factory=list)
    processing: DataProcessing = field(default_factory=DataProcessing)
    splits: DataSplits = field(default_factory=DataSplits)
    knowledge_base: KnowledgeBase = field(default_factory=KnowledgeBase)


# ── Search Config Dataclasses ──

@dataclass
class RateLimit:
    requests_per_minute: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 2


@dataclass
class SearchCache:
    enabled: bool = True
    ttl_hours: int = 24
    max_entries: int = 10000
    storage_path: str = "./data/knowledge_base/search_cache.db"


@dataclass
class DocsScraper:
    enabled: bool = True
    output_dir: str = "./docs/"
    max_pages_per_site: int = 100
    respect_robots_txt: bool = True
    request_delay_seconds: int = 1


@dataclass
class SearchConfig:
    provider: str = "tavily"
    api_key_env: str = "TAVILY_API_KEY"
    max_results: int = 5
    search_depth: str = "advanced"
    include_domains: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=list)
    rate_limit: RateLimit = field(default_factory=RateLimit)
    cache: SearchCache = field(default_factory=SearchCache)
    docs_scraper: DocsScraper = field(default_factory=DocsScraper)


# ── Config Loader ──

def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively convert a dict to a dataclass, handling nested dataclasses and lists."""
    if not isinstance(data, dict):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs: dict[str, Any] = {}

    for key, value in data.items():
        if key not in field_types:
            continue

        field_type = field_types[key]

        # Handle list of dataclasses (e.g., list[DataSource])
        if isinstance(value, list) and hasattr(field_type, "__origin__"):
            # For list[DataSource] etc.
            inner = getattr(field_type, "__args__", [None])[0]
            if inner and hasattr(inner, "__dataclass_fields__"):
                kwargs[key] = [_dict_to_dataclass(inner, item) for item in value]
            else:
                kwargs[key] = value
        elif isinstance(value, dict):
            # Try to match nested dataclass
            field_cls = cls.__dataclass_fields__[key].default_factory if hasattr(cls.__dataclass_fields__[key], "default_factory") else None
            # Get actual type from annotations
            import typing
            actual_type = typing.get_type_hints(cls).get(key)
            if actual_type and hasattr(actual_type, "__dataclass_fields__"):
                kwargs[key] = _dict_to_dataclass(actual_type, value)
            else:
                kwargs[key] = value
        else:
            kwargs[key] = value

    return cls(**kwargs)


class ConfigLoader:
    """
    Loads and provides typed access to all YAML configuration files.

    Usage:
        loader = ConfigLoader()
        model_cfg = loader.model
        training_cfg = loader.training
        data_cfg = loader.data
        search_cfg = loader.search
    """

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        self.config_dir = config_dir or Path(__file__).resolve().parents[2] / "configs"
        self._model: Optional[ModelConfig] = None
        self._training: Optional[TrainingConfig] = None
        self._data: Optional[DataConfig] = None
        self._search: Optional[SearchConfig] = None

    def _load_yaml(self, filename: str) -> dict[str, Any]:
        """Load a YAML file from the config directory."""
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def model(self) -> ModelConfig:
        """Load and return typed model configuration."""
        if self._model is None:
            raw = self._load_yaml("model_config.yaml")
            model_data = raw.get("model", {})
            self._model = _dict_to_dataclass(ModelConfig, model_data)
            # HuggingFace config is at root level
            hf_data = raw.get("huggingface", {})
            if hf_data:
                self._model.huggingface = _dict_to_dataclass(HuggingFaceConfig, hf_data)
        return self._model

    @property
    def training(self) -> TrainingConfig:
        """Load and return typed training configuration."""
        if self._training is None:
            raw = self._load_yaml("training_config.yaml")
            training_data = raw.get("training", {})
            self._training = _dict_to_dataclass(TrainingConfig, training_data)
            # WandB config is at root level
            wandb_data = raw.get("wandb", {})
            if wandb_data:
                self._training.wandb = _dict_to_dataclass(WandbConfig, wandb_data)
        return self._training

    @property
    def data(self) -> DataConfig:
        """Load and return typed data configuration."""
        if self._data is None:
            raw = self._load_yaml("data_config.yaml")
            dataset_data = raw.get("dataset", {})
            self._data = _dict_to_dataclass(DataConfig, dataset_data)
        return self._data

    @property
    def search(self) -> SearchConfig:
        """Load and return typed search configuration."""
        if self._search is None:
            raw = self._load_yaml("search_config.yaml")
            search_data = raw.get("search", {})
            self._search = _dict_to_dataclass(SearchConfig, search_data)
        return self._search

    def reload(self) -> None:
        """Force reload all configurations from disk."""
        self._model = None
        self._training = None
        self._data = None
        self._search = None

    def print_summary(self) -> None:
        """Print a summary of all loaded configurations."""
        print("\n╔══════════════════════════════════════════╗")
        print("║    MINDI 1.5 — Configuration Summary     ║")
        print("╠══════════════════════════════════════════╣")

        m = self.model
        print(f"  Model:      {m.name} v{m.version}")
        print(f"  Base:       {m.base.name} ({m.base.parameters})")
        print(f"  Vision:     {m.vision.name}")
        print(f"  LoRA:       r={m.lora.rank}, alpha={m.lora.alpha}")
        print(f"  Output:     {m.output.framework} + {m.output.styling} + {m.output.language}")

        print("╠──────────────────────────────────────────╣")

        t = self.training
        print(f"  Epochs:     {t.epochs}")
        print(f"  Batch:      {t.batch_size} (effective: {t.effective_batch_size})")
        print(f"  LR:         {t.learning_rate}")
        print(f"  Precision:  {t.precision}")
        print(f"  Seq length: {t.max_seq_length}")

        print("╠──────────────────────────────────────────╣")

        d = self.data
        print(f"  Dataset:    {d.name} ({d.target_size:,} target)")
        print(f"  Sources:    {len(d.sources)}")
        print(f"  Format:     {d.format}")

        print("╠──────────────────────────────────────────╣")

        s = self.search
        print(f"  Provider:   {s.provider}")
        print(f"  Max results: {s.max_results}")
        print(f"  Domains:    {len(s.include_domains)} included, {len(s.exclude_domains)} excluded")

        print("╚══════════════════════════════════════════╝\n")
