import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


def get_repo_root() -> Path:
    env_root = os.environ.get("RETINAX_REPO_ROOT")
    if env_root:
        return Path(env_root)
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").exists() and (parent / "pyproject.toml").exists():
            return parent
    return current.parents[2]


REPO_ROOT = get_repo_root()


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_paths(obj: Any, root: Path) -> Any:
    if isinstance(obj, str):
        p = Path(obj)
        if not p.is_absolute():
            return str(root / p)
        return obj
    elif isinstance(obj, dict):
        return {k: _resolve_paths(v, root) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_paths(v, root) for v in obj]
    return obj


@dataclass
class DRConfig:
    arch: str = "efficientnet_b3"
    img_size: int = 384
    pretrained: bool = True
    epochs: int = 25
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-4
    drop_rate: float = 0.3
    drop_path_rate: float = 0.2
    loss: str = "ce_sqrtw"
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    early_stop_patience: int = 8
    primary_fold: int = 0
    num_classes: int = 5
    class_names: list = field(default_factory=lambda: ["No DR", "Mild", "Moderate", "Severe", "Proliferative"])
    mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])
    clahe: bool = False


@dataclass
class QualityConfig:
    arch: str = "mobilenetv3_small_100"
    img_size: int = 320
    pretrained: bool = True
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    target_sensitivity: float = 0.95
    num_classes: int = 2
    class_names: list = field(default_factory=lambda: ["gradable", "ungradable"])
    mean: list = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class MCConfig:
    n_passes: int = 30
    enabled: bool = True


@dataclass
class ReferralConfig:
    target_rate: float = 0.15
    method: str = "entropy"


@dataclass
class XAIConfig:
    methods: list = field(default_factory=lambda: ["grad_cam_pp", "score_cam", "integrated_gradients"])
    default_method: str = "grad_cam_pp"
    target_layer: str = "blocks[-1]"
    ig_steps: int = 32
    ig_batch_size: int = 8
    ig_baseline: str = "black"
    blur_sigma: float = 4.0
    binarization_thresholds: list = field(default_factory=lambda: [0.25, 0.5, 0.75])
    binarization_fractions: list = field(default_factory=lambda: [0.02, 0.05, 0.10, 0.20])
    pointing_tolerance_px: int = 0


@dataclass
class DataConfig:
    data_root: str = "./data"
    smoke_root: str = "./smoke_data"
    metadata_dir: str = "./metadata"
    cache_dir: str = "./cache"
    artifacts_dir: str = "./artifacts"
    outputs_dir: str = "./outputs"
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class DeployConfig:
    onnx_opset: int = 17
    benchmark_runs: int = 50
    benchmark_warmup: int = 10
    threads: list = field(default_factory=lambda: [1, 4])


@dataclass
class Config:
    dr: DRConfig = field(default_factory=DRConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    mc: MCConfig = field(default_factory=MCConfig)
    referral: ReferralConfig = field(default_factory=ReferralConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    seed: int = 42
    device: str = "auto"


def load_config(config_path: Optional[str] = None, env_overrides: Optional[Dict] = None) -> Config:
    if config_path is None:
        config_path = os.environ.get("RETINAX_CONFIG", "configs/default.yaml")
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    if "_extends" in cfg_dict:
        base_path = config_path.parent / cfg_dict.pop("_extends")
        with open(base_path) as f:
            base_dict = yaml.safe_load(f)
        cfg_dict = _deep_merge(base_dict, cfg_dict)

    cfg_dict = _resolve_paths(cfg_dict, REPO_ROOT)

    if env_overrides is None:
        env_overrides = {}
    for key, value in env_overrides.items():
        parts = key.split(".")
        d = cfg_dict
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = type(d.get(parts[-1], value))(value)

    def _dict_to_dataclass(cls, d):
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in field_names}
        for k, v in filtered.items():
            field_type = cls.__dataclass_fields__[k].type
            if hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                filtered[k] = list(v)
        return cls(**filtered)

    return Config(
        dr=_dict_to_dataclass(DRConfig, cfg_dict.get("dr", {})),
        quality=_dict_to_dataclass(QualityConfig, cfg_dict.get("quality", {})),
        mc=_dict_to_dataclass(MCConfig, cfg_dict.get("mc", {})),
        referral=_dict_to_dataclass(ReferralConfig, cfg_dict.get("referral", {})),
        xai=_dict_to_dataclass(XAIConfig, cfg_dict.get("xai", {})),
        data=_dict_to_dataclass(DataConfig, cfg_dict.get("data", {})),
        deploy=_dict_to_dataclass(DeployConfig, cfg_dict.get("deploy", {})),
        seed=cfg_dict.get("seed", 42),
        device=cfg_dict.get("device", "auto"),
    )