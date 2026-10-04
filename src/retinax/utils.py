import os
import random
import json
import hashlib
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def get_versions() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def run_info(config: Any, extra: Optional[Dict] = None) -> Dict[str, Any]:
    info = {
        "seed": getattr(config, "seed", 42),
        "versions": get_versions(),
        "git_hash": get_git_hash(),
        "platform": platform.platform(),
        "device": str(get_device(getattr(config, "device", "auto"))),
    }
    if extra:
        info.update(extra)
    return info


def write_run_info(output_dir: Path, info: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)


def setup_logging(name: str = "retinax", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def checksum_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


REPO_ROOT = Path(__file__).resolve().parents[2]