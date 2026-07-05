"""Coleta informações de hardware/ambiente para registrar em cada run.json."""
from __future__ import annotations

import os
import platform
import subprocess
from typing import Any, Dict


def get_hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    try:
        import torch  # type: ignore

        info["torch_available"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_available"] = False
        info["cuda_available"] = False

    return info


def get_code_version() -> Dict[str, Any]:
    """Tenta capturar o commit git atual, sem quebrar se não houver repositório."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ) != 0
        return {"git_commit": commit, "dirty": dirty}
    except Exception:
        return {"git_commit": None, "dirty": None}
