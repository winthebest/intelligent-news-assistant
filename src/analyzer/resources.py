from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# "min VRAM or RAM (GB)" -> recommended Qwen checkpoint. Empirical: Ollama
# Q4 needs ~1.4x model size in RAM plus ~2GB for KV-cache at num_ctx=8192.
_QWEN_LADDER: list[tuple[float, str]] = [
    (24.0, "ollama/qwen2.5:32b-instruct"),
    (11.0, "ollama/qwen2.5:14b-instruct"),
    (6.5, "ollama/qwen2.5:7b-instruct"),
    (3.0, "ollama/qwen2.5:3b-instruct"),
    (0.0, "ollama/qwen2.5:1.5b-instruct"),
]


@dataclass
class HostCapacity:
    """Coarse snapshot of what the host can afford for local LLM."""

    ram_gb: Optional[float]
    gpu_vram_gb: Optional[float]
    gpu_name: Optional[str]

    @property
    def effective_gb(self) -> float:
        """Best available memory budget: GPU VRAM if present, else 70% RAM."""
        if self.gpu_vram_gb is not None and self.gpu_vram_gb > 0:
            return self.gpu_vram_gb
        if self.ram_gb is not None:
            return self.ram_gb * 0.7
        return 4.0  # conservative default when probing fails

    def recommend_ollama_model(self) -> str:
        """Biggest Qwen2.5 checkpoint that fits ``effective_gb``."""
        budget = self.effective_gb
        for threshold, model in _QWEN_LADDER:
            if budget >= threshold:
                return model
        return _QWEN_LADDER[-1][1]


def _detect_ram_gb() -> Optional[float]:
    try:
        import psutil  # noqa: WPS433

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    # psutil missing: try os.sysconf on *nix; unavailable on Windows.
    import os

    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return (pages * page_size) / (1024**3)
    except (ValueError, OSError):
        pass
    return None


def _detect_gpu() -> tuple[Optional[float], Optional[str]]:
    """Return (total_vram_gb, gpu_name) or (None, None) if no NVIDIA GPU."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None, None
    try:
        out = subprocess.check_output(
            [
                smi,
                "--query-gpu=memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            timeout=2.0,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("nvidia-smi probe failed: %s", e)
        return None, None

    # Pick the first GPU: Ollama only schedules on one today.
    first_line = out.strip().splitlines()[0] if out.strip() else ""
    if not first_line:
        return None, None
    parts = [p.strip() for p in first_line.split(",")]
    try:
        vram_mb = float(parts[0])
        name = parts[1] if len(parts) > 1 else "NVIDIA GPU"
    except (ValueError, IndexError):
        return None, None
    return vram_mb / 1024.0, name


def probe_host() -> HostCapacity:
    """One-shot probe of RAM + GPU. Safe to call at import time."""
    ram = _detect_ram_gb()
    vram, gpu_name = _detect_gpu()
    cap = HostCapacity(ram_gb=ram, gpu_vram_gb=vram, gpu_name=gpu_name)
    logger.info(
        "Host capacity: RAM=%s GB, GPU=%s (%s GB VRAM) -> effective=%.1f GB",
        f"{ram:.1f}" if ram else "unknown",
        gpu_name or "none",
        f"{vram:.1f}" if vram else "0",
        cap.effective_gb,
    )
    return cap
