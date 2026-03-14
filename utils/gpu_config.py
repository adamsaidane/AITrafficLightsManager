"""
gpu_config.py — Gestion centralisée du GPU pour tout le projet.

Priorité de sélection :
    1. CUDA (NVIDIA GPU)
    2. MPS  (Apple Silicon GPU)
    3. CPU  (fallback uniquement — jamais utilisé si GPU disponible)

Usage :
    from utils.gpu_config import DEVICE, gpu_info, assert_gpu

    tensor = tensor.to(DEVICE)   # envoie vers le bon device automatiquement
"""

from __future__ import annotations
import torch
import os


def _select_device() -> torch.device:
    """
    Sélectionne le meilleur device disponible.
    Lance un avertissement si aucun GPU n'est trouvé.
    """
    # Permet de forcer CPU via variable d'env (debug uniquement)
    if os.environ.get("FORCE_CPU", "0") == "1":
        print("[GPU] FORCE_CPU=1 — utilisation du CPU forcée")
        return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name  = torch.cuda.get_device_name(0)
        mem   = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] CUDA disponible — {name} ({mem:.1f} GB VRAM)")
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[GPU] Apple MPS disponible — GPU Apple Silicon activé")
        return torch.device("mps")

    print("[GPU] AVERTISSEMENT : Aucun GPU détecté — utilisation du CPU")
    print("       Pour forcer le CPU en debug : FORCE_CPU=1 python main.py")
    return torch.device("cpu")


# Device global partagé par tous les agents
DEVICE: torch.device = _select_device()


def gpu_info() -> dict:
    """Retourne un dictionnaire d'informations sur le GPU actif."""
    info = {"device": str(DEVICE), "type": DEVICE.type}
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update({
            "name":          props.name,
            "vram_gb":       round(props.total_memory / 1024**3, 2),
            "cuda_version":  torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "multi_gpu":     torch.cuda.device_count() > 1,
            "gpu_count":     torch.cuda.device_count(),
        })
    return info


def assert_gpu() -> None:
    """Lève une erreur si pas de GPU — utile pour lancer l'entraînement."""
    if DEVICE.type == "cpu":
        raise RuntimeError(
            "Aucun GPU détecté. Installez CUDA (NVIDIA) ou utilisez Apple MPS.\n"
            "Pour ignorer cette erreur : FORCE_CPU=1 python main.py"
        )


def move_batch(batch: tuple) -> tuple:
    """
    Déplace un batch de tenseurs (tuple) vers DEVICE.
    Pratique pour déplacer (states, actions, rewards, ...) d'un coup.
    """
    return tuple(
        t.to(DEVICE) if isinstance(t, torch.Tensor) else t
        for t in batch
    )


def benchmark_gpu(size: int = 2048) -> float:
    """
    Mini-benchmark : multiplie deux matrices sur GPU et retourne le temps (ms).
    Permet de vérifier que le GPU est bien utilisé.
    """
    import time
    a = torch.randn(size, size, device=DEVICE)
    b = torch.randn(size, size, device=DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = a @ b
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[GPU] Benchmark {size}×{size} matmul : {elapsed_ms:.2f} ms sur {DEVICE}")
    return elapsed_ms
