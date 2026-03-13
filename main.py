"""
main.py — Benchmark complet : baselines sans RL vs agents entraînés.

Lance ce script unique pour :
  1. Évaluer TOUS les cas sans apprentissage (4 baselines)
  2. Entraîner les 3 agents RL (Q-Learning, DQN, PPO)
  3. Évaluer les agents entraînés
  4. Comparer tout le monde sur les mêmes métriques
  5. Générer un rapport complet (tableau + graphiques)

Usage :
    python main.py                         # pipeline complet
    python main.py --skip-training         # évalue uniquement (checkpoints existants)
    python main.py --agents dqn ppo        # entraîne seulement DQN + PPO
    python main.py --eval-episodes 5       # 5 épisodes d'évaluation par agent
    python main.py --train-episodes 200    # override num_episodes pour l'entraînement
    python main.py --quick                 # mode rapide : 50 épisodes, 2 éval
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import json
import yaml
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Assure que les imports relatifs fonctionnent ────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from env.sumo_env import SUMOEnvironment
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from training.trainer import Trainer


# ============================================================================
#  UTILITAIRES
# ============================================================================

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_sumo_cfg(config: dict) -> None:
    """Génère intersection.sumocfg si absent."""
    sim = config["simulation"]
    content = f"""<configuration>
    <input>
        <net-file value="{sim['net_file']}"/>
        <route-files value="{sim['route_file']}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{sim['total_steps']}"/>
    </time>
</configuration>"""
    with open(sim["sumo_cfg"], "w") as f:
        f.write(content)


def banner(title: str, width: int = 65) -> None:
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def section(title: str) -> None:
    print(f"\n  ── {title} ──")


# ============================================================================
#  BASELINES (cas sans RL)
# ============================================================================

class RandomBaseline:
    """Choisit une phase aléatoire à chaque décision."""

    def __init__(self, num_phases: int) -> None:
        self.num_phases = num_phases

    def get_greedy_action(self, obs) -> int:
        return random.randrange(self.num_phases)

    def select_action(self, obs) -> int:
        return self.get_greedy_action(obs)


class FixedTimeBaseline:
    """Round-robin fixe : alterne les phases toutes les `green_time` secondes."""

    def __init__(self, num_phases: int, green_time: int = 30) -> None:
        self.num_phases = num_phases
        self.green_time = green_time
        self._timer = 0
        self._phase = 0

    def get_greedy_action(self, obs) -> int:
        self._timer += 1
        if self._timer >= self.green_time:
            self._timer = 0
            self._phase = (self._phase + 1) % self.num_phases
        return self._phase

    def select_action(self, obs) -> int:
        return self.get_greedy_action(obs)

    def reset(self) -> None:
        self._timer = 0
        self._phase = 0


class MaxPressureBaseline:
    """
    Max-Pressure : à chaque décision, active la phase qui dessert
    les voies avec le plus de véhicules en attente.

    Heuristique classique sans apprentissage, souvent très compétitive.
    """

    # Voies servies par chaque phase (NS vs EW)
    NS_PHASES = {0, 2, 3, 6}   # phases qui servent principalement NS
    EW_PHASES = {1, 4, 5, 7}   # phases qui servent principalement EW

    def __init__(self, config: dict) -> None:
        self.lanes: List[str] = config["simulation"]["lanes"]
        self.num_phases: int = len(config["simulation"]["phases"])

    def get_greedy_action(self, obs) -> int:
        try:
            import traci
            existing = set(traci.lane.getIDList())

            ns_pressure = sum(
                traci.lane.getLastStepHaltingNumber(l)
                for l in self.lanes
                if l.startswith(("N_", "S_")) and l in existing
            )
            ew_pressure = sum(
                traci.lane.getLastStepHaltingNumber(l)
                for l in self.lanes
                if l.startswith(("E_", "W_")) and l in existing
            )
            # Choisit le groupe de phases le plus chargé
            if ns_pressure >= ew_pressure:
                return random.choice(list(self.NS_PHASES))
            else:
                return random.choice(list(self.EW_PHASES))
        except Exception:
            return 0

    def select_action(self, obs) -> int:
        return self.get_greedy_action(obs)


class ActuatedBaseline:
    """
    Feux actuated : comme FixedTime mais prolonge la phase verte
    si des véhicules sont encore détectés sur les voies actives.
    Extension max = max_green, extension min = min_green.
    """

    def __init__(self, config: dict, min_green: int = 15,
                 max_green: int = 50, extension: int = 5) -> None:
        sim = config["simulation"]
        self.lanes: List[str] = sim["lanes"]
        self.num_phases: int = len(sim["phases"])
        self.min_green = min_green
        self.max_green = max_green
        self.extension = extension
        self._phase = 0
        self._timer = 0

    def _has_demand(self) -> bool:
        try:
            import traci
            existing = set(traci.lane.getIDList())
            return any(
                traci.lane.getLastStepVehicleNumber(l) > 0
                for l in self.lanes if l in existing
            )
        except Exception:
            return False

    def get_greedy_action(self, obs) -> int:
        self._timer += 1
        # Force changement si max_green atteint
        if self._timer >= self.max_green:
            self._phase = (self._phase + 1) % self.num_phases
            self._timer = 0
        # Permet extension si min_green passé ET trafic présent
        elif self._timer >= self.min_green and not self._has_demand():
            self._phase = (self._phase + 1) % self.num_phases
            self._timer = 0
        return self._phase

    def select_action(self, obs) -> int:
        return self.get_greedy_action(obs)


# ============================================================================
#  RUNNER D'ÉPISODE GÉNÉRIQUE
# ============================================================================

def run_episode(agent, config: dict) -> Dict:
    """
    Exécute un épisode complet avec n'importe quel agent (baseline ou RL).
    Retourne les métriques de l'épisode.
    """
    env = SUMOEnvironment(config)
    obs = env.start()

    total_steps = config["simulation"]["total_steps"]
    ep_reward = ep_wait = ep_queue = ep_arrived = 0.0
    phase_counts: Dict[int, int] = {}
    step = 0

    while step < total_steps:
        action = agent.get_greedy_action(obs)
        phase_counts[action] = phase_counts.get(action, 0) + 1
        obs, reward, done, info = env.step(action)

        ep_reward += reward
        ep_wait   += info["total_waiting"]
        ep_queue  += info["total_queue"]
        ep_arrived += info["arrived"]
        step += 1
        if done:
            break

    env.close()
    n = max(step, 1)
    return {
        "reward":       ep_reward,
        "mean_waiting": ep_wait   / n,
        "mean_queue":   ep_queue  / n,
        "total_arrived": ep_arrived,
        "steps":        step,
        "phase_counts": phase_counts,
    }


def evaluate_agent(agent, config: dict, n_episodes: int,
                   label: str = "") -> Dict:
    """
    Exécute n_episodes et retourne les statistiques agrégées.
    """
    results = []
    for i in range(n_episodes):
        # Reset les baselines avec état interne entre épisodes
        if hasattr(agent, "reset"):
            agent.reset()
        r = run_episode(agent, config)
        results.append(r)
        print(f"    [{label}] épisode {i+1}/{n_episodes} | "
              f"reward={r['reward']:>8.1f} | "
              f"wait={r['mean_waiting']:>6.1f}s | "
              f"arrivés={r['total_arrived']:>4.0f}")

    keys = [k for k in results[0] if k != "phase_counts"]
    agg: Dict = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"]  = float(np.std(vals))

    # Fusionne les phase_counts
    merged: Dict[int, int] = {}
    for r in results:
        for ph, cnt in r["phase_counts"].items():
            merged[ph] = merged.get(ph, 0) + cnt
    agg["phase_counts"] = merged

    return agg


# ============================================================================
#  ENTRAÎNEMENT RL
# ============================================================================

def train_agent(agent_type: str, config: dict) -> Optional[str]:
    """
    Lance l'entraînement et retourne le chemin du meilleur checkpoint.
    Retourne None si l'entraînement échoue.
    """
    banner(f"Entraînement : {agent_type.upper()}")
    try:
        trainer = Trainer(config, agent_type=agent_type)
        trainer.train()
        ext = ".json" if agent_type == "qlearning" else ".pt"
        ckpt = os.path.join(config["paths"]["models_dir"],
                            f"{agent_type}_best{ext}")
        return ckpt if os.path.exists(ckpt) else None
    except Exception as e:
        print(f"  [ERREUR] Entraînement {agent_type} échoué : {e}")
        return None


def load_trained_agent(agent_type: str, checkpoint: str,
                       config: dict):
    """Charge un agent entraîné depuis un checkpoint."""
    env = SUMOEnvironment(config)
    if agent_type == "qlearning":
        agent = QLearningAgent(config)
    elif agent_type == "dqn":
        agent = DQNAgent(env.obs_dim, env.action_dim, config)
    elif agent_type == "ppo":
        agent = PPOAgent(env.obs_dim, env.action_dim, config)
    else:
        raise ValueError(agent_type)
    agent.load(checkpoint)
    return agent


# ============================================================================
#  RAPPORT ET VISUALISATIONS
# ============================================================================

METRIC_INFO = {
    "reward_mean":        ("Récompense moyenne",        "↑ meilleur", "#4a90d9"),
    "mean_waiting_mean":  ("Temps d'attente moyen (s)", "↓ meilleur", "#e07b54"),
    "mean_queue_mean":    ("File d'attente moyenne",    "↓ meilleur", "#e0a030"),
    "total_arrived_mean": ("Véhicules arrivés",         "↑ meilleur", "#5bab6e"),
}


def _color_for(name: str) -> str:
    """Couleur de barre selon le type d'agent."""
    if "Fixe" in name or "Aléatoire" in name:
        return "#aaaaaa"
    if "MaxPressure" in name or "Actuated" in name:
        return "#7eb0d4"
    if "QLearning" in name:
        return "#fd7f6f"
    if "DQN" in name:
        return "#b2e061"
    if "PPO" in name:
        return "#bd7ebe"
    return "#888888"


def plot_comparison(df: pd.DataFrame, out_dir: str) -> str:
    """
    Génère le graphique de comparaison 2×2 (4 métriques).
    Retourne le chemin du fichier PNG.
    """
    agents = df.index.tolist()
    colors = [_color_for(a) for a in agents]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#f8f8f8")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    for idx, (col, (title, direction, _)) in enumerate(METRIC_INFO.items()):
        if col not in df.columns:
            continue
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor("white")

        values = df[col].values
        err_col = col.replace("_mean", "_std")
        errors = df[err_col].values if err_col in df.columns else None

        bars = ax.bar(range(len(agents)), values, color=colors,
                      yerr=errors, capsize=5, edgecolor="white",
                      linewidth=0.8, error_kw={"elinewidth": 1.5,
                                                "ecolor": "#555555"})

        # Valeur au-dessus de chaque barre
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(values) * 0.015),
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color="#333333")

        # Mise en valeur du meilleur
        if "↑" in direction:
            best_idx = int(np.argmax(values))
        else:
            best_idx = int(np.argmin(values))
        bars[best_idx].set_edgecolor("#222222")
        bars[best_idx].set_linewidth(2)
        ax.get_xticklabels()  # force rendering

        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(agents, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{title}\n({direction})", fontweight="bold",
                     fontsize=11, pad=10)
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Comparaison complète : Baselines vs Agents RL",
                 fontsize=16, fontweight="bold", y=0.98)

    out = os.path.join(out_dir, "full_comparison.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    return out


def plot_radar(df: pd.DataFrame, out_dir: str) -> str:
    """
    Graphique radar (araignée) : normalise chaque métrique pour
    mettre tous les agents sur le même référentiel 0–1.
    """
    metrics = [c for c in METRIC_INFO if c in df.columns]
    labels  = [METRIC_INFO[m][0] for m in metrics]
    N = len(metrics)

    # Normalisation 0–1 (meilleur = 1)
    norm_df = df[metrics].copy()
    for col in metrics:
        vmin, vmax = norm_df[col].min(), norm_df[col].max()
        if vmax == vmin:
            norm_df[col] = 0.5
        else:
            normalized = (norm_df[col] - vmin) / (vmax - vmin)
            # Pour les métriques "↓ meilleur" on inverse
            if "↓" in METRIC_INFO[col][1]:
                normalized = 1.0 - normalized
            norm_df[col] = normalized

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9),
                           subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#f8f8f8")
    ax.set_facecolor("white")

    for agent in df.index:
        values = norm_df.loc[agent].values.tolist()
        values += values[:1]
        color = _color_for(agent)
        ax.plot(angles, values, "o-", linewidth=2,
                label=agent, color=color, markersize=5)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=8)
    ax.set_title("Performance normalisée (100% = meilleur agent)",
                 pad=20, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=9, framealpha=0.9)

    out = os.path.join(out_dir, "radar_comparison.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out


def plot_phase_distribution(phase_data: Dict[str, Dict[int, int]],
                            out_dir: str) -> str:
    """Distribution des phases sélectionnées par chaque agent."""
    agents = list(phase_data.keys())
    all_phases = sorted({p for pc in phase_data.values() for p in pc})

    fig, axes = plt.subplots(1, len(agents),
                             figsize=(4 * len(agents), 4), sharey=False)
    if len(agents) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#f8f8f8")

    for ax, agent in zip(axes, agents):
        ax.set_facecolor("white")
        pc = phase_data[agent]
        counts = [pc.get(p, 0) for p in all_phases]
        total  = sum(counts) or 1
        pcts   = [100 * c / total for c in counts]
        color  = _color_for(agent)
        ax.bar(all_phases, pcts, color=color, edgecolor="white")
        ax.set_title(agent, fontsize=10, fontweight="bold")
        ax.set_xlabel("Phase")
        ax.set_ylabel("% sélection" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Distribution des phases sélectionnées",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(out_dir, "phase_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def print_report(df: pd.DataFrame, baseline_names: List[str]) -> None:
    """Affiche le tableau de résultats dans le terminal."""
    banner("RAPPORT FINAL")

    cols_display = {
        "reward_mean":        "Récompense",
        "mean_waiting_mean":  "Attente moy. (s)",
        "mean_queue_mean":    "File moy.",
        "total_arrived_mean": "Arrivés",
    }
    display = df[[c for c in cols_display if c in df.columns]].copy()
    display.columns = [cols_display[c] for c in display.columns]

    print("\n" + display.to_string(float_format="{:.2f}".format))

    # Meilleur agent par métrique
    print("\n  Meilleur par métrique :")
    for col, label in cols_display.items():
        if col not in df.columns:
            continue
        if "Récompense" in label or "Arrivés" in label:
            best = df[col].idxmax()
        else:
            best = df[col].idxmin()
        print(f"    {label:<22} → {best}")

    # Gains vs baseline fixe
    ref = "Fixe-30s"
    if ref in df.index:
        print(f"\n  Gains vs {ref} (baseline de référence) :")
        for col, label in cols_display.items():
            if col not in df.columns:
                continue
            ref_val = df.loc[ref, col]
            for agent in df.index:
                if agent == ref or agent in baseline_names:
                    continue
                val  = df.loc[agent, col]
                diff = val - ref_val
                if ref_val != 0:
                    pct = 100 * diff / abs(ref_val)
                else:
                    pct = 0.0
                sign = "+" if diff >= 0 else ""
                print(f"    [{agent}] {label:<22} "
                      f"{sign}{diff:>8.2f}  ({sign}{pct:.1f}%)")


def save_json_report(results: Dict, out_dir: str) -> str:
    """Sauvegarde les résultats bruts en JSON."""
    clean = {}
    for agent, agg in results.items():
        clean[agent] = {k: v for k, v in agg.items() if k != "phase_counts"}
    path = os.path.join(out_dir, "benchmark_results.json")
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    return path


# ============================================================================
#  PIPELINE PRINCIPAL
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark complet Traffic RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",          default="configs/config.yaml")
    parser.add_argument("--agents",          nargs="+",
                        default=["qlearning", "dqn", "ppo"],
                        choices=["qlearning", "dqn", "ppo"])
    parser.add_argument("--skip-training",   action="store_true",
                        help="Ne pas entraîner, utiliser les checkpoints existants")
    parser.add_argument("--eval-episodes",   type=int, default=5,
                        help="Nombre d'épisodes d'évaluation par agent (défaut: 5)")
    parser.add_argument("--train-episodes",  type=int, default=None,
                        help="Override du nombre d'épisodes d'entraînement")
    parser.add_argument("--quick",           action="store_true",
                        help="Mode rapide : 50 épisodes entraînement, 2 éval")
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    config = load_config(args.config)
    ensure_sumo_cfg(config)

    if args.quick:
        config["simulation"]["num_episodes"] = 50
        args.eval_episodes = 2
        print("  [Mode rapide] 50 épisodes, 2 évaluations par agent")

    if args.train_episodes:
        config["simulation"]["num_episodes"] = args.train_episodes

    plots_dir  = config["paths"]["plots_dir"]
    models_dir = config["paths"]["models_dir"]
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n  Run ID : {run_id}")
    print(f"  Agents RL : {args.agents}")
    print(f"  Épisodes évaluation : {args.eval_episodes}")
    print(f"  Épisodes entraînement : {config['simulation']['num_episodes']}")

    num_phases = len(config["simulation"]["phases"])
    all_results: Dict[str, Dict] = {}
    baseline_names: List[str] = []

    # ════════════════════════════════════════════════════════════════════════
    #  PHASE 1 — BASELINES (sans RL)
    # ════════════════════════════════════════════════════════════════════════
    banner("PHASE 1 : Baselines sans apprentissage")

    baselines = [
        ("Aléatoire",    RandomBaseline(num_phases)),
        ("Fixe-15s",     FixedTimeBaseline(num_phases, green_time=15)),
        ("Fixe-30s",     FixedTimeBaseline(num_phases, green_time=30)),
        ("Fixe-60s",     FixedTimeBaseline(num_phases, green_time=60)),
        ("MaxPressure",  MaxPressureBaseline(config)),
        ("Actuated",     ActuatedBaseline(config)),
    ]

    for name, agent in baselines:
        section(f"Baseline : {name}")
        agg = evaluate_agent(agent, config, args.eval_episodes, label=name)
        all_results[name] = agg
        baseline_names.append(name)
        print(f"    → Récompense moy. = {agg['reward_mean']:.2f} "
              f"| Attente = {agg['mean_waiting_mean']:.1f}s "
              f"| Arrivés = {agg['total_arrived_mean']:.0f}")

    # ════════════════════════════════════════════════════════════════════════
    #  PHASE 2 — ENTRAÎNEMENT RL
    # ════════════════════════════════════════════════════════════════════════
    trained_checkpoints: Dict[str, str] = {}

    if not args.skip_training:
        banner("PHASE 2 : Entraînement des agents RL")
        for agent_type in args.agents:
            ckpt = train_agent(agent_type, config)
            if ckpt:
                trained_checkpoints[agent_type] = ckpt
                print(f"  ✓ {agent_type.upper()} → {ckpt}")
            else:
                print(f"  ✗ {agent_type.upper()} : entraînement échoué ou SUMO absent")
    else:
        banner("PHASE 2 : Chargement des checkpoints existants")
        for agent_type in args.agents:
            ext  = ".json" if agent_type == "qlearning" else ".pt"
            ckpt = os.path.join(models_dir, f"{agent_type}_best{ext}")
            if os.path.exists(ckpt):
                trained_checkpoints[agent_type] = ckpt
                print(f"  ✓ {agent_type.upper()} → {ckpt}")
            else:
                print(f"  ✗ {agent_type.upper()} : checkpoint introuvable ({ckpt})")

    # ════════════════════════════════════════════════════════════════════════
    #  PHASE 3 — ÉVALUATION DES AGENTS RL
    # ════════════════════════════════════════════════════════════════════════
    if trained_checkpoints:
        banner("PHASE 3 : Évaluation des agents RL entraînés")

        rl_label_map = {
            "qlearning": "QLearning",
            "dqn":       "DQN",
            "ppo":       "PPO",
        }

        for agent_type, ckpt in trained_checkpoints.items():
            label = rl_label_map.get(agent_type, agent_type.upper())
            section(f"Agent : {label}")
            try:
                agent = load_trained_agent(agent_type, ckpt, config)
                agg = evaluate_agent(agent, config, args.eval_episodes,
                                     label=label)
                all_results[label] = agg
                print(f"    → Récompense moy. = {agg['reward_mean']:.2f} "
                      f"| Attente = {agg['mean_waiting_mean']:.1f}s "
                      f"| Arrivés = {agg['total_arrived_mean']:.0f}")
            except Exception as e:
                print(f"    [ERREUR] Évaluation {label} : {e}")
    else:
        print("\n  Aucun agent RL entraîné — comparaison baselines uniquement.")

    # ════════════════════════════════════════════════════════════════════════
    #  PHASE 4 — RAPPORT ET GRAPHIQUES
    # ════════════════════════════════════════════════════════════════════════
    banner("PHASE 4 : Rapport et visualisations")

    # Construire le DataFrame
    metric_cols = ["reward_mean", "reward_std",
                   "mean_waiting_mean", "mean_waiting_std",
                   "mean_queue_mean", "mean_queue_std",
                   "total_arrived_mean", "total_arrived_std"]

    rows = []
    for agent_name, agg in all_results.items():
        row = {"agent": agent_name}
        for col in metric_cols:
            row[col] = agg.get(col, float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows).set_index("agent")

    # Rapport terminal
    print_report(df, baseline_names)

    # Graphiques
    section("Génération des graphiques")

    paths_generated = []

    p = plot_comparison(df, plots_dir)
    paths_generated.append(p)
    print(f"  ✓ Comparaison barres        → {p}")

    p = plot_radar(df, plots_dir)
    paths_generated.append(p)
    print(f"  ✓ Radar normalisé           → {p}")

    # Distribution des phases (seulement agents qui ont un phase_counts)
    phase_data = {
        name: agg["phase_counts"]
        for name, agg in all_results.items()
        if agg.get("phase_counts")
    }
    if phase_data:
        p = plot_phase_distribution(phase_data, plots_dir)
        paths_generated.append(p)
        print(f"  ✓ Distribution des phases  → {p}")

    # JSON brut
    json_path = save_json_report(all_results, plots_dir)
    print(f"  ✓ Résultats JSON            → {json_path}")

    # CSV final
    csv_path = os.path.join(plots_dir, "benchmark_results.csv")
    df.to_csv(csv_path, float_format="%.4f")
    print(f"  ✓ Tableau CSV               → {csv_path}")

    # ════════════════════════════════════════════════════════════════════════
    #  RÉSUMÉ FINAL
    # ════════════════════════════════════════════════════════════════════════
    banner("RÉSUMÉ EXÉCUTIF")

    # Classement global par récompense moyenne
    if "reward_mean" in df.columns:
        ranked = df["reward_mean"].sort_values(ascending=False)
        print("\n  Classement par récompense :")
        for rank, (agent, val) in enumerate(ranked.items(), 1):
            tag = " ← meilleur" if rank == 1 else ""
            is_rl = agent not in baseline_names
            kind  = "[RL]  " if is_rl else "[BASE]"
            print(f"    {rank}. {kind} {agent:<18} {val:>10.2f}{tag}")

    # Meilleur RL vs meilleure baseline
    rl_agents = [a for a in df.index if a not in baseline_names]
    if rl_agents and "reward_mean" in df.columns:
        best_rl       = df.loc[rl_agents, "reward_mean"].idxmax()
        best_rl_val   = df.loc[best_rl, "reward_mean"]
        best_base     = df.loc[baseline_names, "reward_mean"].idxmax()
        best_base_val = df.loc[best_base, "reward_mean"]
        gain          = best_rl_val - best_base_val
        pct           = 100 * gain / abs(best_base_val) if best_base_val else 0

        print(f"\n  Meilleur agent RL   : {best_rl} ({best_rl_val:.2f})")
        print(f"  Meilleure baseline  : {best_base} ({best_base_val:.2f})")
        sign = "+" if gain >= 0 else ""
        print(f"  Gain RL vs baseline : {sign}{gain:.2f} ({sign}{pct:.1f}%)")

    print(f"\n  Fichiers générés dans : {os.path.abspath(plots_dir)}/")
    for p in paths_generated:
        print(f"    • {os.path.basename(p)}")
    print()


# ============================================================================
if __name__ == "__main__":
    t_start = time.time()
    main()
    elapsed = time.time() - t_start
    m, s = divmod(int(elapsed), 60)
    print(f"  Durée totale : {m}m {s}s\n")
