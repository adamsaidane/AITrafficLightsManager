"""
patch_traffic.py — Réduit la densité du trafic dans le fichier .rou.xml SUMO.

Pourquoi c'est nécessaire :
    Le flux actuel (~2700 veh/heure) est proche de la saturation physique
    de l'intersection. En zone saturée, toutes les actions RL donnent un
    résultat similaire (mauvais) → signal de récompense non-informatif
    → l'agent n'apprend pas.

    La littérature RL trafic recommande de démarrer entre 40–60% de la
    capacité maximale, puis d'augmenter progressivement.

Curriculum d'entraînement recommandé :
    Étape 1 : --density 0.4   (40% capacité)  →  500 épisodes
    Étape 2 : --density 0.6   (60%)           →  500 épisodes
    Étape 3 : --density 0.8   (80%)           →  500 épisodes
    Étape 4 : --density 1.0   (100% = actuel) →  évaluation finale

Usage :
    python patch_traffic.py --rou "SUMO intersection/intersection.rou.xml" --density 0.5
    python patch_traffic.py --rou "SUMO intersection/intersection.rou.xml" --density 0.5 --preview
"""

from __future__ import annotations
import argparse
import re
import shutil
import os
from pathlib import Path


def patch_veh_per_hour(content: str, factor: float) -> tuple[str, dict]:
    """
    Multiplie tous les attributs vehsPerHour / vehs-per-hour / frequency
    par `factor`. Retourne le contenu modifié + statistiques.
    """
    stats = {"modified": 0, "original": [], "new": []}

    # Motifs couvrant les variantes de nommage SUMO
    patterns = [
        (r'vehsPerHour="([\d.]+)"',  'vehsPerHour="{}"'),
        (r'vehs-per-hour="([\d.]+)"', 'vehs-per-hour="{}"'),
        (r'frequency="([\d.]+)"',     'frequency="{}"'),
        (r'period="([\d.]+)"',        'period="{}"'),   # period = 1/freq, inverse factor
    ]

    result = content
    for pattern, template in patterns:
        is_period = "period" in pattern

        def replacer(m, _factor=factor, _tmpl=template, _inv=is_period):
            orig = float(m.group(1))
            # period est l'inverse du débit : réduire débit = augmenter period
            new_val = orig / _factor if _inv else orig * _factor
            stats["original"].append(orig)
            stats["new"].append(new_val)
            stats["modified"] += 1
            return _tmpl.format(f"{new_val:.1f}")

        result = re.sub(pattern, replacer, result)

    return result, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Réduit la densité de trafic dans un fichier .rou.xml SUMO"
    )
    parser.add_argument("--rou",     required=True, help="Chemin vers le fichier .rou.xml")
    parser.add_argument("--density", type=float, default=0.5,
                        help="Facteur de densité (0.5 = 50% du flux original)")
    parser.add_argument("--preview", action="store_true",
                        help="Affiche le résultat sans écrire le fichier")
    parser.add_argument("--output",  default=None,
                        help="Fichier de sortie (défaut: écrase l'original)")
    args = parser.parse_args()

    rou_path = Path(args.rou)
    if not rou_path.exists():
        print(f"ERREUR : fichier introuvable : {rou_path}")
        return

    # Backup automatique
    backup = rou_path.with_suffix(".rou.xml.bak")
    if not backup.exists():
        shutil.copy2(rou_path, backup)
        print(f"  Backup créé : {backup}")

    content = rou_path.read_text(encoding="utf-8")
    patched, stats = patch_veh_per_hour(content, args.density)

    if stats["modified"] == 0:
        print(f"  AVERTISSEMENT : aucun attribut de flux trouvé dans {rou_path}")
        print("  Attributs recherchés : vehsPerHour, vehs-per-hour, frequency, period")
        return

    print(f"\n  Fichier : {rou_path}")
    print(f"  Facteur : {args.density:.0%}")
    print(f"  Attributs modifiés : {stats['modified']}")
    for orig, new in zip(stats["original"], stats["new"]):
        print(f"    {orig:.0f} veh/h  →  {new:.0f} veh/h")

    if args.preview:
        print("\n  [MODE PREVIEW — fichier non modifié]")
        print("  Retirez --preview pour appliquer les modifications.")
        return

    out_path = Path(args.output) if args.output else rou_path
    out_path.write_text(patched, encoding="utf-8")
    print(f"\n  Fichier mis à jour : {out_path}")
    print(f"  Pour restaurer l'original : copy {backup} {rou_path}")


# ── Curriculum helper ──────────────────────────────────────────────────────

def curriculum_steps(rou_path: str, stages: list[tuple[float, int]]) -> None:
    """
    Affiche les commandes à exécuter pour un entraînement curriculaire.

    stages = [(density, episodes), ...]
    Exemple : [(0.4, 500), (0.6, 500), (0.8, 500), (1.0, 0)]
    """
    print("\n  Curriculum d'entraînement recommandé :")
    print("  " + "─" * 55)
    for density, episodes in stages:
        label = f"{density:.0%} capacité"
        if episodes > 0:
            print(f"  1. python patch_traffic.py --rou \"{rou_path}\" --density {density}")
            print(f"     python main.py --agents dqn ppo --train-episodes {episodes} --skip-phase1")
        else:
            print(f"  {len(stages)}. python patch_traffic.py --rou \"{rou_path}\" --density {density}")
            print(f"     python main.py --skip-training --eval-episodes 20")
        print()


if __name__ == "__main__":
    main()

    # Afficher le curriculum si lancé sans arguments avancés
    import sys
    if "--rou" in sys.argv:
        rou = sys.argv[sys.argv.index("--rou") + 1]
        print()
        curriculum_steps(rou, [(0.4, 500), (0.6, 500), (0.8, 500), (1.0, 0)])
