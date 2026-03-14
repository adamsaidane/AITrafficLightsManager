"""
patch_traffic.py — Ajuste la densite du trafic dans intersection.rou.xml

Syntaxe SUMO detectee automatiquement :
    probability="0.08"   → multiplie par factor
    vehsPerHour="720"    → multiplie par factor
    period="12.5"        → divise par factor (inverse)
    frequency="0.08"     → multiplie par factor

Usage :
    python patch_traffic.py --rou "SUMO intersection/intersection.rou.xml" --density 0.5
    python patch_traffic.py --rou "SUMO intersection/intersection.rou.xml" --density 0.5 --preview

Curriculum recommande :
    Etape 1 : --density 0.4  (apprentissage des bases)
    Etape 2 : --density 0.6  (consolidation)
    Etape 3 : --density 0.8  (proche du reel)
    Etape 4 : --density 1.0  (evaluation finale)
"""

from __future__ import annotations
import argparse
import re
import shutil
import glob
from pathlib import Path


# ------------------------------------------------------------------
# Tous les attributs SUMO qui controlent le debit, avec leur sens
# ------------------------------------------------------------------
PATTERNS = [
    # (regex,               template,              inverse?)
    (r'probability="([\d.]+)"',  'probability="{}"',  False),
    (r'vehsPerHour="([\d.]+)"',  'vehsPerHour="{}"',  False),
    (r'vehs-per-hour="([\d.]+)"','vehs-per-hour="{}"', False),
    (r'frequency="([\d.]+)"',    'frequency="{}"',    False),
    (r'period="([\d.]+)"',       'period="{}"',       True),   # inverse
    (r'number="(\d+)"',          'number="{}"',       False),  # flow total count
]


def patch_density(content: str, factor: float) -> tuple[str, list]:
    """Applique le facteur a tous les attributs de flux trouves."""
    changes = []
    result = content

    for pattern, template, inverse in PATTERNS:
        matches = re.findall(pattern, result)
        if not matches:
            continue

        def make_replacer(tmpl, inv, fac):
            def replacer(m):
                orig = float(m.group(1))
                new  = orig / fac if inv else orig * fac
                # Conserver le format : entier si >= 1, sinon 3 decimales
                fmt  = f"{new:.0f}" if new >= 10 else f"{new:.4f}"
                changes.append((tmpl.split('=')[0], orig, new))
                return tmpl.format(fmt)
            return replacer

        result = re.sub(pattern, make_replacer(template, inverse, factor), result)

    return result, changes


def show_curriculum(rou: str) -> None:
    """Affiche les commandes du curriculum d'entrainement."""
    print("\n  Curriculum d'entrainement recommande :")
    print("  " + "-" * 56)
    steps = [
        (0.4, 500,  "debut — apprentissage des bases"),
        (0.6, 500,  "intermediaire — consolidation"),
        (0.8, 500,  "avance — proche du reel"),
        (1.0, None, "evaluation finale — flux complet"),
    ]
    for i, (d, ep, label) in enumerate(steps, 1):
        print(f"\n  Etape {i} — {label}")
        print(f'    python patch_traffic.py --rou "{rou}" --density {d}')
        if ep:
            print(f'    python main.py --agents dqn ppo --train-episodes {ep} --skip-phase1')
        else:
            print(f'    python main.py --skip-training --eval-episodes 20')
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Ajuste la densite de trafic SUMO")
    p.add_argument("--rou",     required=True)
    p.add_argument("--density", type=float, default=0.5,
                   help="Facteur 0.0-1.0 (0.5 = 50%% du flux original)")
    p.add_argument("--preview", action="store_true",
                   help="Affiche sans modifier le fichier")
    p.add_argument("--output",  default=None)
    args = p.parse_args()

    # --- Validation du chemin ---
    if str(args.rou).strip(".'\" ") in ("", "...", "path/to/file"):
        print(f'\n  ERREUR : chemin invalide : "{args.rou}"')
        found = glob.glob("**/*.rou.xml", recursive=True)
        if found:
            print("\n  Fichiers .rou.xml trouves :")
            for f in found:
                print(f"    {f}")
            print(f'\n  Essaie : python patch_traffic.py --rou "{found[0]}" --density {args.density}')
        return

    rou_path = Path(args.rou)
    if not rou_path.exists():
        print(f'\n  ERREUR : fichier introuvable : {rou_path.resolve()}')
        found = glob.glob("**/*.rou.xml", recursive=True)
        if found:
            print("\n  Fichiers .rou.xml trouves dans ce dossier :")
            for f in found:
                print(f"    {f}")
            print(f'\n  Essaie : python patch_traffic.py --rou "{found[0]}" --density {args.density}')
        else:
            print("  Aucun .rou.xml trouve. Lance ce script depuis le dossier TrafficManager.")
        return

    if not 0.0 < args.density <= 2.0:
        print(f"  ERREUR : --density doit etre entre 0.0 et 2.0 (recu : {args.density})")
        return

    # --- Lecture ---
    content = rou_path.read_text(encoding="utf-8")

    # --- Patch ---
    patched, changes = patch_density(content, args.density)

    # --- Rapport ---
    print(f"\n  Fichier  : {rou_path}")
    print(f"  Facteur  : {args.density:.0%}  ({args.density})")

    if not changes:
        print("\n  AVERTISSEMENT : aucun attribut de flux trouve.")
        print("  Attributs recherches : probability, vehsPerHour, period, frequency, number")
        print("\n  Contenu du fichier (10 premieres lignes) :")
        for line in content.splitlines()[:10]:
            print(f"    {line}")
        return

    print(f"\n  {len(changes)} attribut(s) modifie(s) :")
    seen = set()
    for attr, orig, new in changes:
        key = (attr, orig)
        if key not in seen:
            seen.add(key)
            arrow = "/" if "period" in attr else "*"
            print(f"    {attr:<20} {orig:.4f}  →  {new:.4f}   ({arrow}{args.density})")

    # Estimation du flux resultant
    prob_changes = [(o, n) for a, o, n in changes if "prob" in a]
    if prob_changes:
        total_prob_new = sum(n for _, n in prob_changes)
        total_prob_old = sum(o for o, _ in prob_changes)
        veh_h_old = total_prob_old * 3600
        veh_h_new = total_prob_new * 3600
        print(f"\n  Flux total estime :")
        print(f"    Avant  : {veh_h_old:.0f} veh/h")
        print(f"    Apres  : {veh_h_new:.0f} veh/h")

    if args.preview:
        print("\n  [PREVIEW — fichier non modifie]")
        print('  Retire --preview pour appliquer.')
        show_curriculum(args.rou)
        return

    # --- Backup automatique ---
    backup = rou_path.with_suffix(".bak")
    if not backup.exists():
        shutil.copy2(rou_path, backup)
        print(f"\n  Backup cree : {backup}")
    else:
        print(f"\n  Backup existant conserve : {backup}")

    # --- Ecriture ---
    out = Path(args.output) if args.output else rou_path
    out.write_text(patched, encoding="utf-8")
    print(f"  Fichier mis a jour : {out}")

    show_curriculum(args.rou)


if __name__ == "__main__":
    main()
