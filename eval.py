"""
eval.py — Evalúa un modelo entrenado contra el heurístico y el aleatorio.

Uso:
    python eval.py                        # evalúa best_model.pt
    python eval.py --model model.pt       # evalúa otro checkpoint
    python eval.py --games 200            # más partidas para mayor precisión
"""

import argparse
from collections import Counter

import torch

from catanatron import Game, Color
from catanatron.models.player import RandomPlayer

from model import CatanNet
from neural_player import NeuralPlayer
from heuristic_player import HeuristicPlayer

COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]


def run_tournament(model, num_games, opponent_type):
    """
    Enfrenta la red neuronal (RED) contra 3 oponentes del tipo dado.
    Devuelve un Counter de victorias por color.
    """
    wins = Counter()

    for _ in range(num_games):
        rl_player = NeuralPlayer(Color.RED, model, training=False)

        if opponent_type == "heuristic":
            others = [HeuristicPlayer(c) for c in [Color.BLUE, Color.WHITE, Color.ORANGE]]
        else:
            others = [RandomPlayer(c) for c in [Color.BLUE, Color.WHITE, Color.ORANGE]]

        game = Game([rl_player] + others)
        game.play()

        winner = game.winning_color()
        if winner:
            wins[winner] += 1

    return wins


def print_results(wins, num_games, opponent_type, label="Red Neuronal"):
    total = sum(wins.values())
    print(f"\nResultados vs {opponent_type} ({num_games} partidas):")
    print(f"  {label} (RED):  {wins[Color.RED]:3d} victorias  ({100*wins[Color.RED]/num_games:.1f}%)")

    opp_name = "Heurístico" if opponent_type == "heuristic" else "Aleatorio"
    for color in [Color.BLUE, Color.WHITE, Color.ORANGE]:
        print(f"  {opp_name} ({color.name}): {wins[color]:3d} victorias  ({100*wins[color]/num_games:.1f}%)")

    baseline = 25.0
    target   = 46.5 if opponent_type == "heuristic" else 25.0
    actual   = 100 * wins[Color.RED] / num_games

    print()
    print(f"  Baseline aleatorio:  {baseline:.1f}%")
    print(f"  Objetivo (Fase 3):   {target:.1f}%")
    print(f"  Red neuronal:        {actual:.1f}%  ", end="")

    if actual >= target:
        print("✓ OBJETIVO SUPERADO")
    elif actual >= baseline:
        print(f"(por encima del baseline, faltan {target - actual:.1f} pp)")
    else:
        print(f"(por debajo del baseline, sigue entrenando)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="best_model.pt",
                        help="Ruta al checkpoint del modelo")
    parser.add_argument("--games",  type=int, default=200,
                        help="Número de partidas por evaluación")
    args = parser.parse_args()

    print(f"Cargando modelo desde {args.model}...")
    checkpoint = torch.load(args.model, weights_only=True)
    model = CatanNet()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    episode = checkpoint.get("episode", "?")
    print(f"  Episodio de entrenamiento: {episode}")

    # Evaluación vs heurístico
    wins_h = run_tournament(model, args.games, "heuristic")
    print_results(wins_h, args.games, "heuristic")

    # Evaluación vs aleatorio
    wins_r = run_tournament(model, args.games, "random")
    print_results(wins_r, args.games, "random")


if __name__ == "__main__":
    main()
