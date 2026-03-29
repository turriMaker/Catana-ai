from catanatron import Game, Color
from catanatron.models.player import Player, RandomPlayer
from catanatron.models.enums import ActionType
from collections import Counter
import random

class HeuristicPlayer(Player):
    PRIORITY = [
        ActionType.BUILD_CITY,
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_ROAD,
        ActionType.BUY_DEVELOPMENT_CARD,
        ActionType.MARITIME_TRADE,
    ]

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        for action_type in self.PRIORITY:
            candidates = [a for a in playable_actions if a.action_type == action_type]
            if candidates:
                return random.choice(candidates)

        return random.choice(playable_actions)


def benchmark(num_games=200):
    wins = Counter()

    for _ in range(num_games):
        players = [
            HeuristicPlayer(Color.RED),
            RandomPlayer(Color.BLUE),
            RandomPlayer(Color.WHITE),
            RandomPlayer(Color.ORANGE),
        ]
        game = Game(players)
        game.play()
        winner = game.winning_color()
        if winner:
            wins[winner] += 1

    print(f"\nResultados en {num_games} partidas:")
    print(f"  RED (Heurístico): {wins[Color.RED]} victorias ({100*wins[Color.RED]/num_games:.1f}%)")
    for color in [Color.BLUE, Color.WHITE, Color.ORANGE]:
        print(f"  {color.name} (Aleatorio): {wins[color]} victorias ({100*wins[color]/num_games:.1f}%)")

    mejora = (wins[Color.RED]/num_games - 0.25) * 100
    print(f"\n  Esperado si fuera aleatorio: 25.0%")
    print(f"  Mejora sobre aleatorio: {mejora:+.1f} puntos porcentuales")

benchmark(200)
