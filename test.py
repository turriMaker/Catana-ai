from catanatron import Game, Color
from catanatron.models.player import RandomPlayer
from collections import Counter

def benchmark(num_games=100):
    wins = Counter()
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    
    for _ in range(num_games):
        players = [RandomPlayer(c) for c in colors]
        game = Game(players)
        game.play()
        winner = game.winning_color()
        if winner:
            wins[winner] += 1
    
    print(f"Resultados en {num_games} partidas:")
    for color, w in wins.most_common():
        print(f"  {color.name}: {w} victorias ({100*w/num_games:.1f}%)")

benchmark(200)