"""
train.py — REINFORCE training loop for the Catan neural network.

Usage:
    python train.py                      # train from scratch
    python train.py --load model.pt      # resume from checkpoint
"""

import argparse
import csv
import os
import random
from collections import Counter, deque

import torch
import torch.optim as optim

from catanatron import Game, Color
from catanatron.models.player import RandomPlayer

from model import CatanNet
from neural_player import NeuralPlayer
from heuristic_player import HeuristicPlayer

# ─── Hyperparameters ──────────────────────────────────────────────────────────

LR           = 1e-4      # learning rate
HIDDEN_SIZE  = 256
NUM_LAYERS   = 3
EVAL_EVERY   = 100       # evaluate against heuristic every N episodes
EVAL_GAMES   = 50        # games per evaluation
SAVE_EVERY   = 500       # save checkpoint every N episodes
COLORS       = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]

# ─── Reward shaping ───────────────────────────────────────────────────────────

def compute_reward(winner_color, my_color, state, my_color_ref):
    """Return a scalar reward for the RL player."""
    if winner_color == my_color:
        return 1.0
    else:
        return -1.0


# ─── Single training episode ──────────────────────────────────────────────────

def run_episode(model, optimizer, opponents="heuristic"):
    """
    Play one game with the RL player as RED.
    Compute REINFORCE loss and update the model.

    Returns: reward (+1 win / -1 loss)
    """
    my_color = Color.RED

    if opponents == "heuristic":
        others = [HeuristicPlayer(c) for c in [Color.BLUE, Color.WHITE, Color.ORANGE]]
    else:
        others = [RandomPlayer(c) for c in [Color.BLUE, Color.WHITE, Color.ORANGE]]

    rl_player = NeuralPlayer(my_color, model, training=True)
    players   = [rl_player] + others

    game = Game(players)
    game.play()

    winner = game.winning_color()
    reward = compute_reward(winner, my_color, game.state, my_color)

    if not rl_player.log_probs:
        return reward

    # REINFORCE loss: -E[log π(a|s) * R]
    log_probs = torch.stack(rl_player.log_probs)
    loss = -(log_probs * reward).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return reward


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, num_games=EVAL_GAMES, opponent="heuristic"):
    """Play num_games in eval mode and return win rate."""
    model.eval()
    wins = 0
    my_color = Color.RED

    for _ in range(num_games):
        if opponent == "heuristic":
            others = [HeuristicPlayer(c) for c in [Color.BLUE, Color.WHITE, Color.ORANGE]]
        else:
            others = [RandomPlayer(c) for c in [Color.BLUE, Color.WHITE, Color.ORANGE]]

        rl_player = NeuralPlayer(my_color, model, training=False)
        players   = [rl_player] + others

        game = Game(players)
        game.play()

        if game.winning_color() == my_color:
            wins += 1

    model.train()
    return wins / num_games


# ─── Training loop ────────────────────────────────────────────────────────────

def train(num_episodes=5000, load_path=None, save_path="model.pt", opponents="heuristic",
          metrics_path="metrics.csv"):
    model     = CatanNet(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    start_episode = 0

    if load_path:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_episode = checkpoint.get("episode", 0)
        print(f"Resumed from episode {start_episode}")

    model.train()

    recent_rewards = deque(maxlen=100)
    best_win_rate  = 0.0

    # Open CSV for logging (append mode so --load preserves history)
    csv_exists = os.path.exists(metrics_path)
    csv_file   = open(metrics_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["episode", "win_rate_train", "win_rate_eval"])

    print(f"Training for {num_episodes} episodes vs {opponents} opponents")
    print(f"  Model:   {HIDDEN_SIZE} hidden × {NUM_LAYERS} layers")
    print(f"  LR:      {LR}")
    print(f"  Metrics: {metrics_path}")
    print()

    for ep in range(start_episode, start_episode + num_episodes):
        reward = run_episode(model, optimizer, opponents=opponents)
        recent_rewards.append(reward)

        if (ep + 1) % 10 == 0:
            avg = sum(recent_rewards) / len(recent_rewards)
            win_rate_train = (avg + 1) / 2
            print(f"Episode {ep+1:5d} | avg reward (last {len(recent_rewards)}): {avg:+.3f} | win rate: {win_rate_train:.1%}")
            csv_writer.writerow([ep + 1, f"{win_rate_train:.4f}", ""])
            csv_file.flush()

        if (ep + 1) % EVAL_EVERY == 0:
            win_rate_eval = evaluate(model)
            print(f"  ── Eval vs heuristic ({EVAL_GAMES} games): {win_rate_eval:.1%} ──")
            # Overwrite the last train row with the eval value
            csv_writer.writerow([ep + 1, "", f"{win_rate_eval:.4f}"])
            csv_file.flush()

            if win_rate_eval > best_win_rate:
                best_win_rate = win_rate_eval
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "episode": ep + 1,
                }, "best_model.pt")
                print(f"     New best! Saved to best_model.pt")

        if (ep + 1) % SAVE_EVERY == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": ep + 1,
            }, save_path)
            print(f"  Checkpoint saved to {save_path}")

    csv_file.close()

    # Final save
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": start_episode + num_episodes,
    }, save_path)
    print(f"\nTraining done. Model saved to {save_path}")
    print(f"Best win rate vs heuristic: {best_win_rate:.1%}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int,   default=5000)
    parser.add_argument("--load",       type=str,   default=None)
    parser.add_argument("--save",       type=str,   default="model.pt")
    parser.add_argument("--opponents",  type=str,   default="heuristic",
                        choices=["heuristic", "random"])
    parser.add_argument("--metrics",    type=str,   default="metrics.csv")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        load_path=args.load,
        save_path=args.save,
        opponents=args.opponents,
        metrics_path=args.metrics,
    )
