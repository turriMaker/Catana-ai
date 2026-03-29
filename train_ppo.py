"""
train_ppo.py — PPO training loop for Catan AI (Fase 4).

Usage:
    # Iniciar desde cero
    python train_ppo.py

    # Cargar pesos REINFORCE previos (recomendado)
    python train_ppo.py --load best_model.pt

    # Continuar un checkpoint PPO
    python train_ppo.py --load model_ppo.pt

    # Self-play contra versiones anteriores del pool
    python train_ppo.py --load best_model_ppo.pt --opponents selfplay

    # Mezcla: 50% pool, 50% heurístico
    python train_ppo.py --load best_model_ppo.pt --opponents mixed

    # Cambiar hiperparámetros
    python train_ppo.py --load best_model.pt --episodes 10000 --batch 32
"""

import argparse
import csv
import glob
import os
import random
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim

from catanatron import Game, Color
from catanatron.models.player import RandomPlayer

from model import CatanActorCritic
from neural_player import PPOPlayer
from heuristic_player import HeuristicPlayer

# ─── Hiperparámetros ──────────────────────────────────────────────────────────

LR            = 1e-4
HIDDEN_SIZE   = 256
NUM_LAYERS    = 3
GAMMA         = 0.99   # descuento (usar 1.0 si el juego es muy largo)
CLIP_EPS      = 0.15    # epsilon del clipping PPO
PPO_EPOCHS    = 2      # actualizaciones por batch
VALUE_COEF    = 0.5    # peso de la pérdida del crítico
ENTROPY_COEF  = 0.01   # bonificación de entropía (exploración)
MAX_GRAD_NORM = 0.5

BATCH_SIZE       = 64     # episodios por actualización
EVAL_EVERY       = 100    # evaluar cada N episodios
EVAL_GAMES       = 200
SAVE_EVERY       = 500    # checkpoint periódico
POOL_SAVE_EVERY  = 500    # guardar snapshot en pool
POOL_DIR         = "pool"
COLORS           = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]


# ─── Pool de jugadores anteriores ─────────────────────────────────────────────

def save_pool_snapshot(model, ep):
    """Guarda una copia del modelo actual en el pool."""
    os.makedirs(POOL_DIR, exist_ok=True)
    path = os.path.join(POOL_DIR, f"model_ep{ep}.pt")
    torch.save(model.state_dict(), path)
    return path


def load_pool_opponents(colors):
    """
    Carga oponentes aleatorios del pool (uno por color).
    Retorna lista de PPOPlayer con modelos congelados, o None si el pool está vacío.
    """
    snapshots = sorted(glob.glob(os.path.join(POOL_DIR, "model_ep*.pt")))
    if not snapshots:
        return None

    opponents = []
    for color in colors:
        snap_path = random.choice(snapshots)
        pool_model = CatanActorCritic(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
        pool_model.load_state_dict(torch.load(snap_path, map_location="cpu"))
        pool_model.eval()
        opponents.append(PPOPlayer(color, pool_model, training=False))
    return opponents


# ─── Recolección de un episodio ───────────────────────────────────────────────

def collect_episode(model, opponents="heuristic", mix=0.5):
    """
    Juega una partida con PPOPlayer como RED.

    opponents:
        "heuristic"  — todos los oponentes son HeuristicPlayer
        "random"     — todos los oponentes son RandomPlayer
        "selfplay"   — oponentes cargados del pool (o heurístico si pool vacío)
        "mixed"      — cada oponente es pool o heurístico con p=0.5 cada uno

    Retorna (states, actions, log_probs_old, masks, reward, pool_count).
    pool_count: cuántos de los 3 oponentes vinieron del pool (0-3).
    """
    my_color = Color.RED
    other_colors = [Color.BLUE, Color.WHITE, Color.ORANGE]
    pool_count = 0

    if opponents == "heuristic":
        others = [HeuristicPlayer(c) for c in other_colors]

    elif opponents == "random":
        others = [RandomPlayer(c) for c in other_colors]

    elif opponents == "selfplay":
        pool_ops = load_pool_opponents(other_colors)
        if pool_ops is None:
            others = [HeuristicPlayer(c) for c in other_colors]
        else:
            others = pool_ops
            pool_count = 3

    elif opponents == "mixed":
        pool_ops = load_pool_opponents(other_colors)
        if pool_ops is None:
            others = [HeuristicPlayer(c) for c in other_colors]
        else:
            others = []
            for i, c in enumerate(other_colors):
                if random.random() < mix:
                    others.append(pool_ops[i])
                    pool_count += 1
                else:
                    others.append(HeuristicPlayer(c))

    else:
        others = [HeuristicPlayer(c) for c in other_colors]

    ppo_player = PPOPlayer(my_color, model, training=True)
    game = Game([ppo_player] + others)
    game.play()

    reward = 1.0 if game.winning_color() == my_color else -1.0
    return (ppo_player.states, ppo_player.actions,
            ppo_player.log_probs, ppo_player.masks, reward, pool_count)


# ─── Actualización PPO ────────────────────────────────────────────────────────

def ppo_update(model, optimizer, trajectories):
    """
    trajectories: lista de (states, actions, log_probs_old, masks, reward)
    Realiza PPO_EPOCHS rondas de actualización y retorna policy_loss promedio.
    """
    all_states, all_actions = [], []
    all_lp_old, all_masks   = [], []
    all_returns             = []

    for states, actions, log_probs_old, masks, reward, *_ in trajectories:
        T = len(states)
        if T == 0:
            continue
        # Retorno descontado: G_t = gamma^(T-1-t) * R
        # (recompensa terminal; todos los pasos comparten el mismo resultado)
        returns = [GAMMA ** (T - 1 - t) * reward for t in range(T)]

        all_states.extend(states)
        all_actions.extend(actions)
        all_lp_old.extend(log_probs_old)
        all_masks.extend(masks)
        all_returns.extend(returns)

    if not all_states:
        return 0.0

    states_t  = torch.stack(all_states)                              # (N, STATE)
    actions_t = torch.tensor(all_actions, dtype=torch.long)          # (N,)
    lp_old_t  = torch.stack(all_lp_old)                              # (N,)
    masks_t   = torch.stack(all_masks)                               # (N, ACTION)
    returns_t = torch.tensor(all_returns, dtype=torch.float32)       # (N,)

    total_policy_loss = 0.0

    for _ in range(PPO_EPOCHS):
        logits, values = model(states_t)                             # (N, A), (N,)

        # Distribución mascarada sobre acciones válidas
        masked_logits = logits.clone()
        masked_logits[~masks_t] = float('-inf')
        probs = torch.softmax(masked_logits, dim=-1)
        dist  = torch.distributions.Categorical(probs)

        log_probs_new = dist.log_prob(actions_t)                     # (N,)

        # Ventaja: A_t = G_t - V(s_t), normalizada
        advantages = returns_t - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Pérdida PPO-Clip
        ratio  = torch.exp(log_probs_new - lp_old_t)
        surr1  = ratio * advantages
        surr2  = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Pérdida del crítico (MSE)
        value_loss = F.mse_loss(values, returns_t)

        # Bonificación de entropía (incentiva exploración)
        entropy = dist.entropy().mean()

        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_policy_loss += policy_loss.item()

    return total_policy_loss / PPO_EPOCHS


# ─── Evaluación ───────────────────────────────────────────────────────────────

def evaluate(model, num_games=EVAL_GAMES, opponent="heuristic"):
    model.eval()
    wins = 0
    my_color = Color.RED

    for _ in range(num_games):
        if opponent == "heuristic":
            others = [HeuristicPlayer(c) for c in [Color.BLUE, Color.WHITE, Color.ORANGE]]
        else:
            others = [RandomPlayer(c) for c in [Color.BLUE, Color.WHITE, Color.ORANGE]]

        player = PPOPlayer(my_color, model, training=False)
        game   = Game([player] + others)
        game.play()

        if game.winning_color() == my_color:
            wins += 1

    model.train()
    return wins / num_games


# ─── Loop principal ───────────────────────────────────────────────────────────

def train(num_episodes=5000, load_path=None, save_path="model_ppo.pt",
          opponents="heuristic", metrics_path="metrics_ppo.csv",
          batch_size=BATCH_SIZE, mix=0.5):

    model = CatanActorCritic(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    start_episode = 0

    if load_path:
        checkpoint = torch.load(load_path, map_location="cpu")
        sd = checkpoint.get("model", checkpoint)
        if any(k.startswith("net.") for k in sd.keys()):
            # Checkpoint REINFORCE → transferir trunk + policy head
            model.load_from_reinforce(sd)
            start_episode = checkpoint.get("episode", 0)
            print(f"Pesos REINFORCE cargados desde {load_path} (episodio {start_episode})")
            print("  Value head inicializado aleatoriamente.")
        else:
            model.load_state_dict(sd)
            start_episode = checkpoint.get("episode", 0)
            print(f"Checkpoint PPO cargado desde {load_path} (episodio {start_episode})")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()

    best_win_rate = 0.0
    recent_wins   = []

    csv_exists = os.path.exists(metrics_path)
    csv_file   = open(metrics_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["episode", "win_rate_train", "win_rate_eval", "policy_loss", "pool_pct"])

    # Guardar snapshot inicial en el pool (para que selfplay/mixed no arranquen vacíos)
    if opponents in ("selfplay", "mixed"):
        os.makedirs(POOL_DIR, exist_ok=True)
        existing = glob.glob(os.path.join(POOL_DIR, "model_ep*.pt"))
        if not existing:
            path = save_pool_snapshot(model, start_episode)
            print(f"Pool vacío — snapshot inicial guardado: {path}")

    pool_size = len(glob.glob(os.path.join(POOL_DIR, "model_ep*.pt")))

    print(f"PPO — entrenando {num_episodes} episodios vs {opponents}")
    print(f"  batch={batch_size}, ppo_epochs={PPO_EPOCHS}, clip={CLIP_EPS}, lr={LR}")
    print(f"  Pool: {pool_size} modelos en '{POOL_DIR}/' (guardando cada {POOL_SAVE_EVERY} ep)")
    print(f"  Métricas: {metrics_path}")
    print()

    ep = start_episode
    end_ep = start_episode + num_episodes

    while ep < end_ep:
        # Recolectar un batch de episodios
        batch = []
        batch_wins = 0
        batch_pool_ops = 0   # total oponentes pool en el batch (máx batch_size*3)
        for _ in range(min(batch_size, end_ep - ep)):
            traj = collect_episode(model, opponents=opponents, mix=mix)
            batch.append(traj)
            batch_wins += 1 if traj[4] > 0 else 0
            batch_pool_ops += traj[5]
            ep += 1

        # Actualizar con PPO
        policy_loss = ppo_update(model, optimizer, batch)

        win_rate = batch_wins / len(batch)
        recent_wins.append(win_rate)
        if len(recent_wins) > 10:
            recent_wins.pop(0)
        avg_wr = sum(recent_wins) / len(recent_wins)

        if opponents in ("selfplay", "mixed"):
            total_ops = len(batch) * 3
            pool_pct = batch_pool_ops / total_ops
            mix_str = f" | pool: {batch_pool_ops}/{total_ops} ({pool_pct:.0%})"
        else:
            mix_str = ""

        print(f"Episodio {ep:6d} | win rate batch: {win_rate:.1%} "
              f"| avg(10 batches): {avg_wr:.1%} | policy_loss: {policy_loss:.4f}"
              f"{mix_str}")
        pool_pct = batch_pool_ops / (len(batch) * 3) if opponents in ("selfplay", "mixed") else ""
        csv_writer.writerow([ep, f"{win_rate:.4f}", "", f"{policy_loss:.4f}", f"{pool_pct:.4f}" if pool_pct != "" else ""])
        csv_file.flush()

        # Evaluación formal
        if ep % EVAL_EVERY == 0 or ep >= end_ep:
            wr_eval = evaluate(model)
            print(f"  ── Eval vs heurístico ({EVAL_GAMES} partidas): {wr_eval:.1%} ──")
            csv_writer.writerow([ep, "", f"{wr_eval:.4f}", ""])
            csv_file.flush()

            if wr_eval > best_win_rate:
                best_win_rate = wr_eval
                torch.save({"model": model.state_dict(),
                            "episode": ep}, "best_model_ppo.pt")
                print(f"     ¡Nuevo récord! Guardado en best_model_ppo.pt")

        # Checkpoint periódico
        if ep % SAVE_EVERY == 0:
            torch.save({"model": model.state_dict(),
                        "episode": ep}, save_path)
            print(f"  Checkpoint guardado en {save_path}")

        # Snapshot al pool
        if ep % POOL_SAVE_EVERY == 0:
            path = save_pool_snapshot(model, ep)
            pool_size = len(glob.glob(os.path.join(POOL_DIR, "model_ep*.pt")))
            print(f"  Pool snapshot: {path} ({pool_size} modelos en pool/)")

    csv_file.close()
    torch.save({"model": model.state_dict(), "episode": ep}, save_path)
    print(f"\nEntrenamiento terminado. Modelo guardado en {save_path}")
    print(f"Mejor win rate vs heurístico: {best_win_rate:.1%}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int, default=5000)
    parser.add_argument("--load",      type=str, default=None)
    parser.add_argument("--save",      type=str, default="model_ppo.pt")
    parser.add_argument("--opponents", type=str, default="heuristic",
                        choices=["heuristic", "random", "selfplay", "mixed"])
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    parser.add_argument("--metrics",   type=str, default=f"corridas/metrics_ppo_{ts}.csv")
    parser.add_argument("--batch",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--mix",       type=float, default=0.5,
                        help="Proporción de oponentes del pool en modo mixed (0.0–1.0)")
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        load_path=args.load,
        save_path=args.save,
        opponents=args.opponents,
        metrics_path=args.metrics,
        batch_size=args.batch,
        mix=args.mix,
    )
