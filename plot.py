"""
plot.py — Visualiza el progreso del entrenamiento.

Uso:
    python plot.py --file corridas/metrics_ppo_001.csv
    python plot.py --file corridas/metrics_ppo_001.csv corridas/metrics_ppo_002.csv
    python plot.py --file corridas/metrics_ppo_001.csv --save grafico.png
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

COLORS = [
    ("steelblue",  "crimson"),
    ("seagreen",   "darkorange"),
    ("mediumpurple","deeppink"),
    ("saddlebrown","teal"),
    ("slategray",  "gold"),
]


def load_metrics(path):
    episodes_train, win_rates_train = [], []
    episodes_eval,  win_rates_eval  = [], []
    episodes_pool,  pool_pcts       = [], []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["episode"] == "episode":
                continue
            ep = int(row["episode"])
            if row.get("win_rate_train"):
                episodes_train.append(ep)
                win_rates_train.append(float(row["win_rate_train"]) * 100)
            if row.get("win_rate_eval"):
                episodes_eval.append(ep)
                win_rates_eval.append(float(row["win_rate_eval"]) * 100)
            if row.get("pool_pct"):
                episodes_pool.append(ep)
                pool_pcts.append(float(row["pool_pct"]) * 100)

    return episodes_train, win_rates_train, episodes_eval, win_rates_eval, episodes_pool, pool_pcts


def smooth(values, window=20):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


def plot(paths, save_path=None):
    any_pool = False
    all_data = []
    for path in paths:
        data = load_metrics(path)
        all_data.append(data)
        if data[4]:  # episodes_pool no vacío
            any_pool = True

    # Si hay datos de pool usamos dos subplots apilados, sino uno solo
    if any_pool:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax2 = None

    ax.axhline(25.0, color="gray",   linestyle="--", linewidth=1, label="Baseline aleatorio (25%)")
    ax.axhline(46.5, color="orange", linestyle="--", linewidth=1, label="Objetivo Fase 3 (46.5%)")

    multi = len(paths) > 1

    for i, (path, data) in enumerate(zip(paths, all_data)):
        ep_train, wr_train, ep_eval, wr_eval, ep_pool, pool_pcts = data
        color_train, color_eval = COLORS[i % len(COLORS)]
        label = os.path.basename(path).replace(".csv", "") if multi else None

        if ep_train:
            ax.plot(ep_train, wr_train, color=color_train, alpha=0.15, linewidth=0.8)
            ax.plot(ep_train, smooth(wr_train, window=30), color=color_train,
                    linewidth=2, label=f"{label} — entrenamiento" if multi else "Win rate entrenamiento (media móvil)")

        if ep_eval:
            ax.plot(ep_eval, wr_eval, "o-", color=color_eval, linewidth=2,
                    markersize=5, label=f"{label} — eval" if multi else "Eval vs heurístico (cada 100 ep)")

        if ax2 is not None and ep_pool:
            ax2.plot(ep_pool, smooth(pool_pcts, window=10), color=color_train,
                     linewidth=2, label=label)
            ax2.plot(ep_pool, pool_pcts, color=color_train, alpha=0.15, linewidth=0.8)

    ax.set_ylabel("Tasa de victoria (%)", fontsize=12)
    any_ppo = any("ppo" in p.lower() for p in paths)
    fase = "4 (PPO)" if any_ppo else "3 (REINFORCE)"
    title = f"Comparación de corridas — Catan AI Fase {fase}" if multi else f"Progreso del entrenamiento — Catan AI Fase {fase}"
    ax.set_title(title, fontsize=14)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_ylim(0, 80)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if ax2 is not None:
        ax2.set_xlabel("Episodio", fontsize=12)
        ax2.set_ylabel("% oponentes pool", fontsize=11)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax2.set_ylim(0, 105)
        ax2.axhline(50.0, color="gray", linestyle=":", linewidth=1)
        ax2.grid(True, alpha=0.3)
        if multi:
            ax2.legend(fontsize=9)
    else:
        ax.set_xlabel("Episodio", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Gráfico guardado en {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, nargs="+", default=["metrics.csv"])
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    plot(args.file, save_path=args.save)


if __name__ == "__main__":
    main()
