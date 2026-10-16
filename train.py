"""
Catanbot trainer: self-play evolution of Catanatron's value-function weights.

How it gets better over time
----------------------------
- data/champion.json holds the current best weights (starts from Catanatron's defaults).
- Each generation, a few "challenger" weight sets are made by randomly nudging the champion.
- Each challenger plays the champion head-to-head (1v1) in many games, run in parallel
  across your CPU cores. Every board is played twice with seats swapped, so luck of the
  board and turn order cancels out.
- A challenger only replaces the champion if it wins by a statistically clear margin.
  So the champion can only get stronger, never drift worse.
- Progress is appended to data/history.csv, including a fixed benchmark against the stock
  Catanatron bot, so you can chart improvement over time.
- Stop any time with Ctrl+C; rerun the same command later and it picks up where it left off.

Usage (from the catanbot folder, with the venv active):
    python train.py                          # train forever, all cores
    python train.py --generations 20         # stop after 20 generations
    python train.py --workers 6 --games 400  # 6 processes, 400 games per challenger
    python train.py --vps 15 --discard 9     # match your room's rules
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

from catanatron import Game, Color
from catanatron.players.value import ValueFunctionPlayer, DEFAULT_WEIGHTS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")  # committed to git so progress follows you between machines
CHAMPION_PATH = os.path.join(DATA_DIR, "champion.json")
HISTORY_PATH = os.path.join(DATA_DIR, "history.csv")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "champions")

# public_vps dominates by design ("winning is best at all costs"); don't mutate it.
FROZEN = {"public_vps"}


# ---------------------------------------------------------------- game workers
def make_player(color, weights):
    # value_fn="contender" is required: Catanatron's "base" scorer ignores custom weights.
    return ValueFunctionPlayer(
        color, ValueFunctionPlayer.Params(value_fn="contender", weights=dict(weights))
    )


def play_pair(job):
    """Play one board twice with seats swapped. Returns challenger's score in [0, 2]."""
    seed, challenger, champion, rules = job
    score = 0.0
    for challenger_first in (True, False):
        a = make_player(Color.RED, challenger if challenger_first else champion)
        b = make_player(Color.BLUE, champion if challenger_first else challenger)
        game = Game([a, b], seed=seed, **rules)
        winner = game.play()
        challenger_color = Color.RED if challenger_first else Color.BLUE
        if winner is None:
            score += 0.5  # hit the turn limit: call it a draw
        elif winner == challenger_color:
            score += 1.0
    return score


def match(pool, challenger, champion, n_games, rules, base_seed):
    pairs = max(1, n_games // 2)
    jobs = [(base_seed + i, challenger, champion, rules) for i in range(pairs)]
    total = sum(pool.map(play_pair, jobs, chunksize=1))
    return total / (2 * pairs), 2 * pairs


# ---------------------------------------------------------------- evolution
def mutate(weights, sigma, rng):
    """Multiply each weight by a log-normal factor, so signs and scales are preserved."""
    out = {}
    for k, v in weights.items():
        if k in FROZEN or v == 0:
            out[k] = v if v != 0 else rng.choice([0, rng.uniform(0, 1)])
        else:
            out[k] = v * math.exp(rng.gauss(0, sigma))
    return out


def z_score(win_rate, n):
    """How many standard errors above 50% this result is."""
    return (win_rate - 0.5) / math.sqrt(0.25 / n)


def load_champion():
    if os.path.exists(CHAMPION_PATH):
        with open(CHAMPION_PATH) as f:
            data = json.load(f)
        return data["weights"], data.get("generation", 0), data.get("sigma", 0.3)
    return dict(DEFAULT_WEIGHTS), 0, 0.3


def save_champion(weights, generation, sigma):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = CHAMPION_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"generation": generation, "sigma": sigma, "weights": weights}, f, indent=2)
    os.replace(tmp, CHAMPION_PATH)  # atomic, so Ctrl+C can't corrupt it
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(os.path.join(CHECKPOINT_DIR, f"gen{generation:04d}.json"), "w") as f:
        json.dump(weights, f, indent=2)


def log_history(row):
    os.makedirs(DATA_DIR, exist_ok=True)
    new = not os.path.exists(HISTORY_PATH)
    with open(HISTORY_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel game processes (default: all cores but one)")
    ap.add_argument("--generations", type=int, default=0, help="0 = run until Ctrl+C")
    ap.add_argument("--challengers", type=int, default=4, help="mutants tried per generation")
    ap.add_argument("--games", type=int, default=200, help="screening games per challenger")
    ap.add_argument("--confirm", type=int, default=600, help="extra games to confirm a promotion")
    ap.add_argument("--benchmark", type=int, default=400, help="games vs stock bot after each promotion")
    ap.add_argument("--z", type=float, default=2.0, help="evidence needed to promote (std errors)")
    ap.add_argument("--vps", type=int, default=10, help="victory points to win")
    ap.add_argument("--discard", type=int, default=7, help="discard limit when a 7 is rolled")
    ap.add_argument("--friendly-robber", action="store_true", help="robber can't hit players under 3 VP")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rules = {"vps_to_win": args.vps, "discard_limit": args.discard,
             "friendly_robber": args.friendly_robber}
    champion, generation, sigma = load_champion()
    rng = random.Random(args.seed)
    seed_base = rng.randrange(10**9)

    print(f"Catanbot trainer | {args.workers} workers | rules {rules}")
    print(f"Resuming at generation {generation}, mutation size {sigma:.3f}. Ctrl+C to stop.\n")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        try:
            while args.generations == 0 or generation < args.generations:
                t0 = time.time()
                generation += 1
                best, best_rate, best_n = None, 0.0, 0

                # 1) Screen several mutants quickly.
                for i in range(args.challengers):
                    cand = mutate(champion, sigma, rng)
                    seed_base += 100_000
                    rate, n = match(pool, cand, champion, args.games, rules, seed_base)
                    print(f"  gen {generation} challenger {i + 1}: {rate:.1%} over {n} games")
                    if rate > best_rate:
                        best, best_rate, best_n = cand, rate, n

                # 2) Confirm the best one on fresh boards before trusting it.
                promoted = False
                if best is not None and best_rate > 0.5:
                    seed_base += 100_000
                    c_rate, c_n = match(pool, best, champion, args.confirm, rules, seed_base)
                    z = z_score(c_rate, c_n)  # judge on unseen boards only
                    print(f"  confirm: {c_rate:.1%} over {c_n} games (z={z:+.2f})")
                    if z >= args.z:
                        champion, promoted = best, True

                # 3) Adapt mutation size: explore more when stuck, refine when improving.
                sigma = min(1.0, sigma * 1.1) if not promoted else max(0.05, sigma * 0.85)

                bench = ""
                if promoted:
                    seed_base += 100_000
                    bench_rate, _ = match(pool, champion, dict(DEFAULT_WEIGHTS),
                                          args.benchmark, rules, seed_base)
                    bench = f"{bench_rate:.4f}"
                    print(f"  ** PROMOTED. New champion beats stock Catanatron {bench_rate:.1%} of the time")

                save_champion(champion, generation, sigma)
                log_history({
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "generation": generation,
                    "best_screen_rate": f"{best_rate:.4f}",
                    "promoted": int(promoted),
                    "vs_stock_rate": bench,
                    "sigma": f"{sigma:.4f}",
                    "seconds": f"{time.time() - t0:.1f}",
                })
                print(f"  gen {generation} done in {time.time() - t0:.0f}s\n")
        except KeyboardInterrupt:
            print("\nStopping. Champion saved; rerun the same command to continue.")
            save_champion(champion, generation, sigma)
            pool.shutdown(cancel_futures=True)
            sys.exit(0)


if __name__ == "__main__":  # required on Windows for multiprocessing
    main()
