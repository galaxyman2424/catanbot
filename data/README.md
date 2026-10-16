# Training data

Written by `train.py` and committed to git, so training progress moves with the repo between machines.

- `champion.json` — the current best weights, the generation number, and the mutation size. The trainer resumes from here.
- `history.csv` — one row per generation: best challenger score, whether it was promoted, and (on promotions) the champion's win rate against stock Catanatron.
- `champions/genNNNN.json` — a snapshot of the champion after each generation, for rolling back or charting.

To restart training from scratch, delete `champion.json`, `history.csv` and `champions/`.
