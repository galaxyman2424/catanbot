# Catanbot

A Catan bot for 1v1 games on colonist.io, built on [Catanatron](https://github.com/bcollazo/catanatron).
It improves itself through parallel self-play: new versions only replace the champion when they beat it by a statistically clear margin.

## Layout

| Path | What it is |
| --- | --- |
| `train.py` | Self-play trainer. Runs games in parallel on all cores but one. |
| `data/` | Training progress (champion weights, history, snapshots). Committed so it follows the repo. |
| `games/` | ColonyHistorian recordings of real colonist.io games. |
| `requirements.txt` | Catanatron, pinned to a tested commit. |

## Setup (Windows, PowerShell)

Requires regular CPython 3.11+ (not GraalPy) and Git.

```powershell
git clone https://github.com/galaxyman2424/catanbot.git
cd catanbot
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Mac/Linux, use `python3 -m venv venv` and `source venv/bin/activate` instead.

## Training

```powershell
python train.py --generations 3 --games 40 --confirm 40 --benchmark 40   # quick check
python train.py                                                           # train until Ctrl+C
python train.py --vps 15 --discard 9 --friendly-robber                    # match room rules
python train.py --help                                                    # all options
```

Ctrl+C stops safely; rerun the same command to resume.

## Saving progress

After a training session, commit the results so the other machine picks them up:

```powershell
git add data
git commit -m "Training: gen <N>"
git push
```

Always `git pull` before you start training, so you resume from the latest champion.
