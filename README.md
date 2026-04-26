# china-grant-rl-model

Reinforcement learning for train energy optimization. A Tianshou REINFORCE agent learns to control locomotive throttle (notch 0–8) each second to minimize energy over a 74.9 km A→B trip.

Handoff between coding agents: `pnpm dlx continues`

**Components:** NeTrainSim (C++ simulator) · Gymnasium env wrapper · Tianshou REINFORCE training

## Setup

**Prerequisites:** macOS, Python 3.10+, `brew install qt cmake`

```bash
# 1. Build the C++ simulator
cd NeTrainSim-adjusted && ./build-mac.sh && cd ..

# 2. Generate NeTrainSim input files from route data
python data/generate_netrainsim_input.py

# 3. Python environment
source venv/bin/activate
pip install -r rl/requirements.txt

# 4. Train
python rl/train.py
```

## Project layout

| Path | Purpose |
|------|---------|
| `data/coordinates.csv` | 750 nodes: (node_id, x_m, y_m) for the A→B route |
| `data/data.csv` | 749 segments: grade (‰), curvature, speed limit (m/s) |
| `data/generate_netrainsim_input.py` | Converts CSVs → `data/netrainsim/*.dat` |
| `NeTrainSim-adjusted/` | C++ freight train simulator (source + build) |
| `rl/train_env.py` | Gymnasium environment wrapping the simulator |
| `rl/train.py` | Tianshou REINFORCE training script |
| `CLAUDE.md` | Architecture reference and agent instructions |
| `requirements.txt` | Dependencies such as `gymnasium`, `tianshou`, `torch`, etc. |

## Gotchas discovered during setup

Three non-obvious things that caused failures and required fixes:

1. **Grade unit is per mille (‰), not percent.**  
   `data/data.csv` stores grades as ‰ (e.g. `6.28` = 6.28‰ = 0.628%). NeTrainSim's Davis resistance formula expects percent, so `generate_netrainsim_input.py` divides every grade value by 10 before writing to `linksFile.dat`. If you skip this step, the simulator sees a grade ~10× steeper than reality and the train stalls immediately.

2. **The `-e` flag requires an explicit `true` value.**  
   `-e` is not a boolean switch — it defaults to `"false"` and must be passed as `-e true`. Running just `-e` silently does nothing, so no trajectory CSV is produced and the Python env crashes. Every invocation must include `-e true`.

3. **The sample project train is too heavy for this route.**  
   The NeTrainSim sample consist is ~8 000 t (72 freight cars + 4 large locos). Even at the real grade of 0.628% the resistance exceeds the tractive force and the train stalls. The `trainsFile.dat` generated here uses a custom light consist: 1 diesel loco (5 000 kW, 90 t) + 4 freight cars (50 t each) ≈ 290 t total, which has no trouble with the route.

## Verify setup

```bash
# Check simulator runs with route data (run from inside NeTrainSim-adjusted/)
cd NeTrainSim-adjusted && mkdir -p res
./build-mac/src/NeTrainSimConsole/NeTrainSim \
  -n ../data/netrainsim/nodesFile.dat \
  -l ../data/netrainsim/linksFile.dat \
  -t ../data/netrainsim/trainsFile.dat \
  -o res -e true -p 1.0        # -e takes 'true', not a bare flag

# Check Gymnasium env compliance
python -c "
import sys; sys.path.insert(0,'.')
from rl.train_env import NeTrainSimEnv
from gymnasium.utils.env_checker import check_env
check_env(NeTrainSimEnv())
print('OK')
"
```
