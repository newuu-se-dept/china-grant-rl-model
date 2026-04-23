# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RL-based train energy optimization: a Tianshou REINFORCE agent controls locomotive throttle
(notch 0–8) each second to minimize total energy consumption over a full A→B trip while
respecting speed limits and arriving within the scheduled time window.

Three components:
1. **NeTrainSim** (`NeTrainSim-adjusted/`) — C++/Qt6 freight train simulator
2. **RL layer** (`rl/`) — Python: Gymnasium env + Tianshou REINFORCE training script
3. **Docs** (`gymnasium-docs/`, `tianshou-docs/`) — API reference (do not modify)

## Repository Layout

```
data/
  coordinates.csv                 750 nodes: tab-sep, no header → node_id  x_m  y_m
  data.csv                        749 segments: comma-sep, header → idx,Grade%,Curvature,Speed_limit_mps
  generate_netrainsim_input.py    Converts CSVs → NeTrainSim .dat files in data/netrainsim/
  netrainsim/
    nodesFile.dat                 Generated — 750 nodes
    linksFile.dat                 Generated — 749 links
    trainsFile.dat                Generated — single diesel train, path node 1→750

NeTrainSim-adjusted/
  src/
    NeTrainSimConsole/main.cpp    CLI entry point; add --interactive flag here (Phase 2)
    NeTrainSim/
      simulator.h / simulator.cpp Time-step loop, CSV output, pause/resume API
      simulatorapi.h              C++ programmatic API (singleton)
      traindefinition/
        train.h / train.cpp       Physics; optimumThrottleLevels injection point
        locomotive.h / locomotive.cpp  throttleLevel (0–1), currentLocNotch (0–8)
      network/
        readwritenetwork.cpp      Parses .dat files; speed in m/s, length in meters
        netlink.cpp               freeFlowSpeed stored and used as m/s
  src/data/sampleProject/        Reference sample (binary .dat → ASCII text confirmed)
  build-mac.sh                   Build script

rl/
  train_env.py                   NeTrainSimEnv (gymnasium.Env subclass)
  train.py                       Tianshou REINFORCE training script
  requirements.txt               Pinned deps matching installed venv

venv/                            Python virtualenv (gymnasium 1.3.0, tianshou 0.5.1, torch 2.x)
```

## Build & Run

### NeTrainSim (C++)

Prerequisites: `brew install qt cmake`

```bash
cd NeTrainSim-adjusted
./build-mac.sh          # builds NeTrainSimConsole; runs sample simulation into res/
```

Manual build:
```bash
cmake -B build-mac -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI=OFF -DBUILD_SERVER=OFF \
  -DCMAKE_PREFIX_PATH=$(brew --prefix qt6)
cmake --build build-mac --target NeTrainSimConsole -j$(sysctl -n hw.logicalcpu)
```

Run with our data + trajectory export (run from inside `NeTrainSim-adjusted/`):
```bash
./build-mac/src/NeTrainSimConsole/NeTrainSim \
  -n ../data/netrainsim/nodesFile.dat \
  -l ../data/netrainsim/linksFile.dat \
  -t ../data/netrainsim/trainsFile.dat \
  -o res -e true -p 1.0
```

CLI flags: `-n` nodes, `-l` links, `-t` trains, `-o` output dir, `-p` timestep in seconds
(default 1.0), `-z enable optimizer`.
**`-e true`** exports the trajectory CSV — `-e` takes a value, NOT a bare flag. `-e` alone uses
default `false` and produces no CSV.

### Data preparation

```bash
python data/generate_netrainsim_input.py   # generates data/netrainsim/*.dat
```

Re-run this whenever `coordinates.csv` or `data.csv` changes.

### Python RL layer

```bash
source venv/bin/activate
pip install -r rl/requirements.txt   # if venv is fresh
python rl/train.py
```

Validate Gymnasium env compliance before training:
```bash
python -c "
import sys; sys.path.insert(0,'.')
from rl.train_env import NeTrainSimEnv
from gymnasium.utils.env_checker import check_env
check_env(NeTrainSimEnv())
print('OK')
"
```

## Data Files

### Source data (in `data/`)

**`coordinates.csv`** — tab-separated, no header, 750 rows:
```
node_id   x_meters    y_meters
1         -27375.11   -21357.03
...
750        29742.51    18689.05
```
Total route: 74.87 km straight-line path (A=node 1, B=node 750).
Consecutive nodes are ~100 m apart.

**`data.csv`** — comma-separated, has header, 749 rows (one per link segment):
```
,Grade,Curvature,Speed limit
1,-0.145,0.042,1.0
...
```
- **Grade**: **per mille (‰)** — divided by 10 when writing to linksFile.dat so NeTrainSim sees %
  (NeTrainSim's Davis formula `20 × weight_tons × grade` expects grade in %). Max: ±6.28‰ = ±0.628%
- **Curvature**: unit passed through to NeTrainSim as-is
- **Speed limit**: **m/s** — values are km/h ÷ 3.6: 1.0, 3.0, 11.1(40), 16.6(60), 19.4(70), 22.2(80)

### Generated NeTrainSim input (in `data/netrainsim/`)

**`nodesFile.dat`** format (ASCII, tab-separated):
```
This is the node file of route1		
<count>  <xScale>  <yScale>           ← scales=1 (coords already in meters)
<id>  <x>  <y>  <isTerminal>  <dwellTime>  <desc>
```
Nodes 1 and 750 are marked `isTerminal=1`.

**`linksFile.dat`** format (ASCII, tab-separated):
```
This is the link file of route1     (many tabs)
<count>  <lengthScale>  <speedScale>   ← both=1
<id>  <from>  <to>  <length_m>  <speed_mps>  <signalNo>  <grade_pct>  <curvature>
       <directions>  <speedVariation>  <hasCatenary>
```
Link lengths are Euclidean distances between consecutive coordinate pairs (meters).
Speed is in m/s (confirmed in netlink.cpp: `length/freeFlowSpeed` gives seconds).
Grade is stored as percent (%) — generator divides data.csv ‰ values by 10.
`directions=1` (unidirectional A→B), `hasCatenary=0` (diesel).

**`trainsFile.dat`** format (ASCII):
```
Automatic Trains Definition
1                                           ← number of trains
1  <path>  <startTime>  <frictionCoef>  <loco_defs>  <car_defs>
```
Loco field order: `Count, Power(kW), TransmissionEff, NoOfAxles, AirDragCoeff, FrontalArea(m²), Length(m), GrossWeight(t), Type`
Car field order:  `Count, NoOfAxles, AirDragCoeff, FrontalArea(m²), Length(m), GrossWeight(t), TareWeight(t), Type`
Current config: 1 loco (5000 kW, 90 t, 4 axles) + 4 cars (50 t each) = ~290 t total.
Locomotive default max speed: 33.33 m/s (100/3); route speed limits are the binding constraint.

## Architecture

### C++ Simulation Engine

**Entry point:** `src/NeTrainSimConsole/main.cpp` — QCommandLineParser, calls
`SimulatorAPI::ContinuousMode::createNewSimulationEnvironmentFromFiles()`, then
`sim->runSimulation()` (blocking).

**Core classes:**
- **`Simulator`** (`simulator.h/cpp`) — time-step loop: `runSimulation()` calls
  `runOneTimeStep()` → `playTrainOneTimeStep()` → writes one CSV row per train per step.
  Exposes `pauseSimulation()` / `resumeSimulation()` / `runOneTimeStep()` (Q_INVOKABLE).
- **`Network`** (`network/network.h`) — nodes + links + signals; train route is fixed.
- **`Train`** (`traindefinition/train.h`) — `getStepAcceleration()` → `moveTrain()`;
  holds `optimumThrottleLevels` queue (built-in optimizer injection point).
- **`Locomotive`** (`locomotive.h`) — `currentLocNotch` (int 0–8), `throttleLevel`
  (double 0–1). Notch-throttle mapping: quadratic `(N/8)^2`.

**Trajectory CSV columns** (written per timestep when `-e` flag used):
```
TrainNo, TStep_s, TravelledDistance_m, Acceleration_mps2, Speed_mps,
LinkMaxSpeed_mps, EnergyConsumption_KWH, DelayTimeToEach_s, DelayTime_s,
Stoppings, tractiveForce_N, ResistanceForces_N, CurrentUsedTractivePower_kw,
GradeAtTip_Perc, CurvatureAtTip_Perc, FirstLocoNotchPosition, optimizationEnabled
```
`EnergyConsumption_KWH` is the cumulative total up to that timestep.

### Integration Strategy

**Phase 1 (implemented — no C++ changes):**
- `NeTrainSimEnv.reset()` runs the full A→B simulation via `subprocess.run()`.
- The trajectory CSV is loaded into memory; `step()` advances one row per call.
- Actions are recorded in `info["notch"]` but do NOT affect the simulation
  (physics is pre-computed). Tests the full Gymnasium/Tianshou pipeline.
- Episode length: ~6,700 steps (74.87 km at ~11 m/s average).

**Phase 2 (TODO — requires C++ change):**
- Add `--interactive` flag to `main.cpp`; the binary loops: run one timestep →
  write JSON state to stdout → read JSON action from stdin → set throttle → repeat.
- Python `step()` sends `{"notch": N}` and reads the response.
- This enables real per-step RL control.

**stdout state JSON (simulator → Python, Phase 2):**
```json
{
  "timestep": 42, "speed_mps": 15.3, "position_m": 1230.0,
  "grade_perc": 1.2, "curvature_perc": 0.0, "remaining_dist_m": 73639.6,
  "energy_kwh": 12.4, "link_max_speed_mps": 11.1, "terminated": false
}
```
**stdin action JSON (Python → simulator, Phase 2):** `{"notch": 6}`

### Gymnasium Environment (`rl/train_env.py`)

**Observation space** (7 floats, `Box`):
```
[speed_mps, position_m, grade_perc, curvature_perc,
 remaining_dist_m, energy_kwh, link_max_speed_mps]
```

**Action space:** `Discrete(9)` — notch 0–8 (maps to locomotive `currentLocNotch`).

**Reward (per step):**
```
r = -delta_energy_kwh
  - 0.1 * max(0, speed_mps - link_max_speed_mps)   # speed limit penalty
  + 100.0 on terminated=True (train arrived)         # arrival bonus
  - 50.0  on truncated=True  (trajectory ended early) # timeout penalty
```

**Episode boundaries:**
- `terminated=True`: `position_m >= TOTAL_ROUTE_LENGTH_M` (74,869.6 m)
- `truncated=True`: trajectory CSV exhausted without reaching destination

### Tianshou REINFORCE (`rl/train.py`) — tianshou 0.5.1 API

```python
from tianshou.policy import PGPolicy          # not tianshou.algorithm.*
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor  # use Actor, not DiscreteActor

net   = Net(state_shape=(7,), hidden_sizes=[128, 64], device="cpu")
actor = Actor(net, action_shape=9, softmax_output=True, device="cpu")
policy = PGPolicy(
    model=actor, optim=...,
    dist_fn=torch.distributions.Categorical,
    discount_factor=0.99,
    reward_normalization=True,
)
trainer = OnpolicyTrainer(
    policy, train_collector, test_collector,
    max_epoch=200, step_per_epoch=10_000,
    episode_per_collect=1,   # collect full episodes (on-policy requirement)
    repeat_per_collect=4, episode_per_test=1, batch_size=512,
)
```
**Important tianshou 0.5.1 notes:**
- Use `episode_per_collect` (not `step_per_collect`) for episodic envs — REINFORCE needs complete episodes before each update.
- `reward_normalization=True` normalizes discounted returns across the batch; helps with REINFORCE's high-variance gradients.
- Switch to `step_per_collect` + PPO if training is too slow (PPO can update mid-episode).

## RL Design Decisions

| Dimension | Choice | Rationale |
|-----------|--------|-----------|
| Action space | `Discrete(9)` notch 0–8 | Matches simulator internals; Categorical is lower-variance than Normal for REINFORCE |
| State space | 7 features from CSV | All per-step physics state the policy needs |
| Reward | Dense per-step energy delta + terminal bonus | Sparse reward makes REINFORCE too slow |
| Algorithm | REINFORCE (`PGPolicy`) | Start here; switch to PPO if gradient variance too high |
| Parallelism | `DummyVectorEnv(n=1)` | Each env = one C++ subprocess; scale with `SubprocVectorEnv` if needed |
| Phase | Phase 1 (episodic) now, Phase 2 (interactive) when C++ is ready | |

## C++ Modifications Required (Phase 2)

**Files to modify:**

1. `src/NeTrainSimConsole/main.cpp`
   - Add `QCommandLineOption interactiveOption({"I", "interactive"}, "Interactive RL mode")`
   - After `createNewSimulationEnvironmentFromFiles()`, if `--interactive`: run the JSON loop
     instead of calling `sim->runSimulation()`

2. `src/NeTrainSim/simulator.cpp`
   - Add `runInteractiveLoop()`: calls `runOneTimeStep()`, serializes train state to JSON
     on stdout, reads action JSON from stdin, sets `train->locomotives[0]->throttleLevel`
     before next step

**Injection point:** `locomotive->throttleLevel` (double 0–1) directly before each
`runOneTimeStep()` call. The quadratic notch mapping happens internally. Alternatively,
pre-populate `train->optimumThrottleLevels` queue (the built-in optimizer's path).

## Key Data Facts

- Route: 74.87 km, 750 nodes, 749 links, ~100 m per segment
- Locomotive max speed: 14.8645 m/s (≈53.5 km/h)
- Speed limits: 1.0, 3.0, 11.1, 16.6, 19.4, 22.2 m/s (3.6, 10.8, 40, 60, 70, 80 km/h)
- Episode length at 11.1 m/s average: ~6,744 simulator steps (1 step = 1 second)
- NeTrainSim speed unit in .dat files: **m/s** (confirmed in netlink.cpp)
- NeTrainSim .dat files: ASCII text, tab-separated, NOT binary

## Testing & Validation

```bash
# 1. Verify data generation
python data/generate_netrainsim_input.py

# 2. Verify simulator runs with our data (run from inside NeTrainSim-adjusted/)
cd NeTrainSim-adjusted && mkdir -p res
./build-mac/src/NeTrainSimConsole/NeTrainSim \
  -n ../data/netrainsim/nodesFile.dat \
  -l ../data/netrainsim/linksFile.dat \
  -t ../data/netrainsim/trainsFile.dat \
  -o res -e true -p 1.0        # note: -e takes 'true', not a bare flag
ls res/trainTrajectory_*.csv   # must show a CSV with ~7100 rows

# 3. Validate Gymnasium env
python -c "
import sys; sys.path.insert(0,'.')
from rl.train_env import NeTrainSimEnv
from gymnasium.utils.env_checker import check_env
check_env(NeTrainSimEnv())
print('OK')
"

# 4. Run training
python rl/train.py
```
