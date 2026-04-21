# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository combines two components:
1. **NeTrainSim** (`NeTrainSim-adjusted/`) — A C++/Qt6 freight train network simulator for modeling longitudinal dynamics and energy consumption
2. **Gymnasium docs** (`gymnasium-docs/`) — RL environment documentation and tutorials for training agents on train optimization problems

The goal is to integrate NeTrainSim as a Gymnasium-compatible RL environment for train energy optimization.

## Build & Run

### NeTrainSim (C++)

Prerequisites: `brew install qt cmake`

```bash
cd NeTrainSim-adjusted
./build-mac.sh          # builds and runs a sample simulation
```

The build script compiles with GUI and server disabled, then runs the console binary against sample data into `res/`.

Manual build:
```bash
cmake -B build-mac -DCMAKE_BUILD_TYPE=Release -DBUILD_GUI=OFF -DBUILD_SERVER=OFF -DCMAKE_PREFIX_PATH=$(brew --prefix qt6)
cmake --build build-mac --target NeTrainSimConsole -j$(sysctl -n hw.logicalcpu)
mkdir -p res
```

Run the simulator:
```bash
./build-mac/src/NeTrainSimConsole/NeTrainSim \
  -n src/data/sampleProject/nodesFile.dat \
  -l src/data/sampleProject/linksFile.dat \
  -t src/data/sampleProject/dieselTrain.dat \
  -o res
```

Key CLI flags: `-n` nodes file, `-l` links file, `-t` trains file, `-o` output dir, `-p` time step (default 1.0s), `-z` enable optimization, `-e` export trajectory.

### Gymnasium docs (Python)

```bash
cd gymnasium-docs
pip install gymnasium
pip install -r requirements.txt
make dirhtml                          # build once
sphinx-autobuild -b dirhtml --watch ../gymnasium --re-ignore "pickle$" . _build
# then visit http://localhost:8000
```

## Architecture

### C++ Simulation Engine

**Entry point:** `NeTrainSim-adjusted/src/NeTrainSimConsole/main.cpp` — parses CLI args via QCommandLineParser, constructs Network and Simulator, runs simulation, writes output files.

**Core classes:**

- **`Simulator`** (`simulator.h/cpp`) — main loop: advances time step by step, moves trains, resolves signal conflicts, generates trajectory/summary output files
- **`Network`** (`network/network.h`) — rail graph: nodes (`NetNode`), edges (`NetLink`), signals (`NetSignal`, `NetSignalGroupController`); populated from `.dat` files by `ReadWriteNetwork`
- **`Train`** (`traindefinition/train.h`) — largest class (~42K lines); holds a consist of locomotives and cars, computes traction forces, braking, speed profiles, and energy consumption per time step
- **`Locomotive`** / **`Car`** — train components with mass, drag, and powertrain properties; `Locomotive` drives traction
- **`EnergyConsumption`** — models diesel, electric, battery, and hybrid powertrains; called per time step inside `Train`

**Data flow:**
```
*.dat files → ReadWriteNetwork → Network + Train list
                                         ↓
                                    Simulator (time-step loop)
                                         ↓
                              trainSummary_*.txt + trainTrajectory_*.csv
```

**Output:** CSV trajectory (second-by-second position, speed, energy) and TXT summary per simulation run.

## RL Environment Design (project-specific)

The Gymnasium wrapper around NeTrainSim is the core deliverable. Key design constraints:

- **State space:** train speed (m/s), position along route (m), track grade (%), curvature, remaining distance, current energy level
- **Action space:** discrete or continuous throttle/brake commands fed back into the simulator's traction model
- **Reward:** minimize total energy consumption per trip; penalize schedule deviation and speed limit violations
- **Episode:** one full trip from origin to destination; terminated on arrival or derailment/timeout
- **Integration point:** wrap `NeTrainSimConsole` via subprocess or extract `Simulator`/`Train` logic into a shared library callable from Python
- **Preferred RL stack:** Tianshou (thu-ml/tianshou) or Stable-Baselines3; PPO is the default algorithm to try first
- **Output to parse:** `trainTrajectory_*.csv` — second-by-second position, speed, energy columns

## Key Data Files

Sample project lives in `NeTrainSim-adjusted/src/data/sampleProject/`:
- `nodesFile.dat` — node IDs and coordinates
- `linksFile.dat` — link segments between nodes
- `dieselTrain.dat` — train composition (locomotive + cars)

Output files are written to `res/` (created by build script).
