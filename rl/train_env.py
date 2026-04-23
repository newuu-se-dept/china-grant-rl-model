"""
Gymnasium environment wrapping NeTrainSim for train energy optimization.

Phase 1 (current — no C++ changes required):
  reset() runs the full A→B simulation via subprocess, loads the trajectory CSV.
  step() advances through the CSV one row per call; the action is RECORDED in
  info["notch"] but does NOT feed back to the simulator — the physics is fixed.
  This lets you test the full Gymnasium/Tianshou pipeline end-to-end and verify
  that reward signals are sensible before Phase 2 is ready.

Phase 2 (TODO — requires C++ --interactive flag in main.cpp):
  The simulator runs one timestep per step() call, reads action from stdin,
  writes state to stdout. Replace _run_simulation() + _advance() with the
  interactive subprocess protocol described in CLAUDE.md.

Observation space (7 floats):
  [speed_mps, position_m, grade_perc, curvature_perc,
   remaining_dist_m, energy_kwh, link_max_speed_mps]

Action space: Discrete(9) — notch 0-8 (maps to locomotive currentLocNotch)
"""

import csv
import os
import subprocess
import tempfile

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIMULATOR_BIN = os.path.join(
    _REPO, "NeTrainSim-adjusted", "build-mac",
    "src", "NeTrainSimConsole", "NeTrainSim"
)
NODES_FILE  = os.path.join(_REPO, "data", "netrainsim", "nodesFile.dat")
LINKS_FILE  = os.path.join(_REPO, "data", "netrainsim", "linksFile.dat")
TRAINS_FILE = os.path.join(_REPO, "data", "netrainsim", "trainsFile.dat")

TOTAL_ROUTE_LENGTH_M = 74_869.6  # computed from coordinates.csv


class NeTrainSimEnv(gym.Env):
    metadata = {"render_modes": []}

    observation_space = Box(
        low=np.array([0, 0, -15, -10, 0, 0, 0], dtype=np.float32),
        high=np.array([100, TOTAL_ROUTE_LENGTH_M + 500, 15, 10,
                       TOTAL_ROUTE_LENGTH_M + 500, 1e5, 100], dtype=np.float32),
    )
    action_space = Discrete(9)  # notch 0-8

    def __init__(self):
        super().__init__()
        self._trajectory: list[dict] = []
        self._step_idx: int = 0
        self._prev_energy: float = 0.0

        if not os.path.isfile(SIMULATOR_BIN):
            raise FileNotFoundError(
                f"Simulator binary not found: {SIMULATOR_BIN}\n"
                "Run: cd NeTrainSim-adjusted && ./build-mac.sh"
            )
        if not os.path.isfile(NODES_FILE):
            raise FileNotFoundError(
                f"NeTrainSim input files not found in data/netrainsim/.\n"
                "Run: python data/generate_netrainsim_input.py"
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._trajectory = self._run_simulation()
        self._step_idx = 0
        self._prev_energy = 0.0
        obs = self._row_to_obs(self._trajectory[0])
        return obs, {"total_steps": len(self._trajectory)}

    def step(self, action: int):
        self._step_idx = min(self._step_idx + 1, len(self._trajectory) - 1)
        row = self._trajectory[self._step_idx]

        obs = self._row_to_obs(row)
        energy_kwh   = float(row["EnergyConsumption_KWH"])
        speed_mps    = float(row["Speed_mps"])
        max_speed    = float(row["LinkMaxSpeed_mps"])
        position_m   = float(row["TravelledDistance_m"])

        delta_energy  = energy_kwh - self._prev_energy
        self._prev_energy = energy_kwh

        speed_penalty = 0.1 * max(0.0, speed_mps - max_speed)
        reward = -delta_energy - speed_penalty

        terminated = position_m >= TOTAL_ROUTE_LENGTH_M
        truncated  = self._step_idx >= len(self._trajectory) - 1 and not terminated

        if terminated:
            reward += 100.0
        elif truncated:
            reward -= 50.0

        info = {
            "notch":      int(action),
            "energy_kwh": energy_kwh,
            "speed_mps":  speed_mps,
            "position_m": position_m,
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        pass

    # ── Internal helpers ────────────────────────────────────────────────────

    def _run_simulation(self) -> list[dict]:
        with tempfile.TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                [SIMULATOR_BIN,
                 "-n", NODES_FILE,
                 "-l", LINKS_FILE,
                 "-t", TRAINS_FILE,
                 "-o", tmpdir,
                 "-e", "true",  # enable trajectory CSV export (-e takes a value, not a bare flag)
                 "-p", "1.0", # 1-second timestep
                 ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Simulator exited with code {proc.returncode}.\n"
                    f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
                )
            csv_files = [
                f for f in os.listdir(tmpdir)
                if f.startswith("trainTrajectory") and f.endswith(".csv")
            ]
            if not csv_files:
                raise RuntimeError(
                    "Simulator produced no trajectory CSV.\n"
                    f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
                )
            trajectory: list[dict] = []
            with open(os.path.join(tmpdir, csv_files[0])) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trajectory.append(dict(row))

        if not trajectory:
            raise RuntimeError("Trajectory CSV is empty — check simulator output.")
        return trajectory

    def _row_to_obs(self, row: dict) -> np.ndarray:
        position   = float(row["TravelledDistance_m"])
        remaining  = max(0.0, TOTAL_ROUTE_LENGTH_M - position)
        return np.array([
            float(row["Speed_mps"]),
            position,
            float(row["GradeAtTip_Perc"]),
            float(row["CurvatureAtTip_Perc"]),
            remaining,
            float(row["EnergyConsumption_KWH"]),
            float(row["LinkMaxSpeed_mps"]),
        ], dtype=np.float32)
