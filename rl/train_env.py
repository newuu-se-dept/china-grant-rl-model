"""
Gymnasium environment wrapping NeTrainSim for train energy optimization.

Phase 2 (interactive):
  reset() starts the simulator in interactive mode, then sends notch=0 once to
  get the first timestep state.
  step(action) sends {"notch": N} to simulator stdin and reads the next state
  JSON from stdout (prefixed by "NTS_JSON ").

Observation space (7 floats):
  [speed_mps, position_m, grade_perc, curvature_perc,
   remaining_dist_m, energy_kwh, link_max_speed_mps]

Action space: Discrete(9) — notch 0-8 (maps to locomotive currentLocNotch)
"""

import json
import os
import subprocess
import time

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
STATE_PREFIX = "NTS_JSON "


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
        self._proc: subprocess.Popen | None = None
        self._last_state: dict | None = None
        self._prev_energy: float = 0.0
        self._step_count: int = 0
        self._episode_count: int = 0
        self._episode_start: float = 0.0
        self._episode_reward: float = 0.0

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
        self.close()
        self._step_count = 0
        self._episode_reward = 0.0
        self._episode_start = time.time()
        self._episode_count += 1
        self._start_interactive_simulator()
        state = self._send_action_and_read_state(0)
        self._last_state = state
        self._prev_energy = float(state["energy_kwh"])
        obs = self._state_to_obs(state)
        # Return empty info dict — tianshou 0.5.1 Batch cannot create new keys
        # via index assignment, so any non-empty info dict causes a ValueError.
        return obs, {}

    def step(self, action: int):
        notch = int(action)
        if notch < 0 or notch > 8:
            raise ValueError(f"Action notch must be in [0, 8], got {notch}")

        state = self._send_action_and_read_state(notch)
        self._last_state = state
        self._step_count += 1

        obs = self._state_to_obs(state)
        energy_kwh = float(state["energy_kwh"])
        speed_mps = float(state["speed_mps"])
        max_speed = float(state["link_max_speed_mps"])
        position_m = float(state["position_m"])

        delta_energy  = energy_kwh - self._prev_energy
        self._prev_energy = energy_kwh

        speed_penalty = 0.1 * max(0.0, speed_mps - max_speed)
        reward = -delta_energy - speed_penalty

        terminated = bool(state["terminated"]) or position_m >= TOTAL_ROUTE_LENGTH_M
        truncated = False

        if terminated:
            reward += 100.0
        elif truncated:
            reward -= 50.0

        self._episode_reward += reward

        if self._step_count % 500 == 0:
            pct = 100.0 * position_m / TOTAL_ROUTE_LENGTH_M
            print(
                f"\r  ep={self._episode_count}  step={self._step_count:5d}"
                f"  pos={position_m:7.0f}m ({pct:4.1f}%)"
                f"  speed={speed_mps:5.2f}m/s  energy={energy_kwh:7.3f}kWh"
                f"  notch={notch}",
                end="", flush=True,
            )

        if terminated or truncated:
            elapsed = time.time() - self._episode_start
            status = "ARRIVED" if terminated else "TIMEOUT"
            line = (
                f"[{status}] ep={self._episode_count}"
                f"  steps={self._step_count}  dist={position_m:.0f}m"
                f"  energy={energy_kwh:.3f}kWh  reward={self._episode_reward:+.1f}"
                f"  time={elapsed:.1f}s"
            )
            # Pad to 80 chars to overwrite any leftover progress text on the line
            print(f"\r{line:<80}", flush=True)

        # Return empty info dict — tianshou 0.5.1 Batch cannot create new keys
        # via index assignment, so any non-empty info dict causes a ValueError.
        return obs, float(reward), terminated, truncated, {}

    def close(self):
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
            self._proc = None

    # ── Internal helpers ────────────────────────────────────────────────────

    def _start_interactive_simulator(self) -> None:
        self._proc = subprocess.Popen(
            [SIMULATOR_BIN,
             "-n", NODES_FILE,
             "-l", LINKS_FILE,
             "-t", TRAINS_FILE,
             "-p", "1.0",
             "-I"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _send_action_and_read_state(self, notch: int) -> dict:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Interactive simulator process is not running.")

        payload = json.dumps({"notch": int(notch)})
        try:
            self._proc.stdin.write(payload + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError("Simulator stdin pipe is closed.") from exc

        while True:
            line = self._proc.stdout.readline()
            if line == "":
                rc = self._proc.poll()
                raise RuntimeError(
                    f"Simulator terminated before returning state. returncode={rc}"
                )
            line = line.strip()
            if not line.startswith(STATE_PREFIX):
                continue
            raw_json = line[len(STATE_PREFIX):]
            try:
                state = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid simulator state JSON: {raw_json}") from exc
            return state

    def _state_to_obs(self, state: dict) -> np.ndarray:
        position = float(state["position_m"])
        remaining  = max(0.0, TOTAL_ROUTE_LENGTH_M - position)
        return np.array([
            float(state["speed_mps"]),
            position,
            float(state["grade_perc"]),
            float(state["curvature_perc"]),
            remaining,
            float(state["energy_kwh"]),
            float(state["link_max_speed_mps"]),
        ], dtype=np.float32)
