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
import select
import subprocess
import time

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import platform as _platform
_BUILD_DIR = "build-linux" if _platform.system() == "Linux" else "build-mac"
SIMULATOR_BIN = os.path.join(
    _REPO, "NeTrainSim-adjusted", _BUILD_DIR,
    "src", "NeTrainSimConsole", "NeTrainSim"
)
NODES_FILE  = os.path.join(_REPO, "data", "netrainsim_v2", "nodesFile_v2_fixed.dat")
LINKS_FILE  = os.path.join(_REPO, "data", "netrainsim_v2", "linksFile_v2_fixed.dat")
TRAINS_FILE = os.path.join(_REPO, "data", "netrainsim_v2", "trainsFile_rl.dat")

TOTAL_ROUTE_LENGTH_M = 74_891.29  # sum of all 1499 link lengths (linksFile_v2_fixed.dat)
STATE_PREFIX = "NTS_JSON "
MAX_STEPS    = 10_000  # hard ceiling above the nominal ~6,700-step trip
TARGET_STEPS = 4_500   # schedule target: trips longer than this incur a time penalty (4500s ≈ 75 min)

# Normalisation denominators for _state_to_obs
_SPEED_MAX     = 20.0   # ER9E max ≈ 19.4 m/s (links speed limit)
_GRADE_MAX     = 0.7    # route max ±0.628%
_ENERGY_MAX    = 0.25   # per-step energy cap in kWh (observed max ~0.2, headroom to 0.25)
_MAXSPEED_MAX  = 19.4   # ER9E route speed-limit max in m/s (70 km/h)

_LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")


class NeTrainSimEnv(gym.Env):
    metadata = {"render_modes": []}

    # All 7 features are normalized to roughly [-1, 1] or [0, 1] by _state_to_obs.
    observation_space = Box(
        low =np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([2.0, 1.0,  1.0,  1.0, 1.0, 1.0, 2.0], dtype=np.float32),
    )
    action_space = Discrete(9)  # notch 0-8

    def __init__(self):
        super().__init__()
        self._proc: subprocess.Popen | None = None
        self._stderr_log = None   # file handle for simulator stderr
        self._last_state: dict | None = None
        self._cum_energy_kwh: float = 0.0   # running total for logging/reward only
        self._step_count: int = 0
        self._episode_count: int = 0
        self._episode_start: float = 0.0
        self._episode_reward: float = 0.0

        if not os.path.isfile(SIMULATOR_BIN):
            raise FileNotFoundError(
                f"Simulator binary not found: {SIMULATOR_BIN}\n"
                "Run: cd NeTrainSim-adjusted && ./build-linux.sh  (or build-mac.sh on macOS)"
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
        self._cum_energy_kwh = 0.0
        self._episode_start = time.time()
        self._episode_count += 1
        self._start_interactive_simulator()
        state = self._send_action_and_read_state(0)
        self._last_state = state
        step_energy = float(state["energy_kwh"])
        self._cum_energy_kwh = step_energy  # include bootstrap step-0 energy
        # Bootstrap consumed one simulator timestep (notch=0 sent to get initial
        # state). Start _step_count at 1 so it tracks actual simulator steps.
        self._step_count = 1
        obs = self._state_to_obs(state, step_energy)
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

        # energy_kwh from the simulator is per-step energy (not cumulative).
        # EnergyConsumption_KWH in the CSV oscillates 0–0.2 kWh per step
        # depending on throttle; cumEnergyStat (total) reaches ~1,200 kWh.
        step_energy_kwh = float(state["energy_kwh"])
        self._cum_energy_kwh += step_energy_kwh

        obs = self._state_to_obs(state, step_energy_kwh)
        speed_mps = float(state["speed_mps"])
        max_speed = float(state["link_max_speed_mps"])
        position_m = float(state["position_m"])

        speed_penalty = 0.1 * max(0.0, speed_mps - max_speed)
        reward = -step_energy_kwh - speed_penalty

        terminated = bool(state["terminated"]) or position_m >= TOTAL_ROUTE_LENGTH_M
        truncated = self._step_count >= MAX_STEPS and not terminated

        if terminated:
            # Arrival bonus minus time penalty for trips slower than schedule target
            time_penalty = 0.05 * max(0.0, float(self._step_count - TARGET_STEPS))
            reward += 100.0 - time_penalty
        elif truncated:
            reward -= 50.0

        self._episode_reward += reward

        if terminated or truncated:
            elapsed = time.time() - self._episode_start
            pct = 100.0 * position_m / TOTAL_ROUTE_LENGTH_M
            status = "✓ ARRIVED" if terminated else "✗ TIMEOUT"
            print(
                f"[{status}]  ep={self._episode_count:>4d}"
                f"  {self._step_count:>5,} steps"
                f"  {position_m:>7,.0f}m ({pct:4.1f}%)"
                f"  energy={self._cum_energy_kwh:>7.1f} kWh"
                f"  reward={self._episode_reward:>+8.1f}"
                f"  {elapsed:.1f}s",
                flush=True,
            )

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
        if self._stderr_log is not None:
            try:
                self._stderr_log.close()
            except Exception:
                pass
            self._stderr_log = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ── Internal helpers ────────────────────────────────────────────────────

    def _start_interactive_simulator(self) -> None:
        os.makedirs(_LOGS_DIR, exist_ok=True)
        stderr_path = os.path.join(_LOGS_DIR, "netrainsim_stderr.log")
        self._stderr_log = open(stderr_path, "a")
        self._proc = subprocess.Popen(
            [SIMULATOR_BIN,
             "-n", NODES_FILE,
             "-l", LINKS_FILE,
             "-t", TRAINS_FILE,
             "-p", "1.0",
             "-I"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_log,
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

        _READLINE_TIMEOUT = 30.0
        while True:
            ready, _, _ = select.select([self._proc.stdout], [], [], _READLINE_TIMEOUT)
            if not ready:
                rc = self._proc.poll()
                raise RuntimeError(
                    f"Simulator stalled: no output for {_READLINE_TIMEOUT}s "
                    f"(returncode={rc}). Check logs/netrainsim_stderr.log"
                )
            line = self._proc.stdout.readline()
            if line == "":
                rc = self._proc.poll()
                raise RuntimeError(
                    f"Simulator terminated before returning state "
                    f"(returncode={rc}). Check logs/netrainsim_stderr.log"
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

    def _state_to_obs(self, state: dict, step_energy_kwh: float) -> np.ndarray:
        position = float(state["position_m"])
        remaining = max(0.0, TOTAL_ROUTE_LENGTH_M - position)
        obs = np.array([
            float(state["speed_mps"])          / _SPEED_MAX,
            position                           / TOTAL_ROUTE_LENGTH_M,
            float(state["grade_perc"])         / _GRADE_MAX,
            float(state["curvature_perc"]),
            remaining                          / TOTAL_ROUTE_LENGTH_M,
            step_energy_kwh                    / _ENERGY_MAX,
            float(state["link_max_speed_mps"]) / _MAXSPEED_MAX,
        ], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)
