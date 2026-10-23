"""M9: MuJoCo locomotion (HalfCheetah, Hopper) as an ES task, every policy stepped in lockstep.

The classic evolution-strategies benchmark (Salimans et al. 2017; Mania et al. 2018). One
compiled MjModel is shared by all rollouts, one MjData each. Observations, rewards and
termination follow gymnasium's v5 environments step for step (tests check this); only the
reset noise comes from our own generator, seeded by the step, so every worker scores every
policy on the same start states.

    uv run seedling train --task halfcheetah --steps 200 --pop 32 --sigma 0.05 --lr 0.05
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from seedling.es import BatchedMLP


@dataclass(frozen=True)
class Locomotion:
    """One gymnasium v5 locomotion environment, as numbers."""

    name: str
    xml: str
    frame_skip: int
    ctrl_cost_weight: float
    reset_noise: float
    gaussian_qvel_noise: bool  # HalfCheetah: qvel += scale * N(0, 1); Hopper: U(-scale, scale)
    healthy_reward: float = 0.0
    terminate_when_unhealthy: bool = False
    clip_qvel: float | None = None
    healthy_z: tuple[float, float] = (-np.inf, np.inf)
    healthy_angle: tuple[float, float] = (-np.inf, np.inf)
    healthy_state: tuple[float, float] = (-np.inf, np.inf)


HALFCHEETAH = Locomotion("halfcheetah", "half_cheetah.xml", 5, 0.1, 0.1, gaussian_qvel_noise=True)
HOPPER = Locomotion(
    "hopper", "hopper.xml", 4, 1e-3, 5e-3, gaussian_qvel_noise=False,
    healthy_reward=1.0, terminate_when_unhealthy=True, clip_qvel=10.0,
    healthy_z=(0.7, np.inf), healthy_angle=(-0.2, 0.2), healthy_state=(-100.0, 100.0),
)  # fmt: skip
ENVIRONMENTS = {e.name: e for e in (HALFCHEETAH, HOPPER)}


def _xml_path(name: str) -> str:
    import gymnasium.envs.mujoco

    return str(Path(gymnasium.envs.mujoco.__file__).parent / "assets" / name)


class Simulator:
    """Steps many copies of one environment. Not thread-safe: one per process."""

    def __init__(self, env: Locomotion):
        import mujoco

        self.mujoco = mujoco
        self.env = env
        self.model = mujoco.MjModel.from_xml_path(_xml_path(env.xml))
        self.dt = self.model.opt.timestep * env.frame_skip
        reference = mujoco.MjData(self.model)
        self.init_qpos = reference.qpos.copy()
        self.init_qvel = reference.qvel.copy()
        self._datas: list = []

    @property
    def obs_dim(self) -> int:
        return self.model.nq - 1 + self.model.nv

    @property
    def act_dim(self) -> int:
        return self.model.nu

    def datas(self, n: int) -> list:
        while len(self._datas) < n:
            self._datas.append(self.mujoco.MjData(self.model))
        return self._datas[:n]

    def start_state(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        scale = self.env.reset_noise
        qpos = self.init_qpos + rng.uniform(-scale, scale, size=self.model.nq)
        if self.env.gaussian_qvel_noise:
            qvel = self.init_qvel + scale * rng.standard_normal(self.model.nv)
        else:
            qvel = self.init_qvel + rng.uniform(-scale, scale, size=self.model.nv)
        return qpos, qvel

    def set_state(self, data, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.mujoco.mj_resetData(self.model, data)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        self.mujoco.mj_forward(self.model, data)

    def observe(self, data) -> np.ndarray:
        qvel = data.qvel if self.env.clip_qvel is None else np.clip(data.qvel, -self.env.clip_qvel, self.env.clip_qvel)
        return np.concatenate([data.qpos[1:], qvel])

    def healthy(self, data) -> bool:
        env = self.env
        z, angle = data.qpos[1], data.qpos[2]
        state = np.concatenate([data.qpos, data.qvel])[2:]
        return bool(
            env.healthy_z[0] < z < env.healthy_z[1]
            and env.healthy_angle[0] < angle < env.healthy_angle[1]
            and np.all((env.healthy_state[0] < state) & (state < env.healthy_state[1]))
        )

    def step(self, data, action: np.ndarray) -> tuple[float, bool]:
        """Advance one environment step; returns (reward, terminated), as gymnasium v5."""
        env = self.env
        x_before = data.qpos[0]
        data.ctrl[:] = action
        self.mujoco.mj_step(self.model, data, nstep=env.frame_skip)
        x_velocity = (data.qpos[0] - x_before) / self.dt
        healthy = self.healthy(data) if env.terminate_when_unhealthy or env.healthy_reward else True
        reward = x_velocity + env.healthy_reward * healthy - env.ctrl_cost_weight * float(np.sum(action**2))
        return reward, env.terminate_when_unhealthy and not healthy


class LocomotionTask:
    """A Task (seedling.tasks.base): a tanh MLP policy, fitness = mean return of its rollouts.

    Every policy of a step runs from the same `episodes` start states, drawn from the step
    number, for `horizon` environment steps. Observations are standardized with fixed
    statistics from seeded random-action rollouts, so the policy stays a pure function of
    the replicated parameters.
    """

    def __init__(self, env: str = "halfcheetah", hidden: int = 32, episodes: int = 1, horizon: int = 500):
        self.env = ENVIRONMENTS[env]
        self.name = self.env.name
        self.hidden = hidden
        self.episodes = episodes
        self.horizon = horizon
        sim = self.sim
        sizes = [sim.obs_dim, hidden, sim.act_dim] if hidden else [sim.obs_dim, sim.act_dim]
        self.policy = BatchedMLP(sizes)
        self.dim = self.policy.size

    @cached_property
    def sim(self) -> Simulator:
        return Simulator(self.env)

    @cached_property
    def normalizer(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std of observations under uniformly random actions (seeded)."""
        sim, rng = self.sim, np.random.default_rng(12345)
        data, observations = sim.datas(1)[0], []
        for _ in range(4):
            sim.set_state(data, *sim.start_state(rng))
            for _ in range(self.horizon):
                observations.append(sim.observe(data))
                if sim.step(data, rng.uniform(-1, 1, size=sim.act_dim))[1]:
                    break
        obs = np.array(observations)
        return obs.mean(axis=0), np.maximum(obs.std(axis=0), 1e-2)

    def init(self, seed: int) -> torch.Tensor:
        return self.policy.init(torch.Generator().manual_seed(seed))

    def returns(self, thetas: torch.Tensor, seed: int, episodes: int, horizon: int) -> torch.Tensor:
        """Mean return of each of the P policies in thetas (P, d) over the same start states."""
        sim = self.sim
        mean, std = self.normalizer
        p = thetas.shape[0]
        rng = np.random.default_rng([seed, 7])
        starts = [sim.start_state(rng) for _ in range(episodes)]
        datas = sim.datas(p * episodes)
        for k, data in enumerate(datas):
            sim.set_state(data, *starts[k % episodes])
        alive = np.ones(p * episodes, dtype=bool)
        total = np.zeros(p * episodes)
        obs = np.stack([sim.observe(d) for d in datas])
        for _ in range(horizon):
            x = torch.from_numpy(np.clip((obs - mean) / std, -10, 10)).float().view(p, episodes, -1)
            actions = torch.tanh(self.policy(thetas, x)).reshape(p * episodes, -1).double().numpy()
            for k in np.flatnonzero(alive):
                reward, terminated = sim.step(datas[k], actions[k])
                total[k] += reward
                obs[k] = sim.observe(datas[k])
                alive[k] = not terminated
            if not alive.any():
                break
        return torch.from_numpy(total.reshape(p, episodes).mean(axis=1))

    def fitness(self, thetas: torch.Tensor, step: int) -> torch.Tensor:
        return self.returns(thetas, seed=step, episodes=self.episodes, horizon=self.horizon)

    def evaluate(self, theta: torch.Tensor, step: int) -> dict[str, float]:
        """Return over 5 held-out starts at the standard horizon of 1000 steps."""
        score = self.returns(theta[None], seed=10**9 + step, episodes=5, horizon=1000)
        return {"return": float(score[0])}
