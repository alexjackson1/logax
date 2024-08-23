from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Float, Array

import pickle

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass

from evojax.task.base import VectorizedTask
from evojax.task.base import TaskState


@dataclass
class State(TaskState):
    obs: Array
    labels: Array


@beartype
def sample_batch(
    key: Array,
    data: Float[Array, "n in"],
    labels: Float[Array, "n 1"],
    batch_size: int,
) -> Tuple[Float[Array, "b in"], Float[Array, "b 1"]]:
    ix = random.choice(key=key, a=data.shape[0], shape=(batch_size,), replace=False)
    return (jnp.take(data, indices=ix, axis=0), jnp.take(labels, indices=ix, axis=0))


@beartype
def loss(
    prediction: Float[Array, "b 1"], target: Float[Array, "b 1"]
) -> Float[Array, ""]:
    return jnp.mean((prediction - target) ** 2)


@beartype
def accuracy(
    prediction: Float[Array, "b 1"], target: Float[Array, "b 1"]
) -> Float[Array, ""]:
    return jnp.mean(jnp.round(prediction) == target)


@beartype
class LogicLearningTask(VectorizedTask):
    """Logic learning tasks."""

    def __init__(self, batch_size: int = 10, fn: str = "x ∧ y", test: bool = False):
        self.max_steps = 1
        self.obs_shape = tuple([2])
        self.act_shape = tuple([1])

        with open("data/dataset.pkl", "rb") as f:
            data = pickle.load(f)

        d_train = jnp.array(data["train"][0]), jnp.array(data["train"][1][fn])
        d_test = jnp.array(data["test"][0]), jnp.array(data["test"][1][fn])

        def reset_fn(key: Array):
            X, y = d_test if test else sample_batch(key, *d_train, batch_size)
            return State(obs=X, labels=y)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state: State, action):
            if test:
                reward = accuracy(action, state.labels)
            else:
                reward = -loss(action, state.labels)
            return state, reward, jnp.ones(())

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: Array) -> State:
        return self._reset_fn(key)

    def step(self, state: TaskState, action: Array) -> Tuple[TaskState, Array, Array]:
        return self._step_fn(state, action)


@beartype
class MoralMachineTask(VectorizedTask):
    """Moral machine tasks."""

    def __init__(self, batch_size: int = 20, test: bool = False):
        self.max_steps = 1
        self.obs_shape = tuple([2])
        self.act_shape = tuple([1])

        with open("../mm/train_inputs.pkl", "rb") as f:
            train_inputs = pickle.load(f)
        with open("../mm/train_outputs.pkl", "rb") as f:
            train_targets = pickle.load(f)
        with open("../mm/test_inputs.pkl", "rb") as f:
            test_inputs = pickle.load(f)
        with open("../mm/test_outputs.pkl", "rb") as f:
            test_targets = pickle.load(f)

        d_train = jnp.array(train_inputs, dtype=jnp.float_), jnp.array(
            train_targets, dtype=jnp.float_
        ).reshape(-1, 1)
        d_test = jnp.array(test_inputs, dtype=jnp.float_), jnp.array(
            test_targets, dtype=jnp.float_
        ).reshape(-1, 1)

        def reset_fn(key: Array):
            X, y = d_test if test else sample_batch(key, *d_train, batch_size)
            return State(obs=X, labels=y)

        self._reset_fn = jax.jit(jax.vmap(reset_fn))

        def step_fn(state: State, action):
            if test:
                reward = accuracy(action, state.labels)
            else:
                reward = -loss(action, state.labels)
            return state, reward, jnp.ones(())

        self._step_fn = jax.jit(jax.vmap(step_fn))

    def reset(self, key: Array) -> State:
        return self._reset_fn(key)

    def step(self, state: TaskState, action: Array) -> Tuple[TaskState, Array, Array]:
        return self._step_fn(state, action)
