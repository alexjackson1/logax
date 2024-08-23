import logging
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random

from evojax.policy.base import PolicyNetwork
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from evojax.util import get_params_format_fn

from models import FullyConnectedNetwork, NeuralLogicNetwork


class NeuralLogicPolicy(PolicyNetwork):
    def __init__(
        self,
        depth: int,
        width: int,
        nnf: bool = False,
        in_features: int = 2,
        logger: logging.Logger = None,
    ):
        if logger is None:
            self._logger = create_logger("NeuralLogicPolicy")
        else:
            self._logger = logger

        model = NeuralLogicNetwork(depth, width, nnf)
        params = model.init(random.PRNGKey(0), jnp.zeros([1, in_features]))
        self.num_params, format_params_fn = get_params_format_fn(params)

        self._logger.info("NeuralLogicPolicy.num_params = {}".format(self.num_params))
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)

    def get_actions(
        self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState
    ) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states


class FullyConnectedPolicy(PolicyNetwork):
    def __init__(
        self,
        depth: int,
        width: int,
        dropout: float = 0.2,
        in_features: int = 2,
        logger: logging.Logger = None,
    ):
        if logger is None:
            self._logger = create_logger("FullyConnectedPolicy")
        else:
            self._logger = logger

        model = FullyConnectedNetwork(depth, width, dropout)
        params = model.init(random.PRNGKey(0), jnp.zeros([1, in_features]))
        self.num_params, format_params_fn = get_params_format_fn(params)

        self._logger.info(
            "FullyConnectedPolicy.num_params = {}".format(self.num_params)
        )
        self._format_params_fn = jax.vmap(format_params_fn)
        self._forward_fn = jax.vmap(model.apply)

    def get_actions(
        self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState
    ) -> Tuple[jnp.ndarray, PolicyState]:
        params = self._format_params_fn(params)
        return self._forward_fn(params, t_states.obs), p_states
