import pickle
from typing import Callable, Dict, Iterable, Tuple
from beartype import beartype
from beartype.typing import Literal
from jaxtyping import PRNGKeyArray, Float, Array


import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import itertools
from evojax import util

from models import NeuralLogicNetwork


@beartype
def mse_loss(
    predictions: Float[Array, "b 1"], targets: Float[Array, "b 1"]
) -> Float[Array, ""]:
    return jnp.mean(jnp.square(predictions - targets))


@beartype
def accuracy(
    predictions: Float[Array, "b 1"],
    targets: Float[Array, "b 1"],
    threshold: float = 0.005,
) -> Float[Array, ""]:
    return jnp.mean(jnp.abs(predictions - targets) < threshold)


@beartype
def accuracy2(
    predictions: Float[Array, "b 1"],
    targets: Float[Array, "b 1"],
) -> Float[Array, ""]:
    # jnp.round(predictions) and compare with targets
    return jnp.mean(jnp.round(predictions) == targets)


def train_step(
    state: train_state.TrainState,
    batch: Dict[Literal["train_inputs", "train_targets"], Array],
) -> Tuple[train_state.TrainState, Float[Array, "b 1"]]:
    def loss_fn(params):
        preds = state.apply_fn({"params": params}, batch["inputs"])
        loss = mse_loss(preds, batch["targets"])
        return loss

    grads = jax.grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state


train_step: Callable[
    [train_state.TrainState, Dict[Literal["train_inputs", "train_targets"], Array]],
    Tuple[train_state.TrainState, Float[Array, "b 1"]],
] = jax.jit(train_step)


def evaluate(
    state: train_state.TrainState,
    batch: Dict[Literal["train_inputs", "train_targets"], Array],
) -> Tuple[Float[Array, ""], Float[Array, ""]]:
    output = state.apply_fn({"params": state.params}, batch["inputs"])
    loss_value = mse_loss(output, batch["targets"])
    accuracy_value = accuracy(output, batch["targets"])
    return loss_value, accuracy_value


def evaluate2(
    state: train_state.TrainState,
    batch: Dict[Literal["train_inputs", "train_targets"], Array],
) -> Tuple[Float[Array, ""], Float[Array, ""]]:
    output = state.apply_fn({"params": state.params}, batch["inputs"])
    loss_value = mse_loss(output, batch["targets"])
    accuracy_value = accuracy2(output, batch["targets"])
    return loss_value, accuracy_value


def evaluate(
    state: train_state.TrainState,
    batch: Dict[Literal["train_inputs", "train_targets"], Array],
) -> Tuple[Float[Array, ""], Float[Array, ""]]:
    output = state.apply_fn({"params": state.params}, batch["inputs"])
    loss_value = mse_loss(output, batch["targets"])
    accuracy_value = accuracy(output, batch["targets"])
    return loss_value, accuracy_value


evaluate: Callable[
    [train_state.TrainState, Dict[Literal["train_inputs", "train_targets"], Array]],
    Tuple[Float[Array, ""], Float[Array, ""]],
] = jax.jit(evaluate)


# Training loop
def train(
    model: nn.Module,
    initial_params: optax.Params,
    train_data: Iterable[Dict[Literal["train_inputs", "train_targets"], Array]],
    test_data: Dict[Literal["train_inputs", "train_targets"], Array],
    tx: optax.GradientTransformation,
    max_iter: int,
    log_every: int,
    test_every: int,
    logger=None,
):
    if logger is None:
        _logger = util.create_logger("NN Training")
    else:
        _logger = logger

    # Initialize the optimizer (SGD)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=initial_params, tx=tx
    )

    batch_count = 1

    for i, batch in enumerate(itertools.cycle(train_data)):
        if batch_count > max_iter:
            break

        # Adjust learning rate every 20 epochs
        # if epoch % 20 == 0 and epoch > 0:
        #     learning_rate /= 10
        #     state = state.replace(tx=optax.adam(learning_rate))

        state = train_step(state, batch)
        batch_count += 1

        if batch_count % log_every == 0:
            loss_value, accuracy_value = evaluate(state, batch)
            _logger.info(
                f"Iter={batch_count}, loss={loss_value:.4f}, acc={accuracy_value:.2%}"
            )

        if batch_count % test_every == 0:
            test_loss, test_accuracy = evaluate(state, test_data)
            _logger.info(
                f"[TEST] Iter={batch_count}, loss={test_loss:.4f}, acc={test_accuracy:.2%}"
            )

        if i >= max_iter:
            break

    return state


# Training loop
def train2(
    model: nn.Module,
    initial_params: optax.Params,
    train_data: Iterable[Dict[Literal["train_inputs", "train_targets"], Array]],
    test_data: Dict[Literal["train_inputs", "train_targets"], Array],
    tx: optax.GradientTransformation,
    max_iter: int,
    log_every: int,
    test_every: int,
    logger=None,
):
    if logger is None:
        _logger = util.create_logger("NN Training")
    else:
        _logger = logger

    # Initialize the optimizer (SGD)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=initial_params, tx=tx
    )

    batch_count = 1

    for i, batch in enumerate(itertools.cycle(train_data)):
        if batch_count > max_iter:
            break

        # Adjust learning rate every 20 epochs
        # if epoch % 20 == 0 and epoch > 0:
        #     learning_rate /= 10
        #     state = state.replace(tx=optax.adam(learning_rate))

        state = train_step(state, batch)
        batch_count += 1

        if batch_count % log_every == 0:
            loss_value, accuracy_value = evaluate2(state, batch)
            _logger.info(
                f"Iter={batch_count}, loss={loss_value:.4f}, acc={accuracy_value:.2%}"
            )

        if batch_count % test_every == 0:
            test_loss, test_accuracy = evaluate2(state, test_data)
            _logger.info(
                f"[TEST] Iter={batch_count}, loss={test_loss:.4f}, acc={test_accuracy:.2%}"
            )

        if i >= max_iter:
            break

    return state


# Example usage
if __name__ == "__main__":
    # Assuming a toy dataset
    rng = jax.random.PRNGKey(0)

    with open("data/dataset.pkl", "rb") as f:
        data = pickle.load(f)

    train_inputs = jnp.array(data["train"][0])
    train_targets = jnp.array(data["train"][1]["x ⊕ y"])

    test_inputs = jnp.array(data["test"][0])
    test_targets = jnp.array(data["test"][1]["x ⊕ y"])

    # split the training dataset into batches
    batch_size = 20
    train_data = [
        {
            "inputs": train_inputs[i : i + batch_size],
            "targets": train_targets[i : i + batch_size],
        }
        for i in range(0, len(train_inputs), batch_size)
    ]
    test_data = {"inputs": test_inputs, "targets": test_targets}

    model = NeuralLogicNetwork(depth=4, width=5, nnf=False)
    rng, init_rng = jax.random.split(rng)
    initial_params = model.init(init_rng, jnp.ones((1, 2)))

    max_iter = 10000
    log_every = 10
    test_every = 100
    learning_rate = 0.1

    tx = optax.adam(learning_rate)

    final_state = train(
        model,
        initial_params["params"],
        train_data,
        test_data,
        tx,
        max_iter,
        log_every,
        test_every,
    )
