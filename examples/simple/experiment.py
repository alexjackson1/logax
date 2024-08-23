# Hyperparameters
# Optimiser:
#  - Evolutionary optimisation with PGPE and Adam
#  - Traditional gradient descent with SGD
#  - Traditional gradient descent with Adam

# Models:
#  - NeuralLogicNetwork(D, W, NNF)
#  - FullyConnectedNetwork(D, W, 0.2)
#  (Where D in [2, 3, 4] and W in [5, 25, 50, 100])


import argparse
import os
import pickle
import shutil

from beartype.typing import Literal
import jax
from jaxtyping import PRNGKeyArray, Float, Array

import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn

from evojax import util, Trainer
from evojax.algo import PGPE
import optax

from dataset import FnName
from evo.policy import FullyConnectedPolicy, NeuralLogicPolicy
from evo.task import LogicLearningTask
from models import NeuralLogicNetwork, FullyConnectedNetwork
import train


class Arguments(argparse.Namespace):
    optimiser: Literal["evo", "sgd", "adam"]
    model: Literal["ao", "an", "fc"]
    depth: int
    width: int
    seed: int
    debug: bool
    batch_size: int
    function: FnName
    pop_size: int
    max_iter: int
    log_interval: int
    test_interval: int
    learning_rate: float


def optimise_evo(args: Arguments):
    log_dir = "./log/logic/evo"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = util.create_logger(name="LL", log_dir=log_dir, debug=args.debug)
    logger.info("NLRL Evo Optimisation")
    logger.info("=" * 30)

    if args.model in ["ao", "an"]:
        policy = NeuralLogicPolicy(args.depth, args.width, logger=logger)
    elif args.model == "fc":
        policy = FullyConnectedPolicy(args.depth, args.width, logger=logger)

    train_task = LogicLearningTask(
        batch_size=args.batch_size, fn=args.function, test=False
    )
    test_task = LogicLearningTask(
        batch_size=args.batch_size, fn=args.function, test=True
    )

    solver = PGPE(
        pop_size=args.pop_size,
        param_size=policy.num_params,
        optimizer="adam",
        logger=logger,
        seed=args.seed,
    )

    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=args.max_iter,
        log_interval=args.log_interval,
        test_interval=args.test_interval,
        n_repeats=5,
        n_evaluations=1,
        seed=args.seed,
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    src_file = os.path.join(log_dir, "best.npz")
    tar_file = os.path.join(log_dir, "model.npz")
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)


def optimise_sgd(args: Arguments):
    log_dir = "./log/logic/sgd"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = util.create_logger(name="LL", log_dir=log_dir, debug=args.debug)
    logger.info("NLRL SGD Optimisation")
    logger.info("=" * 30)

    if args.model in ["ao", "an"]:
        model = NeuralLogicNetwork(args.depth, args.width, nnf=args.model == "an")
    elif args.model == "fc":
        model = FullyConnectedNetwork(args.depth, args.width, 0.2)

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

    rng, init_rng = jax.random.split(rng)
    initial_params = model.init(init_rng, jnp.ones((1, 2)))

    final_state = train.train(
        model,
        initial_params["params"],
        train_data,
        test_data,
        tx=optax.sgd(args.learning_rate),
        max_iter=args.max_iter,
        log_every=args.log_interval,
        test_every=args.test_interval,
        logger=logger,
    )

    return final_state


def optimise_adam(args: Arguments):
    log_dir = "./log/logic/sgd"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = util.create_logger(name="LL", log_dir=log_dir, debug=args.debug)
    logger.info("NLRL Adam Optimisation")
    logger.info("=" * 30)

    if args.model in ["ao", "an"]:
        model = NeuralLogicNetwork(args.depth, args.width, nnf=args.model == "an")
    elif args.model == "fc":
        model = FullyConnectedNetwork(args.depth, args.width, 0.2)

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

    rng, init_rng = jax.random.split(rng)
    initial_params = model.init(init_rng, jnp.ones((1, 2)))

    final_state = train.train(
        model,
        initial_params["params"],
        train_data,
        test_data,
        tx=optax.adam(args.learning_rate),
        max_iter=args.max_iter,
        log_every=args.log_interval,
        test_every=args.test_interval,
        logger=logger,
    )

    return final_state


def main(args: Arguments):
    if args.optimiser == "evo":
        optimise_evo(args)
    elif args.optimiser == "sgd":
        optimise_sgd(args)
    elif args.optimiser == "adam":
        optimise_adam(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimiser", type=str, default="evo")
    parser.add_argument("--model", type=str, default="ao")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--function", type=str, default="x ⊕ y")
    parser.add_argument("--pop_size", type=int, default=64)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--test_interval", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
