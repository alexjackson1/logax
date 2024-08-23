# Hyperparameters
# Optimiser:
#  - Evolutionary optimisation with PGPE and Adam
#  - Traditional gradient descent with SGD
#  - Traditional gradient descent with Adam

# Models:
#  - NeuralLogicNetwork(D, W, NNF)
#  - FullyConnectedNetwork(D, W, 0.2)
#  (Where D in [2, 3, 4] and W in [5, 25, 50, 100])

import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import os
import pickle
import shutil

from beartype.typing import Literal
import jax

import jax.numpy as jnp

from evojax import util, Trainer
from evojax.algo import PGPE
import optax

from evo.policy import FullyConnectedPolicy, NeuralLogicPolicy
from evo.task import MoralMachineTask
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
        policy = NeuralLogicPolicy(
            args.depth, args.width, logger=logger, in_features=1240
        )
    elif args.model == "fc":
        policy = FullyConnectedPolicy(
            args.depth, args.width, logger=logger, in_features=1240
        )

    train_task = MoralMachineTask(batch_size=args.batch_size, test=False)
    test_task = MoralMachineTask(batch_size=args.batch_size, test=True)

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


def optimise_gd(args: Arguments, optimiser: optax.GradientTransformation):
    log_dir = "./log/logic/sgd"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = util.create_logger(name="LL", log_dir=log_dir, debug=args.debug)
    logger.info("MM NLRL SGD Optimisation")
    logger.info("=" * 30)

    if args.model in ["ao", "an"]:
        model = NeuralLogicNetwork(args.depth, args.width, nnf=args.model == "an")
    elif args.model == "fc":
        model = FullyConnectedNetwork(args.depth, args.width, 0.2)

    rng = jax.random.PRNGKey(0)

    with open("../mm/train_inputs.pkl", "rb") as f:
        train_inputs = jnp.array(pickle.load(f), jnp.float32)
    with open("../mm/train_outputs.pkl", "rb") as f:
        train_targets = jnp.array(pickle.load(f), jnp.float32).reshape(-1, 1)
    with open("../mm/test_inputs.pkl", "rb") as f:
        test_inputs = jnp.array(pickle.load(f), jnp.float32)
    with open("../mm/test_outputs.pkl", "rb") as f:
        test_targets = jnp.array(pickle.load(f), jnp.float32).reshape(-1, 1)

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
    initial_params = model.init(init_rng, jnp.ones((1, train_inputs.shape[1])))

    final_state = train.train2(
        model,
        initial_params["params"],
        train_data,
        test_data,
        tx=optimiser,
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
        optimise_gd(args, optax.sgd(args.learning_rate))
    elif args.optimiser == "adam":
        optimise_gd(args, optax.adam(args.learning_rate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimiser", type=str, default="evo")
    parser.add_argument("--model", type=str, default="ao")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--pop_size", type=int, default=64)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--test_interval", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
