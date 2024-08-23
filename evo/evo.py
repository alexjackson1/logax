import argparse
import os
import shutil

from evojax import Trainer
from evojax.algo import PGPE
from evojax import util

from evo.task import LogicLearningTask
from evo.policy import NeuralLogicPolicy


class Config(argparse.Namespace):
    pop_size: int
    batch_size: int
    max_iter: int
    test_interval: int
    log_interval: int
    seed: int
    gpu_id: str
    debug: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pop-size",
        type=int,
        default=64,
        help="NE population size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5_000,
        help="Max training iterations.",
    )
    parser.add_argument(
        "--test-interval",
        type=int,
        default=1000,
        help="Test interval.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Logging interval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training.",
    )
    # parser.add_argument(
    #     "--center-lr", type=float, default=0.006, help="Center learning rate."
    # )
    # parser.add_argument(
    #     "--std-lr", type=float, default=0.089, help="Std learning rate."
    # )
    # parser.add_argument("--init-std", type=float, default=0.039, help="Initial std.")
    parser.add_argument("--gpu-id", type=str, help="GPU(s) to use.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    config, _ = parser.parse_known_args(namespace=Config())
    return config


def main(config: Config):
    log_dir = "./log/logic"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = util.create_logger(name="LL", log_dir=log_dir, debug=config.debug)
    logger.info("NLRL Optimisation")
    logger.info("=" * 30)

    policy = NeuralLogicPolicy(logger=logger)
    train_task = LogicLearningTask(batch_size=config.batch_size, fn="x ⊕ y", test=False)
    test_task = LogicLearningTask(batch_size=config.batch_size, fn="x ⊕ y", test=True)
    solver = PGPE(
        pop_size=config.pop_size,
        param_size=policy.num_params,
        optimizer="adam",
        # stdev_learning_rate=config.std_lr,
        # init_stdev=config.init_std,
        logger=logger,
        seed=config.seed,
    )

    # Train.
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=5,
        n_evaluations=1,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(log_dir, "best.npz")
    tar_file = os.path.join(log_dir, "model.npz")
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)


if __name__ == "__main__":
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu_id
    main(configs)
