from typing import Callable
from beartype import beartype
from beartype.typing import Tuple, Dict, TypedDict, Literal
from jaxtyping import Float, Array, PRNGKeyArray

import argparse
import pickle

import jax.numpy as jnp
import jax.random as jr


@beartype
def l_and(x: Float[Array, "..."], y: Float[Array, "..."]) -> Float[Array, "..."]:
    """Algebraic conjunction (x ∧ y)."""
    return x * y


@beartype
def l_or(x: Float[Array, "..."], y: Float[Array, "..."]) -> Float[Array, "..."]:
    """Algebraic disjunction (x ∨ y)."""
    return x + y - x * y


@beartype
def l_not(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Algebraic negation (≠g x)."""
    return 1 - x


@beartype
def l_mi(x: Float[Array, "..."], y: Float[Array, "..."]) -> Float[Array, "..."]:
    """Algebraic material implication (x → y)."""
    return l_or(l_not(x), y)


@beartype
def l_xor(x: Float[Array, "..."], y: Float[Array, "..."]) -> Float[Array, "..."]:
    """Algebraic exclusive or (x ⊕ y)."""
    return l_and(l_or(x, y), l_not(l_and(x, y)))


@beartype
def l_eq(x: Float[Array, "..."], y: Float[Array, "..."]) -> Float[Array, "..."]:
    """Algebraic equality (x ≣ y)."""
    return l_not(l_xor(x, y))


FnName = Literal[
    "x ∧ y", "x ∨ y", "≠g x", "≠g y", "x ⊕ y", "(x + y)/2", "x·≠g y", "x", "y", "0.7"
]

FnSignature = Callable[[Float[Array, "n"], Float[Array, "n"]], Float[Array, "n"]]

ALL_FNS: Dict[FnName, FnSignature] = {
    "x ∧ y": l_and,
    "x ∨ y": l_or,
    "≠g x": lambda x, _: l_not(x),
    "≠g y": lambda _, y: l_not(y),
    "x ⊕ y": l_xor,
    "(x + y)/2": lambda x, y: (x + y) / 2,
    "x·≠g y": lambda x, y: x * l_not(y),
    "x": lambda x, _: x,
    "y": lambda _, y: y,
    "0.7": lambda x, _: jnp.full(x.shape, 0.7),
}

FN_NAMES = list(ALL_FNS.keys())


@beartype
def generate_samples(rng: PRNGKeyArray, n: int) -> Float[Array, "n 2"]:
    """Generate n pairs of samples."""
    return jr.uniform(rng, (n, 2))


@beartype
def augment_samples(
    rng: PRNGKeyArray, samples: Float[Array, "n 2"], amplitude: float
) -> Float[Array, "n 2"]:
    """Add Gaussian noise to the samples."""
    return samples + jr.normal(rng, samples.shape) * amplitude


@beartype
def compute_fns(samples: Float[Array, "n 2"]) -> Dict[FnName, Float[Array, "n 1"]]:
    """Compute the results of all functions on the pair of samples."""
    results = {}
    for fn_name, fn in ALL_FNS.items():
        results[fn_name] = fn(samples[:, 0], samples[:, 1]).reshape(-1, 1)
    return results


@beartype
class Dataset(TypedDict):
    train: Tuple[Float[Array, "n 2"], Dict[FnName, Float[Array, "n 1"]]]
    test: Tuple[Float[Array, "n 2"], Dict[FnName, Float[Array, "n 1"]]]


@beartype
def make_dataset(rng: PRNGKeyArray, samples: Tuple[int, int], noise: float) -> Dataset:
    """Generate a dataset of samples and their corresponding targets."""

    rng_train, rng_noise, rng_test = jr.split(rng, 3)

    x_pre_train = generate_samples(rng_train, samples[0])
    x_train = augment_samples(rng_noise, x_pre_train, noise)

    x_test = generate_samples(rng_test, samples[1])

    y_train, y_test = compute_fns(x_pre_train), compute_fns(x_test)

    return Dataset(train=(x_train, y_train), test=(x_test, y_test))


if __name__ == "__main__":
    import argparse

    class Arguments(argparse.Namespace):
        seed: int

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args(namespace=Arguments())

    rng = jr.PRNGKey(args.seed)
    dataset = make_dataset(rng, (10_000, 1_000), 0.0025)

    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    print("Dataset saved.")
