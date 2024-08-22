from beartype import beartype
from beartype.typing import Any, Sequence
from jaxtyping import Float, Array

import jax.numpy as jnp
from jax.nn import sigmoid
from jax import dtypes
from flax import linen as nn


@beartype
def uniform_range(min: float = -0.5, max: float = 0.5) -> nn.initializers.Initializer:
    """Initializer that generates arrays with values uniformly sampled from a range."""

    def init(key: Array, shape: Sequence[int], dtype: Any = jnp.float_) -> Array:
        dtype = dtypes.canonicalize_dtype(dtype)
        return random.uniform(key, shape, dtype, minval=min, maxval=max)

    return init


@beartype
class NeuralLogicRuleLayer(nn.Module):
    input_size: int
    output_size: int
    nnf: bool = False

    def setup(self):
        self.GN = self.param("GN", uniform_range(), (self.input_size, self.output_size))
        self.GR = self.param("GR", uniform_range(), (self.input_size, self.output_size))
        if not self.nnf:
            self.GS = self.param("GS", uniform_range(), (self.output_size,))

    def negation(self, x: Float[Array, "... i"]) -> Float[Array, "... i j"]:
        GN = sigmoid(self.GN)
        x = x[..., None]
        return (1 - GN) * x + GN * (1 - x)

    def conjunction(self, x: Float[Array, "... i j"]) -> Float[Array, "... j"]:
        GR = sigmoid(self.GR)
        x = jnp.log(jnp.clip(x, a_min=1e-10))
        # x = jnp.log(x + 1e-10)
        return jnp.exp(jnp.sum(GR * x, axis=-2))

    def disjunction(self, x: Float[Array, "... i j"]) -> Float[Array, "... j"]:
        return 1 - self.conjunction(1 - x)

    def selection(
        self, x_1: Float[Array, "... j"], x_2: Float[Array, "... j"]
    ) -> Float[Array, "... j"]:
        assert not self.nnf, "Selection is not defined for NNF"
        GS = sigmoid(self.GS)
        return (1 - GS) * x_1 + GS * x_2

    @nn.compact
    def __call__(self, x: Float[Array, "... i"]) -> Float[Array, "... j"]:
        x_neg = self.negation(x)
        x_and = self.conjunction(x_neg)
        if self.nnf:
            return x_and
        x_or = self.disjunction(x_neg)
        return self.selection(x_and, x_or)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(input_size={self.input_size}, "
            f"output_size={self.output_size}, nnf={self.nnf})"
        )


if __name__ == "__main__":
    from jax import random

    key = random.PRNGKey(0)
    batch_size = 3
    input_size = 10
    output_size = 5
    x = random.uniform(key, (batch_size, input_size))
    model = NeuralLogicRuleLayer(
        input_size=input_size, output_size=output_size, nnf=False
    )
    params = model.init(key, x)
    y = model.apply(params, x)
    print(y)
