from beartype import beartype
from beartype.typing import Any, Sequence
from jaxtyping import Float, Array

import jax.numpy as jnp
from jax.nn import sigmoid
from jax import dtypes
import jax.random as random
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
    features: int
    nnf: bool = False

    def negation(
        self, GN: Float[Array, "i j"], x: Float[Array, "... i"]
    ) -> Float[Array, "... i j"]:
        x = x[..., None]
        return (1 - sigmoid(GN)) * x + sigmoid(GN) * (1 - x)

    def conjunction(
        self, GR: Float[Array, "i j"], x: Float[Array, "... i j"]
    ) -> Float[Array, "... j"]:
        x = jnp.log(jnp.clip(x, a_min=1e-10))
        # x = jnp.log(x + 1e-10)
        return jnp.exp(jnp.sum(sigmoid(GR) * x, axis=-2))

    def disjunction(
        self, GR: Float[Array, "i j"], x: Float[Array, "... i j"]
    ) -> Float[Array, "... j"]:
        return 1 - self.conjunction(GR, 1 - x)

    def selection(
        self,
        GS: Float[Array, "j"],
        x_1: Float[Array, "... j"],
        x_2: Float[Array, "... j"],
    ) -> Float[Array, "... j"]:
        return (1 - sigmoid(GS)) * x_1 + sigmoid(GS) * x_2

    @nn.compact
    def __call__(self, x: Float[Array, "... i"]) -> Float[Array, "... j"]:
        i = x.shape[-1]
        GN = self.param("GN", uniform_range(), (i, self.features))
        GR = self.param("GR", uniform_range(), (i, self.features))

        x_neg = self.negation(GN, x)
        x_and = self.conjunction(GR, x_neg)
        if self.nnf:
            return x_and

        GS = self.param("GS", uniform_range(), (self.features,))
        x_or = self.disjunction(GR, x_neg)
        return self.selection(GS, x_and, x_or)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(features={self.features}, nnf={self.nnf})"


if __name__ == "__main__":
    from jax import random

    key = random.PRNGKey(0)
    batch_size = 3
    input_size = 10
    output_size = 5
    x = random.uniform(key, (batch_size, input_size))
    model = NeuralLogicRuleLayer(features=output_size, nnf=False)
    params = model.init(key, x)
    y = model.apply(params, x)
    print(y)
