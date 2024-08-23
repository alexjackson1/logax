from beartype import beartype
from jaxtyping import Float, Array

from flax import linen as nn

from modules.nlrl import NeuralLogicRuleLayer


# class NeuralLogicNet(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         x = NeuralLogicRuleLayer(2, 5, False)(x)
#         x = NeuralLogicRuleLayer(5, 5, False)(x)
#         x = NeuralLogicRuleLayer(5, 5, False)(x)
#         x = NeuralLogicRuleLayer(5, 1, False)(x)
#         return x


@beartype
class NeuralLogicNetwork(nn.Module):
    depth: int
    width: int
    nnf: bool = False
    out_features: int = 1

    @nn.compact
    def __call__(self, x: Float[Array, "... i"]) -> Float[Array, "... j"]:
        assert self.depth >= 2, "Number of layers must be at least 2."
        x = NeuralLogicRuleLayer(self.width, self.nnf)(x)
        for _ in range(self.depth - 2):
            x = NeuralLogicRuleLayer(self.width, self.nnf)(x)
        x = NeuralLogicRuleLayer(self.out_features, self.nnf)(x)
        return x


@beartype
class FullyConnectedNetwork(nn.Module):
    depth: int
    width: int
    dropout: float
    out_features: int = 1

    @nn.compact
    def __call__(self, x: Float[Array, "... in"]) -> Float[Array, "... 1"]:
        assert self.depth >= 2, "Number of layers must be at least 2."
        x = nn.Dense(self.width)(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout, deterministic=True)(x)
        for _ in range(self.depth - 2):
            x = nn.Dense(self.width)(x)
            x = nn.relu(x)
            x = nn.Dropout(self.dropout, deterministic=True)(x)
        x = nn.Dense(self.out_features)(x)
        return x


if __name__ == "__main__":
    import pickle
    import jax.numpy as jnp
    from jax import random

    with open("data/dataset.pkl", "rb") as f:
        data = pickle.load(f)

    model = NeuralLogicNetwork(depth=4, width=5)
    params = model.init(random.PRNGKey(0), jnp.zeros([1, 2]))
    print(params)
    print(model.apply(params, jnp.zeros([1, 2])))

    d_train = jnp.array(data["train"][0]), jnp.array(data["train"][1]["x ⊕ y"])
    d_test = jnp.array(data["test"][0]), jnp.array(data["test"][1]["x ⊕ y"])

    print(d_train[0].shape, d_train[1].shape)
    print(d_test[0].shape, d_test[1].shape)

    print(model.apply(params, d_train[0]).shape)
    print(model.apply(params, d_test[0]).shape)
