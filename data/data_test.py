import pickle
import jax.numpy as jnp

from dataset import ALL_FNS


def print_new_dataset():
    with open("data/dataset.pkl", "rb") as f:
        data = pickle.load(f)

    for fn, f in ALL_FNS.items():
        d_train = jnp.array(data["train"][0]), jnp.array(data["train"][1][fn])
        d_test = jnp.array(data["test"][0]), jnp.array(data["test"][1][fn])

        x_1 = d_train[0][:, 0]
        x_2 = d_train[0][:, 1]

        y = d_train[1]
        y_ = f(x_1, x_2).reshape(-1, 1)
        delta = y - y_
        print(f"{fn}: {delta.mean()=:.6f}")

        x_1 = d_test[0][:, 0]
        x_2 = d_test[0][:, 1]

        y = d_test[1]
        y_ = f(x_1, x_2).reshape(-1, 1)
        delta = y - y_
        print(f"{fn}: {delta.mean()=:.6f}")


def print_old_dataset():
    with open("np_data.npy", "rb") as f:
        data = pickle.load(f)

    for fn, f in ALL_FNS.items():
        d_train = jnp.array(data["train"][0]), jnp.array(data["train"][1][fn])
        d_test = jnp.array(data["test"][0]), jnp.array(data["test"][1][fn])

        x_1 = d_train[0][:, 0]
        x_2 = d_train[0][:, 1]

        y = d_train[1]
        y_ = f(x_1, x_2).reshape(-1, 1)
        delta = y - y_
        print(f"{fn}: {delta.mean()=:.6f}")

        x_1 = d_test[0][:, 0]
        x_2 = d_test[0][:, 1]

        y = d_test[1]
        y_ = f(x_1, x_2).reshape(-1, 1)
        delta = y - y_
        print(f"{fn}: {delta.mean()=:.6f}")


if __name__ == "__main__":
    print_new_dataset()
