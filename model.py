from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from distreqx import distributions as dist


T = TypeVar("T")


class Prior(eqx.Module, Generic[T]):
    dist: dist.AbstractDistribution
    value: T


class BayesianRegression(eqx.Module):
    w: Prior[Float[Array, " N_features"]]
    log_sigma: Prior[Float[Array, ""]]

    def __init__(self, n_features: int):
        self.w = Prior(
            dist.Normal(jnp.zeros((n_features,)), jnp.ones((n_features,))),
            jnp.zeros(n_features),
        )
        self.log_sigma = Prior(dist.Normal(jnp.zeros(1), jnp.ones(1)), jnp.zeros(1))

    def __call__(self, x: Float[Array, " N_features"]) -> dist.Normal:
        return dist.Normal(
            loc=jnp.dot(x, self.w.value), scale=jnp.exp(self.log_sigma.value)
        )

    def log_prob(self, x: Float[Array, " N_features"], y: Float[Array, ""]):
        return self(x).log_prob(y)


if __name__ == "__main__":
    import polars as pl

    # https://num.pyro.ai/en/stable/tutorials/bayesian_regression.html
    DATASET_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv"
    df = pl.read_csv(DATASET_URL, separator=";")

    model = BayesianRegression(n_features=3)
    print('done')
