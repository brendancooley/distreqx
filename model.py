from abc import ABC, abstractmethod
from typing import Generic, ParamSpec, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from distreqx import distributions as dist


T = TypeVar("T")
P = ParamSpec("P")


class Prior(eqx.Module, Generic[T]):
    dist: dist.AbstractDistribution
    value: T


class Model(eqx.Module, ABC):
    """Base class for Bayesian models compatible with variational inference."""

    @abstractmethod
    def log_prob(self, *args, **kwargs) -> Array:
        """Compute log likelihood of data given current parameters."""
        pass

    def get_prior_fields(self) -> dict[str, Prior]:
        """Extract Prior fields from the model for guide initialization."""
        prior_fields = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Prior):
                prior_fields[field_name] = field_value
        return prior_fields

    def log_prior(self) -> Array:
        """Compute log prior probability of current parameter values."""
        log_prob = 0.0
        for field_value in self.get_prior_fields().values():
            log_prob += jnp.sum(field_value.dist.log_prob(field_value.value))
        return log_prob


class AutoGuide(eqx.Module, ABC):
    """Base class for automatic variational guides."""

    template_model: Model

    @abstractmethod
    def sample(self, key: PRNGKeyArray, num_samples: int = 1) -> Model:
        """Sample concrete model instances from the variational distribution."""
        pass

    @abstractmethod
    def log_prob(self, model: Model) -> Array:
        """Compute log probability of model instances under variational distribution."""
        pass

    @abstractmethod
    def entropy(self) -> Array:
        """Compute entropy of the variational distribution."""
        pass


class AutoMeanField(AutoGuide):
    template_model: Model
    guides: dict[str, dist.Normal]

    def __init__(self, template_model: Model):
        self.template_model = template_model
        # Initialize guides from template model's prior fields
        self.guides = {}
        prior_fields = template_model.get_prior_fields()

        for i, (field_name, prior_field) in enumerate(prior_fields.items()):
            # Initialize with N(0, init_scale) in the unconstrained space
            self.guides[field_name] = dist.Normal(
                loc=jnp.zeros_like(prior_field.value),
                scale=jnp.ones_like(prior_field.value),
            )

    def _sample_parameters(
        self, key: PRNGKeyArray, num_samples: int = 1
    ) -> dict[str, Array]:
        """Sample parameters from the variational distributions."""
        keys = jax.random.split(key, len(self.guides))
        samples = {}

        for i, (param_name, guide_dist) in enumerate(self.guides.items()):
            if num_samples == 1:
                samples[param_name] = guide_dist.sample(keys[i])
            else:
                sample_keys = jax.random.split(keys[i], num_samples)
                samples[param_name] = jax.vmap(guide_dist.sample)(sample_keys)

        return samples

    def sample(self, key: PRNGKeyArray, num_samples: int = 1) -> Model:
        """Sample concrete model instances from the variational distribution."""
        param_samples = self._sample_parameters(key, num_samples)

        if num_samples == 1:
            # Single model instance
            return self._apply_samples_to_template(param_samples)
        else:
            # For multiple samples, we need to handle this differently
            # We'll return a pytree structure that can be vmapped over
            sample_dicts = [
                {k: v[i] for k, v in param_samples.items()} for i in range(num_samples)
            ]
            return jax.vmap(self._apply_samples_to_template)(sample_dicts)

    def log_prob(self, model: Model) -> Array:
        """Compute log probability of model instance under variational distribution."""
        # Extract parameter values from the model
        model_prior_fields = model.get_prior_fields()

        log_probs = []
        for param_name, guide_dist in self.guides.items():
            if param_name in model_prior_fields:
                param_value = model_prior_fields[param_name].value
                log_prob = guide_dist.log_prob(param_value)
                log_probs.append(jnp.sum(log_prob))  # Sum over parameter dimensions

        return jnp.sum(jnp.array(log_probs))

    def entropy(self) -> Array:
        """Compute entropy of mean field approximation."""
        entropies = []
        for guide_dist in self.guides.values():
            entropies.append(jnp.sum(guide_dist.entropy()))
        return jnp.sum(jnp.array(entropies))

    def _apply_samples_to_template(self, samples: dict[str, Array]) -> Model:
        """Apply sampled parameters to template model"""
        updates = {}
        for param_name, sample in samples.items():
            if hasattr(self.template_model, param_name):
                prior_field = getattr(self.template_model, param_name)
                if isinstance(prior_field, Prior):
                    updates[param_name] = eqx.tree_at(
                        lambda x: x.value, prior_field, sample
                    )

        return eqx.tree_at(
            lambda m: m,
            self.template_model,
            updates,
            is_leaf=lambda x: isinstance(x, Prior),
        )


class BayesianRegression(Model):
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
