# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Optax utilities for TransformerEngine quantized-weight caching.

Provides helpers for coordinating gradient accumulation with quantized
weight caching so that weights are only quantized once per optimizer step.
"""

import jax.numpy as jnp

__all__ = ["should_quantize"]


def should_quantize(multi_steps_state) -> jnp.ndarray:
    """Check whether quantized weights should be refreshed on this micro-step.

    Returns ``True`` on the first micro-step of each accumulation cycle
    (i.e. right after the optimizer applied an update), indicating that
    the cached quantized weights are stale and must be recomputed.

    Parameters
    ----------
    multi_steps_state : optax.MultiStepsState
        The optimizer state from ``optax.MultiSteps``.

    Returns
    -------
    jnp.ndarray
        Scalar boolean — ``True`` means re-quantize, ``False`` means
        the cache is still valid.

    Example
    -------
    .. code-block:: python

        from transformer_engine.jax.optax import should_quantize
        from transformer_engine.jax.quantize.cache import set_quantized_cache_validity

        optimizer = optax.MultiSteps(optax.adam(lr), every_k_schedule=K)

        for step in range(total_steps * K):
            refresh = bool(jax.device_get(should_quantize(opt_state)))
            var_collect = set_quantized_cache_validity(
                var_collect, is_valid=not refresh
            )
            ...
    """
    return multi_steps_state.mini_step == 0
