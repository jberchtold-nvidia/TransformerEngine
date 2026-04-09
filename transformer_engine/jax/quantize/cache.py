# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Quantized weight caching for gradient accumulation.

This module provides infrastructure for caching quantized weights across
gradient-accumulation micro-steps, avoiding redundant re-quantization of
unchanged master weights.

Design
------
A module-level Python flag ``_USE_QUANTIZED_WEIGHT_CACHE`` controls
whether caching modules quantize fresh or read from cache.  It is set
**before** the JIT'd train-step is called, so it is baked into the trace:

* ``False`` (default) — quantize fresh and populate the cache.
* ``True`` — load cached leaves from Flax variables, skip quantization.

Since the flag is all-or-nothing, every quantization module in the model
sees the same value during a single trace.  This gives exactly **two JIT
compilations**: one for each flag value.  Both traces are fully compatible
with ``lax.scan`` (scan_decoder_layers) because there is no per-module
branching inside the scanned body — the branch is resolved at trace time.
"""

from functools import partial
from typing import List, Tuple
from contextlib import contextmanager

import jax
import jax.numpy as jnp

from .. import cpp_extensions as tex
from .quantizer import Quantizer, QuantizerSet
from .tensor import (
    ScaledTensor1x,
    ScaledTensor2x,
    ScaledTensorFactory,
)

__all__ = [
    "QW_CACHE_COLLECTION",
    "use_quantized_weight_cache",
    "quantize_and_cache_kernel",
    "load_cached_kernel",
    "grouped_quantize_and_cache_kernel",
    "grouped_load_cached_kernel",
]

QW_CACHE_COLLECTION = "quantized_kernel_cache"


# ---------------------------------------------------------------------------
# Global flag — set before each JIT'd call, baked into the trace
# ---------------------------------------------------------------------------

_USE_QUANTIZED_WEIGHT_CACHE: bool = False


@contextmanager
def use_quantized_weight_cache(enabled: bool = True):
    """Context manager to toggle quantized-weight caching.

    Must be used **outside** the JIT boundary (wrapping the call to the
    JIT'd function).  Each boolean value produces a separate JIT trace.

    Example::

        # micro-step 0: quantize fresh, populate cache
        with use_quantized_weight_cache(False):
            (_, aux0), grads0 = grad_func(...)

        # micro-steps 1..K-1: use cached quantized weights
        with use_quantized_weight_cache(True):
            grad_and_loss, aux = jax.lax.scan(accumulate, ...)
    """
    global _USE_QUANTIZED_WEIGHT_CACHE
    prev = _USE_QUANTIZED_WEIGHT_CACHE
    _USE_QUANTIZED_WEIGHT_CACHE = enabled
    try:
        yield
    finally:
        _USE_QUANTIZED_WEIGHT_CACHE = prev


def is_using_quantized_weight_cache() -> bool:
    """Return the current cache flag (read at trace time)."""
    return _USE_QUANTIZED_WEIGHT_CACHE


# ---------------------------------------------------------------------------
# Shape / treedef helpers (no actual quantization)
# ---------------------------------------------------------------------------

def _get_cache_variable_specs(
    kernel_shape: Tuple[int, ...],
    flatten_axis: int,
    quantizer: Quantizer,
) -> List[jax.ShapeDtypeStruct]:
    """Compute shapes/dtypes for cache variables without running quantization."""
    q_layout = quantizer.q_layout
    q_dtype = quantizer.q_dtype
    scaling_mode = quantizer.scaling_mode
    data_layout = quantizer.data_layout

    if flatten_axis < 0:
        flatten_axis = len(kernel_shape) + flatten_axis
    assert 0 < flatten_axis < len(kernel_shape), (
        f"flatten_axis {flatten_axis} out of range for kernel shape {kernel_shape}"
    )

    def _specs_for_1x(is_colwise: bool) -> List[jax.ShapeDtypeStruct]:
        layout_char = data_layout[1] if is_colwise else data_layout[0]
        fa = flatten_axis
        if layout_char == "T":
            fa = len(kernel_shape) - flatten_axis
        scale_shape = scaling_mode.get_scale_shape(
            kernel_shape, data_layout=layout_char,
            is_colwise=is_colwise, is_padded=True, flatten_axis=fa,
        )
        return [
            jax.ShapeDtypeStruct(kernel_shape, q_dtype),
            jax.ShapeDtypeStruct((1,), jnp.float32),
            jax.ShapeDtypeStruct(scale_shape, jnp.float32),
        ]

    if q_layout.is_rowwise_colwise:
        return _specs_for_1x(False) + _specs_for_1x(True)
    if q_layout.is_colwise_only:
        return _specs_for_1x(True)
    return _specs_for_1x(False)


def _build_scaled_tensor_treedef(kernel_shape, flatten_axis, quantizer):
    """Build JAX treedef for the quantized kernel without running quantization."""
    specs = _get_cache_variable_specs(kernel_shape, flatten_axis, quantizer)
    dummy_leaves = [jnp.zeros(s.shape, s.dtype) for s in specs]

    q_layout = quantizer.q_layout
    scaling_mode = quantizer.scaling_mode
    data_layout = quantizer.data_layout
    if flatten_axis < 0:
        flatten_axis = len(kernel_shape) + flatten_axis

    def _make_1x(off, is_colwise):
        layout_char = data_layout[1] if is_colwise else data_layout[0]
        fa = flatten_axis
        if layout_char == "T":
            fa = len(kernel_shape) - flatten_axis
        return ScaledTensorFactory.create_1x(
            data=dummy_leaves[off], scale_inv=dummy_leaves[off + 2],
            amax=dummy_leaves[off + 1], scaling_mode=scaling_mode,
            dq_dtype=jnp.bfloat16, is_colwise=is_colwise,
            data_layout=layout_char, flatten_axis=fa,
        )

    if q_layout.is_rowwise_colwise:
        dummy = ScaledTensor2x(_make_1x(0, False), _make_1x(3, True))
    elif q_layout.is_colwise_only:
        dummy = _make_1x(0, True)
    else:
        dummy = _make_1x(0, False)

    _, treedef = jax.tree_util.tree_flatten(dummy)
    return treedef


def _flatten_axis_for_kernel(kernel_ndim, contracting_dims):
    """Compute the flatten_axis for kernel quantization."""
    _, k_contracting_dims = map(
        tex.sanitize_dims, (kernel_ndim, kernel_ndim), contracting_dims
    )
    return len(k_contracting_dims) - kernel_ndim


# ---------------------------------------------------------------------------
# Dense (non-grouped) cache functions
# ---------------------------------------------------------------------------

def quantize_and_cache_kernel(module, quantizer_set, kernel, contracting_dims):
    """Quantize the kernel and store the result in Flax mutable variables.

    Called when ``_USE_QUANTIZED_WEIGHT_CACHE`` is ``False`` (fresh trace).
    ``stop_gradient`` is applied so the gradient does not flow backward
    through ``tex.quantize``.
    """
    flatten_axis_k = _flatten_axis_for_kernel(kernel.ndim, contracting_dims)

    casted_kernel = tex.quantize(
        kernel,
        flatten_axis=flatten_axis_k,
        quantizer=quantizer_set.kernel,
        amax_scope=tex.AmaxScope.FSDP,
    )

    leaves, _ = jax.tree_util.tree_flatten(casted_kernel)
    if module.is_mutable_collection(QW_CACHE_COLLECTION):
        for i, leaf in enumerate(leaves):
            module.variable(
                QW_CACHE_COLLECTION, f"leaf_{i}",
                lambda l=leaf: jnp.zeros_like(l),
            ).value = leaf

    casted_kernel = jax.lax.stop_gradient(casted_kernel)

    return QuantizerSet(
        x=quantizer_set.x,
        kernel=quantizer_set.kernel,
        dgrad=quantizer_set.dgrad,
        cached_kernel=casted_kernel,
    )


def load_cached_kernel(module, quantizer_set, kernel, contracting_dims):
    """Load a previously cached quantized kernel from Flax variables.

    Called when ``_USE_QUANTIZED_WEIGHT_CACHE`` is ``True`` (cached trace).
    No call to ``tex.quantize`` — shapes and treedef are computed from
    the quantizer config.
    """
    flatten_axis_k = _flatten_axis_for_kernel(kernel.ndim, contracting_dims)

    specs = _get_cache_variable_specs(
        kernel.shape, flatten_axis_k, quantizer_set.kernel,
    )
    leaves = [
        jax.lax.stop_gradient(
            module.variable(
                QW_CACHE_COLLECTION, f"leaf_{i}",
                lambda s=spec.shape, d=spec.dtype: jnp.zeros(s, dtype=d),
            ).value
        )
        for i, spec in enumerate(specs)
    ]

    treedef = _build_scaled_tensor_treedef(
        kernel.shape, flatten_axis_k, quantizer_set.kernel,
    )
    casted_kernel = jax.tree_util.tree_unflatten(treedef, leaves)

    return QuantizerSet(
        x=quantizer_set.x,
        kernel=quantizer_set.kernel,
        dgrad=quantizer_set.dgrad,
        cached_kernel=casted_kernel,
    )


# ---------------------------------------------------------------------------
# Grouped GEMM cache functions
# ---------------------------------------------------------------------------

def _grouped_flatten_axis_for_kernel(kernel_ndim, contracting_dims):
    """Compute flatten_axis for grouped kernel quantization (+1 for G axis)."""
    _, k_contracting_dims = contracting_dims
    return len(k_contracting_dims) - kernel_ndim + 1


# Module-level custom_vjp for tex.grouped_quantize.
# GroupedQuantizePrimitive has no registered JAX differentiation rule.
# This wrapper provides a zero-gradient backward so lax.scan can
# differentiate through the body without failing.
@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _grouped_quantize_no_grad(kernel, quantizer, flatten_axis):
    """tex.grouped_quantize wrapped with a zero-gradient backward."""
    return tex.grouped_quantize(kernel, quantizer, flatten_axis=flatten_axis)


def _grouped_quantize_no_grad_fwd(kernel, quantizer, flatten_axis):
    out = _grouped_quantize_no_grad(kernel, quantizer, flatten_axis)
    return out, (kernel,)


def _grouped_quantize_no_grad_bwd(quantizer, flatten_axis, res, g):
    del quantizer, flatten_axis, g
    (kernel,) = res
    return (jnp.zeros_like(kernel),)


_grouped_quantize_no_grad.defvjp(
    _grouped_quantize_no_grad_fwd,
    _grouped_quantize_no_grad_bwd,
)


def grouped_quantize_and_cache_kernel(
    module, quantizer_set, kernel, group_sizes, contracting_dims
):
    """Grouped-GEMM variant of :func:`quantize_and_cache_kernel`."""
    flatten_axis_k = _grouped_flatten_axis_for_kernel(kernel.ndim, contracting_dims)

    casted_kernel = _grouped_quantize_no_grad(
        kernel, quantizer_set.kernel, flatten_axis_k,
    )

    leaves, treedef = jax.tree_util.tree_flatten(casted_kernel)
    # Store treedef on the module class for grouped_load_cached_kernel.
    type(module)._cached_grouped_treedef = treedef
    if module.is_mutable_collection(QW_CACHE_COLLECTION):
        for i, leaf in enumerate(leaves):
            module.variable(
                QW_CACHE_COLLECTION, f"leaf_{i}",
                lambda l=leaf: jnp.zeros_like(l),
            ).value = leaf
        module.variable(
            QW_CACHE_COLLECTION, "num_leaves",
            lambda: jnp.int32(len(leaves)),
        ).value = jnp.int32(len(leaves))

    casted_kernel = jax.lax.stop_gradient(casted_kernel)

    return QuantizerSet(
        x=quantizer_set.x,
        kernel=quantizer_set.kernel,
        dgrad=quantizer_set.dgrad,
        cached_kernel=casted_kernel,
    )


def grouped_load_cached_kernel(module, quantizer_set, kernel, group_sizes, contracting_dims):
    """Grouped-GEMM variant of :func:`load_cached_kernel`.

    Retrieves the treedef captured by :func:`grouped_quantize_and_cache_kernel`
    on the fresh trace (stored as a class attribute).  No call to
    ``tex.grouped_quantize``.
    """
    num_leaves = int(module.variable(
        QW_CACHE_COLLECTION, "num_leaves", lambda: jnp.int32(0),
    ).value)

    leaves = [
        jax.lax.stop_gradient(
            module.variable(
                QW_CACHE_COLLECTION, f"leaf_{i}",
                lambda: jnp.zeros((), dtype=jnp.float32),
            ).value
        )
        for i in range(num_leaves)
    ]

    treedef = getattr(type(module), "_cached_grouped_treedef", None)
    if treedef is None:
        raise RuntimeError(
            "grouped_load_cached_kernel: treedef not found. "
            "The fresh trace (grouped_quantize_and_cache_kernel) must run first."
        )
    casted_kernel = jax.tree_util.tree_unflatten(treedef, leaves)

    return QuantizerSet(
        x=quantizer_set.x,
        kernel=quantizer_set.kernel,
        dgrad=quantizer_set.dgrad,
        cached_kernel=casted_kernel,
    )
