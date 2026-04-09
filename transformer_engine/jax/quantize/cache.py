# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Quantized weight caching for gradient accumulation.

This module provides infrastructure for caching quantized weights across
gradient-accumulation micro-steps, avoiding redundant re-quantization of
unchanged master weights.

The design uses **two separate JIT traces** rather than ``jax.lax.cond``:

* **Fresh trace** — ``quantize_and_cache_kernel``: runs ``tex.quantize``
  and stores the result in the ``quantized_kernel_cache`` Flax variable
  collection.  Used for the first micro-step of each GA cycle.
* **Cached trace** — ``load_cached_kernel``: loads the previously cached
  quantized kernel from the collection.  Used for micro-steps 2..K.

Two traces are needed because ``tex.quantize`` calls primitives
(``jax_local_amax_wrapper``) that have no VJP rule, making it
incompatible with ``jax.lax.cond`` inside a differentiated context.
Python-level branching naturally produces separate traces cached by JIT.
"""

from typing import List, Tuple

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
    "quantize_and_cache_kernel",
    "load_cached_kernel",
]

QW_CACHE_COLLECTION = "quantized_kernel_cache"


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
# Public API — two functions, one per JIT trace
# ---------------------------------------------------------------------------

def quantize_and_cache_kernel(module, quantizer_set, kernel, contracting_dims):
    """Quantize the kernel and store the result in Flax mutable variables.

    Called on the **first** micro-step of a GA cycle.  The quantized
    leaves are written to the ``quantized_kernel_cache`` collection
    (which must be in the ``mutable`` list of ``model.apply``).

    ``stop_gradient`` is applied to the cached kernel so that the
    gradient does not flow backward through ``tex.quantize`` (whose
    internals lack VJP rules).  Weight gradients still flow correctly
    through ``_dense_bwd_rule`` (the ``custom_vjp`` backward in
    ``dense.py``).

    Returns a new ``QuantizerSet`` with ``cached_kernel`` populated.
    """
    flatten_axis_k = _flatten_axis_for_kernel(kernel.ndim, contracting_dims)

    casted_kernel = tex.quantize(
        kernel,
        flatten_axis=flatten_axis_k,
        quantizer=quantizer_set.kernel,
        amax_scope=tex.AmaxScope.FSDP,
    )

    # Store each leaf array in a Flax mutable variable for later reuse.
    leaves, _ = jax.tree_util.tree_flatten(casted_kernel)
    if module.is_mutable_collection(QW_CACHE_COLLECTION):
        for i, leaf in enumerate(leaves):
            module.variable(
                QW_CACHE_COLLECTION, f"leaf_{i}",
                lambda l=leaf: jnp.zeros_like(l),
            ).value = leaf

    # stop_gradient: the cached kernel is a constant from the gradient
    # perspective.  Without this, JAX would attempt to differentiate
    # backward through tex.quantize (which contains jax_local_amax_wrapper,
    # a primitive with no VJP rule).
    casted_kernel = jax.lax.stop_gradient(casted_kernel)

    return QuantizerSet(
        x=quantizer_set.x,
        kernel=quantizer_set.kernel,
        dgrad=quantizer_set.dgrad,
        cached_kernel=casted_kernel,
    )


def load_cached_kernel(module, quantizer_set, kernel, contracting_dims):
    """Load a previously cached quantized kernel from Flax variables.

    Called on micro-steps **2..K** of a GA cycle.  The cache must have
    been populated by a prior ``quantize_and_cache_kernel`` call whose
    mutable output was merged into the variable collections passed to
    this ``model.apply``.

    Returns a new ``QuantizerSet`` with ``cached_kernel`` populated.
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
