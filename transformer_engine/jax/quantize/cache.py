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

import math
from functools import partial
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
    "grouped_quantize_and_cache_kernel",
    "grouped_load_cached_kernel",
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
            is_colwise=is_colwise, is_padded=False, flatten_axis=fa,
        )
        scale_dtype = scaling_mode.get_scale_dtype()
        return [
            jax.ShapeDtypeStruct(kernel_shape, q_dtype),
            jax.ShapeDtypeStruct((1,), jnp.float32),
            jax.ShapeDtypeStruct(scale_shape, scale_dtype),
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


# ---------------------------------------------------------------------------
# Grouped GEMM kernel caching (MoE)
# ---------------------------------------------------------------------------

def _get_grouped_cache_variable_specs(
    kernel_shape: Tuple[int, ...],
    flatten_axis: int,
    quantizer,
) -> List[jax.ShapeDtypeStruct]:
    """Compute shapes/dtypes for grouped kernel cache variables.

    GroupedScaledTensor1x stores 1D-flattened data and 1D scale_inv.
    When first_dims=None (kernel case), the flat leaves are
    (data, scale_inv, amax) — 3 per 1x component.
    """
    q_layout = quantizer.q_layout
    q_dtype = quantizer.q_dtype
    scaling_mode = quantizer.scaling_mode

    data_shape = (math.prod(kernel_shape),)  # 1D flattened
    n_groups = kernel_shape[0]
    amax_shape = (1,)  # default from ScaledTensorFactory.create when amax=None
    scale_dtype = scaling_mode.get_scale_dtype()

    def _specs_for_1x(is_colwise: bool) -> List[jax.ShapeDtypeStruct]:
        # GroupedScaledTensor1x.__post_init__ validates with is_padded=True
        scale_shape = scaling_mode.get_grouped_scale_shape(
            kernel_shape, n_groups, is_colwise=is_colwise,
            is_padded=True, flatten_axis=flatten_axis,
        )
        # Leaf order matches GroupedScaledTensor1x.tree_flatten:
        # (data, scale_inv, amax, first_dims, last_dims)
        # first_dims=None and last_dims=None contribute 0 leaves.
        return [
            jax.ShapeDtypeStruct(data_shape, q_dtype),      # data
            jax.ShapeDtypeStruct(scale_shape, scale_dtype),  # scale_inv
            jax.ShapeDtypeStruct(amax_shape, jnp.float32),  # amax
        ]

    if q_layout.is_rowwise_colwise:
        return _specs_for_1x(False) + _specs_for_1x(True)
    if q_layout.is_colwise_only:
        return _specs_for_1x(True)
    return _specs_for_1x(False)


def _build_grouped_scaled_tensor_treedef(kernel_shape, flatten_axis, quantizer):
    """Build JAX treedef for a grouped quantized kernel without running quantization."""
    from ..cpp_extensions.quantization import GroupedQuantizePrimitive  # avoid circular

    specs = _get_grouped_cache_variable_specs(kernel_shape, flatten_axis, quantizer)
    dummy_leaves = [jnp.zeros(s.shape, s.dtype) for s in specs]

    q_layout = quantizer.q_layout
    scaling_mode = quantizer.scaling_mode
    data_layout = quantizer.data_layout

    use_v2 = GroupedQuantizePrimitive._use_v2_kernel(
        scaling_mode.value, kernel_shape, flatten_axis,
    )

    def _make_grouped_1x(off, is_colwise):
        layout_char = data_layout[1] if is_colwise else data_layout[0]
        return ScaledTensorFactory.create_1x(
            data=dummy_leaves[off],
            scale_inv=dummy_leaves[off + 1],
            amax=dummy_leaves[off + 2],
            scaling_mode=scaling_mode,
            dq_dtype=jnp.bfloat16,
            is_colwise=is_colwise,
            data_layout=layout_char,
            flatten_axis=flatten_axis,
            first_dims=None,
            last_dims=None,
            original_shape=kernel_shape,
            pre_swizzled=use_v2,
        )

    if q_layout.is_rowwise_colwise:
        dummy = ScaledTensor2x(
            _make_grouped_1x(0, False), _make_grouped_1x(3, True),
        )
    elif q_layout.is_colwise_only:
        dummy = _make_grouped_1x(0, True)
    else:
        dummy = _make_grouped_1x(0, False)

    _, treedef = jax.tree_util.tree_flatten(dummy)
    return treedef


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _grouped_quantize_detached(kernel, quantizer_kernel):
    """Quantize a grouped kernel with no backward pass.

    Wrapped in ``custom_vjp`` so that JAX never tries to differentiate
    through ``GroupedQuantizePrimitive`` (which lacks a VJP rule).
    Weight gradients flow through ``_grouped_dense_bwd_rule`` instead.

    ``quantizer_kernel`` is a ``GroupedQuantizer`` and is marked
    *nondiff* — it is a static argument from JAX's perspective.
    """
    return tex.grouped_quantize(kernel, quantizer_kernel, flatten_axis=-1)

def _grouped_quantize_detached_fwd(kernel, quantizer_kernel):
    result = tex.grouped_quantize(kernel, quantizer_kernel, flatten_axis=-1)
    return result, ()

def _grouped_quantize_detached_bwd(quantizer_kernel, _res, g):
    del quantizer_kernel, g
    # Gradient is not needed — the cached kernel is treated as a constant.
    # Return None for the kernel argument.
    return (None,)

_grouped_quantize_detached.defvjp(
    _grouped_quantize_detached_fwd, _grouped_quantize_detached_bwd,
)


def grouped_quantize_and_cache_kernel(module, quantizer_set, kernel):
    """Quantize a grouped kernel and store the result in Flax mutable variables.

    Grouped variant of ``quantize_and_cache_kernel`` for MoE expert
    weight matrices of shape ``(G, K, N)``.

    The quantization is wrapped in a ``custom_vjp`` so that:
    - ``GroupedQuantizePrimitive`` (no VJP / no shardy rule) is shielded
      from JAX's autodiff and sharding propagation.
    - Callers MUST still ensure this runs inside ``shard_map`` covering
      all EP / FSDP axes.
    """
    casted_kernel = _grouped_quantize_detached(kernel, quantizer_set.kernel)

    leaves, _ = jax.tree_util.tree_flatten(casted_kernel)
    if module.is_mutable_collection(QW_CACHE_COLLECTION):
        for i, leaf in enumerate(leaves):
            module.variable(
                QW_CACHE_COLLECTION, f"leaf_{i}",
                lambda l=leaf: jnp.zeros_like(l),
            ).value = leaf

    return QuantizerSet(
        x=quantizer_set.x,
        kernel=quantizer_set.kernel,
        dgrad=quantizer_set.dgrad,
        cached_kernel=casted_kernel,
    )


def grouped_load_cached_kernel(module, quantizer_set, kernel):
    """Load a previously cached grouped quantized kernel from Flax variables.

    Grouped variant of ``load_cached_kernel`` for MoE expert weight
    matrices of shape ``(G, K, N)``.
    """
    flatten_axis_k = -1

    specs = _get_grouped_cache_variable_specs(
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

    treedef = _build_grouped_scaled_tensor_treedef(
        kernel.shape, flatten_axis_k, quantizer_set.kernel,
    )
    casted_kernel = jax.tree_util.tree_unflatten(treedef, leaves)

    return QuantizerSet(
        x=quantizer_set.x,
        kernel=quantizer_set.kernel,
        dgrad=quantizer_set.dgrad,
        cached_kernel=casted_kernel,
    )
