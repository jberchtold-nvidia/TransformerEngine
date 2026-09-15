# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""cuDNN Frontend JAX API adapter for fused grouped GEMM + SwiGLU."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "grouped_gemm_swiglu",
    "grouped_gemm_swiglu_dependencies_available",
    "pack_swiglu_pair",
    "unpack_swiglu_pair",
]


def pack_swiglu_pair(gate: jax.Array, up: jax.Array) -> jax.Array:
    """Interleave 32-column gate/up blocks as required by the cuDNN kernel."""
    if gate.shape != up.shape:
        raise ValueError(f"gate shape {gate.shape} must match up shape {up.shape}")
    if gate.shape[-1] % 32:
        raise ValueError(f"SwiGLU intermediate dimension {gate.shape[-1]} must be divisible by 32")
    blocks = gate.shape[-1] // 32
    return jnp.stack(
        (
            gate.reshape(*gate.shape[:-1], blocks, 32),
            up.reshape(*up.shape[:-1], blocks, 32),
        ),
        axis=-2,
    ).reshape(*gate.shape[:-1], 2 * gate.shape[-1])


def unpack_swiglu_pair(interleaved: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Undo :func:`pack_swiglu_pair`."""
    if interleaved.shape[-1] % 64:
        raise ValueError(
            f"Interleaved SwiGLU dimension {interleaved.shape[-1]} must be divisible by 64"
        )
    intermediate = interleaved.shape[-1] // 2
    blocks = intermediate // 32
    paired = interleaved.reshape(*interleaved.shape[:-1], blocks, 2, 32)
    return (
        paired[..., 0, :].reshape(*interleaved.shape[:-1], intermediate),
        paired[..., 1, :].reshape(*interleaved.shape[:-1], intermediate),
    )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _sf_atom_shape(groups: int, rows: int, cols: int) -> tuple[int, ...]:
    """Return the physical row-major scale-factor shape used by cudnn.jax.call."""
    return (groups, _ceil_div(rows, 128), _ceil_div(_ceil_div(cols, 32), 4), 32, 4, 4)


def _compact_sf(scale: jax.Array, shape: tuple[int, ...], name: str) -> jax.Array:
    """Trim TE's conservative grouped-scale allocation to cudnn's dense atom view."""
    size = 1
    for extent in shape:
        size *= extent
    if scale.size < size:
        raise ValueError(f"{name} has {scale.size} elements, but cuDNN requires {size}")
    return scale.reshape(-1)[:size].reshape(shape)


def grouped_gemm_swiglu_dependencies_available() -> tuple[bool, str]:
    """Check the public cuDNN JAX API without compiling a kernel."""
    try:
        import cutlass.jax
        from cudnn import grouped_gemm_swiglu_jax_sm100  # noqa: F401

        if not cutlass.jax.is_available():
            return False, "CuTeDSL JAX support is unavailable"
    except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError) as exc:
        return False, str(exc)
    return True, ""


def grouped_gemm_swiglu(
    a: jax.Array,
    b: jax.Array,
    sfa: jax.Array,
    sfb: jax.Array,
    padded_offsets: jax.Array,
    prob: jax.Array,
    *,
    compute_dtype,
    output_dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run cuDNN's dedicated JAX grouped MXFP8 GEMM + SwiGLU API.

    TE owns conservative flat scale allocations for ragged grouped operations.
    cuDNN's JAX API accepts the compact physical atom layout, so this adapter
    exposes the used prefix with a zero-copy reshape before making the call.
    """
    if a.ndim != 3 or a.shape[-1] != 1:
        raise ValueError(f"Expected A[M,K,1], got {a.shape}")
    if b.ndim != 3:
        raise ValueError(f"Expected physical B[E,N,K], got {b.shape}")

    from cudnn import grouped_gemm_swiglu_jax_sm100

    rows, hidden, _ = a.shape
    experts, combined, b_hidden = b.shape
    if hidden != b_hidden:
        raise ValueError(f"A K={hidden} does not match B K={b_hidden}")
    alpha = jnp.ones((experts,), dtype=jnp.float32)
    norm_const = jnp.ones((1,), dtype=jnp.float32)
    result = grouped_gemm_swiglu_jax_sm100(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=_compact_sf(sfa, _sf_atom_shape(1, rows, hidden), "sfa"),
        sfb_tensor=_compact_sf(sfb, _sf_atom_shape(experts, combined, hidden), "sfb"),
        padded_offsets=padded_offsets.astype(jnp.int32),
        alpha_tensor=alpha,
        prob_tensor=prob.astype(jnp.float32),
        norm_const_tensor=norm_const,
        c_dtype=jnp.dtype(compute_dtype),
        d_dtype=jnp.dtype(output_dtype),
        sf_vec_size=32,
        discrete_col_sfd=True,
    )
    return (
        result["c_tensor"],
        result["d_tensor"],
        result["d_col_tensor"],
        result["sfd_row_tensor"].reshape(-1),
        result["sfd_col_tensor"].reshape(-1),
    )
