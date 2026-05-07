"""Compare NVFP4 grouped quantize+GEMM:
- baseline RHT path (use_rht=True): existing graph-safe RHT cast-fusion pair.
- candidate no-RHT path (use_rht=False): persistent kernel + swizzle.

Both should produce numerically similar GEMM outputs (up to RHT vs no-RHT noise) when the
data is well-scaled. We compare against a BF16 reference grouped GEMM.
"""
import jax
import jax.numpy as jnp
import numpy as np

from transformer_engine.jax.quantize import (
    QuantizerFactory,
    ScalingMode,
    TensorUsage,
)
from transformer_engine.jax import cpp_extensions as tex


def reference_grouped_matmul(x, w, group_sizes):
    """BF16 grouped matmul: per-group x[g] @ w[g], where x is split along axis 0."""
    out_groups = []
    base = 0
    for g, gsz in enumerate(group_sizes):
        x_g = x[base:base + int(gsz)]
        out_groups.append((x_g.astype(jnp.float32) @ w[g].astype(jnp.float32)).astype(jnp.bfloat16))
        base += int(gsz)
    return jnp.concatenate(out_groups, axis=0)


def run_grouped_nvfp4(x, w, group_sizes, *, use_rht):
    n_groups = group_sizes.size
    quantizer_set = QuantizerFactory.create_set(
        scaling_mode=ScalingMode.NVFP4_1D_SCALING,
        fwd_dtype=jnp.float4_e2m1fn,
        bwd_dtype=jnp.float4_e2m1fn,
        is_2x2x=True,  # both rowwise + colwise
        n_groups=n_groups,
    )
    sr_rng = jnp.broadcast_to(jnp.asarray([0, 0, 0, 0], dtype=jnp.uint32), (1, 4))
    for sub_q in quantizer_set.x.quantizers:
        sub_q.use_rht = use_rht
        sub_q.stochastic_rounding_rng_state = sr_rng if use_rht else None
    for sub_q in quantizer_set.kernel.quantizers:
        sub_q.use_rht = use_rht
        sub_q.stochastic_rounding_rng_state = sr_rng if use_rht else None

    @jax.jit
    def run(x, w, group_sizes):
        cx = tex.grouped_quantize(x, quantizer_set.x, group_sizes, flatten_axis=-1)
        ck = tex.grouped_quantize(w, quantizer_set.kernel, flatten_axis=-1)
        return tex.grouped_gemm(
            cx.get_tensor(usage=TensorUsage.LHS),
            ck.get_tensor(usage=TensorUsage.RHS),
            contracting_dims=((1,), (1,)),
        )

    return run(x, w, group_sizes)


def main():
    n_groups = 4
    m = 512
    k = 256
    n = 256
    group_sizes = jnp.full((n_groups,), m // n_groups, dtype=jnp.int32)

    key = jax.random.PRNGKey(0)
    x_key, w_key = jax.random.split(key)
    x = jax.random.normal(x_key, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(w_key, (n_groups, k, n), dtype=jnp.bfloat16)

    out_ref = reference_grouped_matmul(x, w, group_sizes)
    out_rht = run_grouped_nvfp4(x, w, group_sizes, use_rht=True)
    out_norht = run_grouped_nvfp4(x, w, group_sizes, use_rht=False)

    out_ref = np.asarray(out_ref).astype(np.float32)
    out_rht = np.asarray(out_rht).astype(np.float32)
    out_norht = np.asarray(out_norht).astype(np.float32)

    def stats(label, actual, ref):
        diff = actual - ref
        rel = np.abs(diff) / (np.abs(ref) + 1e-6)
        print(f"{label}: max_abs={np.max(np.abs(diff)):.4f}, "
              f"mean_abs={np.mean(np.abs(diff)):.4f}, "
              f"max_rel={np.max(rel):.4f}, mean_rel={np.mean(rel):.4f}, "
              f"any_nan={np.any(np.isnan(actual))}, any_inf={np.any(np.isinf(actual))}")

    print(f"ref:    range=[{out_ref.min():.4f}, {out_ref.max():.4f}]")
    print(f"rht:    range=[{out_rht.min():.4f}, {out_rht.max():.4f}]")
    print(f"norht:  range=[{out_norht.min():.4f}, {out_norht.max():.4f}]")
    stats("rht    vs ref", out_rht, out_ref)
    stats("norht  vs ref", out_norht, out_ref)
    stats("norht vs rht", out_norht, out_rht)


if __name__ == "__main__":
    main()
