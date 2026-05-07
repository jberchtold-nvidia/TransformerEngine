"""Smoke test for NVFP4 grouped quantize V2 + grouped GEMM V2 (no-RHT persistent + swizzle path)."""
import jax
import jax.numpy as jnp

from transformer_engine.jax.quantize import (
    QuantizerFactory,
    ScalingMode,
    QuantizeLayout,
    TensorUsage,
)
from transformer_engine.jax import cpp_extensions as tex


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

    quantizer_set = QuantizerFactory.create_set(
        scaling_mode=ScalingMode.NVFP4_1D_SCALING,
        fwd_dtype=jnp.float4_e2m1fn,
        bwd_dtype=jnp.float4_e2m1fn,
        is_2x2x=False,  # exercises the COLWISE-only kernel quantizer path
        n_groups=n_groups,
    )
    # No-RHT path: persistent NVFP4 grouped quantize kernel + grouped scale swizzle.
    for sub_q in quantizer_set.x.quantizers:
        sub_q.use_rht = False
    for sub_q in quantizer_set.kernel.quantizers:
        sub_q.use_rht = False

    print(f"x quantizer q_layout = {quantizer_set.x.q_layout}")
    print(f"kernel quantizer q_layout = {quantizer_set.kernel.q_layout}")

    @jax.jit
    def run(x, w, group_sizes):
        cx = tex.grouped_quantize(x, quantizer_set.x, group_sizes, flatten_axis=-1)
        ck = tex.grouped_quantize(w, quantizer_set.kernel, flatten_axis=-1)
        return tex.grouped_gemm(
            cx.get_tensor(usage=TensorUsage.LHS),
            ck.get_tensor(usage=TensorUsage.RHS),
            contracting_dims=((1,), (1,)),
        )

    out = run(x, w, group_sizes)
    out.block_until_ready()
    print(f"grouped_gemm output shape: {out.shape}, dtype: {out.dtype}")
    print("OK")


if __name__ == "__main__":
    main()
