# Benchmark cases

The suite runs the scripts below from `op_tests/op_benchmarks/triton/` in
aiter. Each one sweeps its own set of shapes internally, so the workbook
holds far more rows than there are scripts: 795 data points over 41
benchmarks that produced numbers on both configurations.

50 scripts are launched. 8 of them fail with the pass
enabled while passing on base LLVM, and 1 is excluded upstream, which is
why 41 rather than 50 appear in the comparison.

## Batched GEMM (5)

- `bench_batched_gemm_a16wfp4`
- `bench_batched_gemm_a8w8`
- `bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant`
- `bench_batched_gemm_afp4wfp4`
- `bench_batched_gemm_bf16`

## GEMM (14)

- `bench_ff_a16w16_fused`
- `bench_fused_gemm_a8w8_blockscale_a16w16`
- `bench_fused_gemm_afp4wfp4_a16w16`
- `bench_gemm_a16w16`
- `bench_gemm_a16w16_gated`
- `bench_gemm_a16w16_gating`
- `bench_gemm_a16w8_blockscale`
- `bench_gemm_a8w8`
- `bench_gemm_a8w8_blockscale`
- `bench_gemm_a8w8_per_token_scale`
- `bench_gemm_a8wfp4`
- `bench_gemm_afp4wfp4`
- `bench_gemm_afp4wfp4_pre_quant_atomic`
- `bench_gmm`

## MoE (8)

- `bench_moe`
- `bench_moe_gemm_a16w4` `--shape 7168 4096 --experts 256 8`
- `bench_moe_gemm_a4w4` `--shape 7168 4096 --experts 256 8`
- `bench_moe_gemm_a8w4` `--shape 7168 4096 --experts 256 8`
- `bench_moe_gemm_a8w8` `--shape 7168 4096 --experts 256 8`
- `bench_moe_gemm_a8w8_blockscale` `--shape 7168 4096 --experts 256 8`
- `bench_moe_gemm_int8_smoothquant` `--shape 7168 4096 --experts 256 8`
- `bench_moe_mx`

## Attention (18)

- `bench_batch_prefill`
- `bench_deepgemm_attention`
- `bench_extend_attention`
- `bench_fav3_sage` `-b 4 -hq 32 -sq 1024 -d 128`
- `bench_fav3_sage_mxfp4` `-b 4 -hq 32 -sq 1024 -d 128`
- `bench_fp8_mqa_logits`
- `bench_hstu_attn`
- `bench_la_paged_decode`
- `bench_mha`
- `bench_mhc`
- `bench_mla`
- `bench_mla_decode`
- `bench_mla_decode_rope`
- `bench_moe_align_block_size`
- `bench_pa_decode`
- `bench_pa_prefill`
- `bench_sage`
- `bench_unified_attention`

## Normalization (1)

- `bench_rmsnorm`

## Routing/TopK (2)

- `bench_moe_routing_sigmoid_top1_fused`
- `bench_topk`

## Other (2)

- `bench_cache_copy`
- `bench_rope`

