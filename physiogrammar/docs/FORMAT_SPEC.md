# PhysioGrammar format

A record starts with a header:

```text
PG1|n=<samples>|bits=<bits>|lo=<minimum>|hi=<maximum>
```

The waveform is min-max quantized to integers in `[0, 2^bits-1]`. Each following line describes one consecutive segment.

## Primitives

For local sample index `t = 0, ..., n-1`:

| Code | Reconstruction | Parameters | Fit used in the paper |
|---|---|---|---|
| `C` | `q(t)=c` | `n,c` | rounded segment mean |
| `L` | line from `a` to `b` | `n,a,b` | least-squares line, rounded endpoints |
| `E` | `q(t)=c+a exp(-t/tau)` | `n,c,a,tau` | `tau` grid, conditional least squares |
| `O` | `q(t)=c+a sin(2 pi k t/n)+b cos(2 pi k t/n)` | `n,c,a,b,k` | dominant detrended FFT bin, then least squares |
| `T` | `q(t)=c+a exp(-0.5((t-mu)/sigma)^2)` | `n,c,a,mu,sigma` | deterministic center and width grid, then least squares |

Exponential time constants use `{n/12, n/8, n/4, n/2, n, 2n}` after integer rounding and duplicate removal. Gaussian widths use `{n/30, n/20, n/12, n/8, n/5}`. The transient center is the largest absolute deviation from the median of the segment edges.

## Text syntax

```text
C|n|c
L|n|a|b
E|n|c|a|tau
O|n|c|a|b|k
T|n|c|a|mu|sigma
```

Example:

```text
PG1|n=500|bits=8|lo=-0.84|hi=1.02
L|50|122|131
O|100|127|22|-9|2
T|50|124|41|27|6
...
```

## Token objective

For a candidate record `s_r`, the dynamic program uses the additive segment-level rate surrogate

```text
sum_r |T(s_r)| + lambda * D_r + gamma
```

where `T` is the configured tokenizer, `D_r` is normalized squared reconstruction error and `gamma` is the segment penalty.

Subword tokenization can have boundary effects after records are concatenated. For that reason, the final representation is serialized first and then tokenized as one complete string. All reported token rates use this whole-string count. `scripts/audit_token_boundaries.py` measures the difference between the additive surrogate and final token accounting.
