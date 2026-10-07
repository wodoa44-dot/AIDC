**Table A5_computational_cost. Measured deployment-package accuracy and computational cost.**

| Model                    | Representation      | Hardware                   |   Batch |   Trials |   Models / request |   Package (MB) |   Inference p95 (ms) |   End-to-end p50 (ms) |   End-to-end p95 (ms) |   Process max RSS (GiB) |   Grouped-outer MAE (W) |
|:-------------------------|:--------------------|:---------------------------|--------:|---------:|-------------------:|---------------:|---------------------:|----------------------:|----------------------:|------------------------:|------------------------:|
| Original deployment      | compact 3 quantiles | x86_64 CPU; 1 model thread |       1 |     1000 |                  6 |          0.417 |                0.419 |                 2.717 |                 2.838 |                   0.18  |                 581.352 |
| Failure-aware deployment | full 15 quantiles   | x86_64 CPU; 1 model thread |       1 |     1000 |                 30 |          4.273 |                4.689 |                 4.051 |                 4.847 |                   7.308 |                 495.815 |

_Warm single-row CPU benchmark for both horizons. max RSS is process high-water mark, not incremental model memory. Training time was not recorded here. Different quantile counts are explicit; latency is not a real-time guarantee._
