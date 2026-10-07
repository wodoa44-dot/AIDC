# Phase 0 identification protocol

Record passive 10-50 ms NVML polls, total energy counters where available, and synchronized node/PDU active power. Store command send/ack times, timestamps, device/firmware/driver identity, workload progress, and request latency. Poll period is not sensor update or averaging resolution: measure repeated readings and lag separately.

On a safe test system, use five or six filler setpoints including zero. Repeat rising and falling steps at matched workload phases. Measure delay to first physical response, 10-90% response, slew, steady-state headroom, termination tail, and main workload throughput or resume latency. Keep device control out of this repository's offline runner.

Define `eta_tracking = integral(actual incremental power over matched baseline) / integral(commanded ideal incremental power)` over a declared observation window. Estimate matched baseline with the same workload and phase; report interference and main-workload power shifts separately. Measure cap response and any firmware smoothing support through read-only queries before testing control. Obtain PCC ramp allowance, critical bands, and SLA from operators or an identified grid model before treating proxy scenarios as compliance tests.

For dynamic serving metadata, synchronize vLLM server, NVML/CPU collectors, and request-client clocks. Record the logger interval definition, timestamp precision, flush/transport delay, and first raw power sample after each `start_time_exc`. Replay only counters whose availability precedes forecast issuance. Capture per-request arrival, queue entry, prefill/decode, completion, and p95/p99 latency if B3/B4 or SLA conclusions are required.
