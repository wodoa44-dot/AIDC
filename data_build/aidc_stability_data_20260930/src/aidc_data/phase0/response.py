"""Offline observed response analysis; paired causal attribution is separate."""
import numpy as np


def response_metrics(samples, command, end_ns=None, baseline_s=1., plateau_s=1.,
                     dwell_s=.2, minimum_delta_w=5.):
    result = {'command_id': command['command_id'], 'trial_id': command['trial_id'],
              'valid': False, 'reason': '', 'd_act_90_s': None,
              'causal_effect_identified': False, 'evidence_type': 'derived',
              'parent_evidence_type': None}

    def invalid(reason):
        result['reason'] = reason
        return result

    if command.get('success') not in (True, 1):
        return invalid('command_failed')
    frame = samples.loc[(samples.trial_id == command['trial_id']) &
                        (samples.gpu_uuid == command['gpu_uuid'])].copy()
    if frame.empty:
        return invalid('no_matching_trial_gpu_samples')
    evidence = set(frame.evidence_type) | {command.get('evidence_type')}
    if len(evidence) != 1 or not evidence <= {'measured_new', 'synthetic_test'}:
        return invalid('mixed_or_unsupported_evidence')
    result['parent_evidence_type'] = next(iter(evidence))
    start, returned = int(command['intended_monotonic_ns']), int(command['return_monotonic_ns'])
    if returned < start:
        return invalid('command_clock_order')
    frame = frame.sort_values('monotonic_start_ns')
    t = (frame.monotonic_start_ns.to_numpy(dtype=np.int64) - start) / 1e9
    p = frame.power_w.to_numpy(dtype=float)
    if len(t) < 4 or np.any(np.diff(t) <= 0):
        return invalid('insufficient_or_duplicate_samples')
    if not np.isfinite(p).all():
        return invalid('power_read_error')
    step = float(np.median(np.diff(t)))
    before = (t >= -baseline_s) & (t < 0)
    after = t >= (returned - start) / 1e9
    if end_ns is not None:
        after &= frame.monotonic_start_ns.to_numpy() < end_ns
    ta, pa = t[after], p[after]
    if before.sum() < 3 or len(ta) < 3:
        return invalid('missing_baseline_or_response')
    if t[before][-1] - t[before][0] < baseline_s - 2 * step:
        return invalid('baseline_too_short')
    if ta[-1] - ta[0] < plateau_s + dwell_s:
        return invalid('response_window_too_short')
    window = (t >= -baseline_s) & (t <= ta[-1])
    if np.max(np.diff(t[window])) > 3 * step:
        return invalid('sampling_gap')
    tail = pa[ta >= ta[-1] - plateau_s]
    baseline, plateau = float(np.median(p[before])), float(np.median(tail))
    delta = plateau - baseline
    noise = max(float(np.std(p[before])), float(np.std(tail)))
    result.update(baseline_w=baseline, plateau_w=plateau, delta_w=delta,
                  response_window_s=float(ta[-1]), sample_interval_s=step,
                  command_api_latency_s=(returned - start) / 1e9)
    if abs(delta) < max(minimum_delta_w, 5 * noise):
        return invalid('no_stable_resolved_plateau')
    half = len(tail) // 2
    if half < 2 or abs(np.median(tail[:half]) - np.median(tail[half:])) > .1 * abs(delta):
        return invalid('plateau_drift')
    progress = (pa - baseline) / delta
    for i in np.flatnonzero(progress >= .9):
        j = np.searchsorted(ta, ta[i] + dwell_s)
        if j < len(ta) and np.all(progress[i:j + 1] >= .9):
            result.update(valid=True, reason='observed_response_only_unpaired',
                          d_act_90_s=float(ta[i]),
                          timing_resolution_s=max(step, float(frame.api_duration_ns.max()) / 1e9),
                          overshoot_fraction=float(max(0., progress.max() - 1.)))
            return result
    return invalid('no_sustained_90_percent_response')
