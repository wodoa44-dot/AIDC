import numpy as np
import pandas as pd
from aidc_data.phase0.response import response_metrics


def fixture():
    t = np.arange(-100, 301) / 50
    samples = pd.DataFrame({'trial_id': 'test', 'gpu_uuid': 'fake',
                            'monotonic_start_ns': np.rint((t + 10) * 1e9).astype('int64'),
                            'api_duration_ns': 1000, 'power_w': np.where(t < .5, 200., 100.),
                            'evidence_type': 'synthetic_test'})
    command = {'command_id': 'c', 'trial_id': 'test', 'gpu_uuid': 'fake', 'success': True,
               'intended_monotonic_ns': 10_000_000_000, 'return_monotonic_ns': 10_010_000_000,
               'evidence_type': 'synthetic_test'}
    return samples, command


def test_known_down_response_and_evidence():
    samples, command = fixture()
    result = response_metrics(samples, command)
    assert result['valid'] and result['d_act_90_s'] == .5
    assert result['delta_w'] == -100
    assert result['parent_evidence_type'] == 'synthetic_test'
    assert not result['causal_effect_identified']


def test_gap_and_flat_response_rejected():
    samples, command = fixture()
    broken = samples.drop(samples.index[110:130])
    assert response_metrics(broken, command)['reason'] == 'sampling_gap'
    samples['power_w'] = 200.
    assert not response_metrics(samples, command)['valid']


def test_next_command_and_failed_command_rejected():
    samples, command = fixture()
    assert not response_metrics(samples, command, end_ns=10_200_000_000)['valid']
    command['success'] = False
    assert response_metrics(samples, command)['reason'] == 'command_failed'


def test_mixed_evidence_rejected():
    samples, command = fixture()
    command['evidence_type'] = 'measured_new'
    assert response_metrics(samples, command)['reason'] == 'mixed_or_unsupported_evidence'
