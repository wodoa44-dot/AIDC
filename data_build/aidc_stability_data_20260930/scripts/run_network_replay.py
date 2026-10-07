"""ANDES Kundur benchmark replay of archived, assumed-actuator E1 outputs."""
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / '.grid_deps'))
import andes

CASE = Path(andes.get_case('kundur/kundur_full.xlsx'))
SOURCE = ROOT / 'experiments/aidc_grid_metadata_20260929/outputs/full_20260928T180455Z/control_traces.parquet'
CONTROLLERS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + '\n')


def simulate(delta_w, dt=.02, copies=1000, hold_s=.2, warmup_s=2., recovery_s=5.):
    ss = andes.load(str(CASE), setup=False, no_output=True, no_undill=True,
                    pycode_path=str(PROJECT / '.grid_pycode'))
    if not ss._load_calls():
        ss.prepare(quick=True, nomp=True)
    ss.Toggle.u.v[:] = [0] * ss.Toggle.n
    ss.PQ.config.p2p = ss.PQ.config.q2q = 1.
    ss.PQ.config.p2i = ss.PQ.config.p2z = 0.
    ss.PQ.config.q2i = ss.PQ.config.q2z = 0.
    base_w = float(ss.config.mva) * 1e6
    original_p = float(ss.PQ.p0.v[0])
    for i, value in enumerate(delta_w):
        ss.add('Alter', {'model': 'PQ', 'dev': 'PQ_0', 'src': 'Ppf',
                        't': round(warmup_s + i * hold_s, 8), 'method': '=',
                        'amount': original_p + float(value) * copies / base_w})
    clear_time = warmup_s + len(delta_w) * hold_s
    ss.add('Alter', {'model': 'PQ', 'dev': 'PQ_0', 'src': 'Ppf',
                    't': round(clear_time, 8), 'method': '=', 'amount': original_p})
    ss.setup()
    if not ss.PFlow.run():
        raise RuntimeError('benchmark power flow did not converge')
    initial_voltage = ss.Bus.v.v.copy()
    ss.TDS.config.tf = clear_time + recovery_s
    ss.TDS.config.no_tqdm = 1
    ss.TDS.config.fixt = 1
    ss.TDS.config.shrinkt = 0
    ss.TDS.config.tstep = dt
    ss.TDS.config.tol = 1e-8
    if not ss.TDS.run() or float(ss.dae.t) < ss.TDS.config.tf - 1e-7:
        raise RuntimeError('benchmark TDS did not reach final time')
    t = ss.dae.ts.t
    v = ss.dae.ts.y[:, ss.Bus.v.a]
    omega = ss.dae.ts.x[:, ss.GENROU.omega.a]
    weights = ss.GENROU.M.v * ss.GENROU.Sn.v
    frequency = np.average(omega, axis=1, weights=weights) * ss.config.freq
    bus_uid = ss.Bus.idx2uid(7)
    trace = pd.DataFrame({'time_s': t, 'coi_frequency_hz': frequency,
                          'bus7_voltage_pu': v[:, bus_uid],
                          'min_bus_voltage_pu': v.min(axis=1),
                          'max_bus_voltage_pu': v.max(axis=1),
                          'max_bus_abs_voltage_deviation_pu': abs(v - initial_voltage).max(axis=1),
                          'evidence_type': 'simulated'})
    sample_i = np.floor((t - warmup_s + 1e-8) / hold_s).astype(int)
    active = (sample_i >= 0) & (sample_i < len(delta_w))
    injected = np.zeros(len(t))
    injected[active] = np.asarray(delta_w)[sample_i[active]] * copies
    trace['scenario_delta_p_w'] = injected
    trace['scenario_delta_q_var'] = 0.
    f200 = np.interp(np.arange(0, t[-1] + 1e-8, .2), t, frequency)
    residual = float(np.max(np.abs(ss.dae.g)))
    metrics = {'max_abs_delta_frequency_hz': float(abs(frequency - ss.config.freq).max()),
               'frequency_nadir_hz': float(frequency.min()),
               'frequency_zenith_hz': float(frequency.max()),
               'max_abs_rocof_200ms_hz_per_s': float(abs(np.diff(f200) / .2).max()),
               'max_bus_abs_voltage_deviation_pu': float(abs(v - initial_voltage).max()),
               'bus7_voltage_min_pu': float(v[:, bus_uid].min()),
               'bus7_voltage_max_pu': float(v[:, bus_uid].max()),
               'final_algebraic_residual_pu': residual,
               'solver_dt_s': dt, 'evidence_type': 'simulated'}
    metadata = {'base_mva': float(ss.config.mva), 'nominal_frequency_hz': float(ss.config.freq),
                'models': {k: m.n for k, m in ss.models.items() if m.n},
                'load_bus': 7, 'load_id': 'PQ_0', 'original_load_p_pu': original_p,
                'original_load_q_pu': float(ss.PQ.q0.v[0]),
                'initial_bus_voltage_pu': initial_voltage.tolist(),
                'frequency_definition': 'inertia-weighted generator rotor speed, not a PCC meter',
                'clear_time_s': clear_time}
    return trace, metrics, metadata


def validation(log):
    with contextlib.redirect_stdout(log):
        equilibrium, em, meta = simulate(np.zeros(50))
        step, sm, _ = simulate(np.full(100, 1000.), recovery_s=10.)
        half, _, _ = simulate(np.full(100, 1000.), dt=.01, recovery_s=10.)
    errors = {column: float(np.max(abs(step[column] - np.interp(step.time_s, half.time_s, half[column]))))
              for column in ['coi_frequency_hz', 'bus7_voltage_pu']}
    checks = {'equilibrium_frequency': em['max_abs_delta_frequency_hz'] < 1e-6,
              'equilibrium_voltage': em['max_bus_abs_voltage_deviation_pu'] < 1e-6,
              'positive_step_lowers_frequency': bool(step.loc[step.time_s.between(2.2, 3.), 'coi_frequency_hz'].mean() < meta['nominal_frequency_hz']),
              'half_step_frequency': errors['coi_frequency_hz'] < 1e-4,
              'half_step_voltage': errors['bus7_voltage_pu'] < 1e-4,
              'final_algebraic_balance': sm['final_algebraic_residual_pu'] < 1e-6}
    return {'passed': all(checks.values()), 'checks': checks,
            'half_step_max_errors': errors, 'step_w_per_node': 1000.,
            'step_evidence_type': 'synthetic_test', 'model': meta}, equilibrium, step, half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-id', default='remaining_20260929')
    parser.add_argument('--max-runs', type=int, default=0)
    args = parser.parse_args()
    out = PROJECT / 'outputs' / args.build_id / 'network_replay'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'status.json').exists():
        raise RuntimeError('Use a fresh build-id to preserve the prior run')
    started = datetime.now(timezone.utc).isoformat()
    dump(out / 'status.json', {'status': 'running', 'started_utc': started})
    with (out / 'execution.log').open('w') as log:
        try:
            checks, eq, step, half = validation(log)
            dump(out / 'validation.json', checks)
            for name, frame in [('equilibrium', eq), ('step', step), ('step_half', half)]:
                frame.to_parquet(out / f'verification_{name}.parquet', index=False)
            if not checks['passed']:
                raise RuntimeError('network validation failed')
            data = pd.read_parquet(SOURCE)
            metrics, traces, manifest = [], [], []
            runs = sorted(data.run_id.unique())
            if args.max_runs:
                runs = runs[:args.max_runs]
            for run in runs:
                selected = None
                for segment, group in data[data.run_id == run].groupby('segment', sort=True):
                    pivot = group.pivot(index='issuance_s', columns='controller', values='pcc_w').sort_index()
                    if not set(CONTROLLERS) <= set(pivot.columns):
                        continue
                    pivot = pivot[CONTROLLERS].dropna().iloc[:50]
                    if len(pivot) == 50 and np.allclose(np.diff(pivot.index), .2):
                        selected = (segment, pivot)
                        break
                if selected is None:
                    manifest.append({'run_id': run, 'status': 'excluded', 'reason': 'no_common_contiguous_10s_window'})
                    continue
                segment, pivot = selected
                reference = float(pivot.C0.iloc[0])
                for controller in CONTROLLERS:
                    start = time.monotonic()
                    with contextlib.redirect_stdout(log):
                        frame, score, meta = simulate(pivot[controller].to_numpy() - reference)
                    scenario = f'{run}:{segment}:{controller}'
                    frame['scenario_id'] = scenario
                    frame['run_id'] = run
                    frame['controller'] = controller
                    score.update(scenario_id=scenario, run_id=run, segment=int(segment), controller=controller)
                    metrics.append(score)
                    traces.append(frame)
                    manifest.append({'scenario_id': scenario, 'run_id': run, 'segment': int(segment),
                                     'controller': controller, 'status': 'completed',
                                     'input_start_s': float(pivot.index[0]), 'input_bins': 50,
                                     'reference_node_power_w': reference, 'synchronized_copies': 1000,
                                     'input_evidence': 'assumed_actuator_E1', 'output_evidence': 'simulated'})
                    print(f'{len(metrics)} scenarios: {scenario} ({time.monotonic() - start:.1f}s)', flush=True)
                    dump(out / 'status.json', {'status': 'running', 'started_utc': started,
                                              'scenarios_completed': len(metrics),
                                              'updated_utc': datetime.now(timezone.utc).isoformat()})
            pd.DataFrame(metrics).to_csv(out / 'grid_metrics.csv', index=False)
            pd.concat(traces, ignore_index=True).to_parquet(out / 'grid_traces.parquet', index=False)
            pd.DataFrame(manifest).to_csv(out / 'scenario_manifest.csv', index=False)
            dump(out / 'grid_model_provenance.json', {
                'engine': 'ANDES', 'engine_version': andes.__version__, 'case_path': str(CASE),
                'case_sha256': digest(CASE), 'source_path': str(SOURCE), 'source_sha256': digest(SOURCE),
                'script_sha256': digest(Path(__file__)),
                'package_license': 'GPL-3.0',
                'license_path': str(PROJECT / '.grid_deps/andes-1.9.2.dist-info/LICENSE'),
                'upstream': 'https://github.com/CURENT/andes/tree/v1.9.2/andes/cases/kundur',
                'documentation': 'https://docs.andes.app/en/v1.9.2/',
                'model': checks['model'],
                'scenario': 'First contiguous common 10s window in first eligible segment of every run; 2s equilibrium then replay then 5s return to original load.',
                'load_mapping': 'PQ_0.Ppf = original_p + (controller_pcc_w - C0_first_w) * 1000 / system_base_w',
                'modifications': ['Disable original line trip', 'Use constant P/Q load conversion',
                                  'Change only active power at bus 7; hold original Q fixed'],
                'assumptions': ['1000 synchronous copies of node power deviation', 'archived E1 actuator assumptions retained',
                                'input is held for 200ms; solver samples are not new measurements'],
                'unsupported': ['facility validation', 'GPU performance', 'UPS switching', 'converter Q control',
                                'weak-grid sensitivity', 'intrinsic damping improvement', 'full-run replay'],
                'validation': checks,
            })
            dump(out / 'status.json', {'status': 'completed', 'scenarios_completed': len(metrics),
                                      'runs': len({m['run_id'] for m in metrics}),
                                      'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
                                      'scope': 'benchmark 10s-prefix replay; no physical control or facility measurements'})
        except Exception as exc:
            dump(out / 'status.json', {'status': 'failed', 'error': str(exc), 'started_utc': started})
            raise


if __name__ == '__main__':
    main()
