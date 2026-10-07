"""Summarize the new network results and retain historical provenance."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import platform
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / '.grid_deps'))
from run_network_replay import digest


def finish(out):
    network = out / 'network_replay'
    state = json.loads((network / 'status.json').read_text())
    if state['status'] != 'completed':
        raise RuntimeError('Network replay has not completed')
    m = pd.read_csv(network / 'grid_metrics.csv')
    trace = pd.read_parquet(network / 'grid_traces.parquet')
    columns = ['max_abs_delta_frequency_hz', 'max_bus_abs_voltage_deviation_pu',
               'max_abs_rocof_200ms_hz_per_s']
    aggregate = m.groupby('controller')[columns].mean()
    aggregate.to_csv(network / 'aggregate_metrics.csv')
    comparisons = []
    for column in columns:
        paired = m.pivot(index='run_id', columns='controller', values=column)
        difference = paired.C4 - paired.C3
        comparisons.append({'metric': column, 'paired_runs': len(paired),
                            'C3_mean': paired.C3.mean(), 'C4_mean': paired.C4.mean(),
                            'mean_reduction_percent': 100 * (1 - paired.C4.mean() / paired.C3.mean()),
                            'improved_runs': int((difference < -1e-12).sum()),
                            'worsened_runs': int((difference > 1e-12).sum()),
                            'tied_runs': int((abs(difference) <= 1e-12).sum())})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(network / 'paired_C4_vs_C3.csv', index=False)
    checks = {'unique_scenario_time': not trace.duplicated(['scenario_id', 'time_s']).any(),
              'finite_outputs': bool(np.isfinite(trace.select_dtypes('number')).all().all()),
              'only_simulated_evidence': bool(trace.evidence_type.eq('simulated').all()),
              'all_six_controllers_per_run': bool(m.groupby('run_id').controller.nunique().eq(6).all()),
              'algebraic_final_residual': bool(m.final_algebraic_residual_pu.lt(1e-6).all()),
              'C0_C1_identical_without_GPU_model': bool(np.allclose(
                  m[m.controller == 'C0'][columns], m[m.controller == 'C1'][columns], rtol=0, atol=1e-12))}
    (network / 'output_validation.json').write_text(json.dumps({'passed': all(checks.values()), 'checks': checks}, indent=2))
    if not all(checks.values()):
        raise RuntimeError('Network output validation failed')
    original = ROOT / 'experiments/aidc_grid_metadata_20260929/outputs/full_20260928T180455Z'
    folds = json.loads((original / 'split_manifest.json').read_text())
    fold_for_run = {run: f['fold'] for f in folds['folds'] for run in f['test_runs']}
    index = pd.read_parquet(PROJECT / 'outputs/main_20260929/forecast_index.parquet',
                            columns=['run_id', 'split_group_id']).drop_duplicates()
    lineage = pd.read_csv(network / 'scenario_manifest.csv')
    lineage['original_test_fold'] = lineage.run_id.map(fold_for_run)
    lineage = lineage.merge(index.rename(columns={'split_group_id': 'conservative_sweep_group'}),
                            on='run_id', how='left', validate='many_to_one')
    lineage.to_csv(network / 'scenario_lineage.csv', index=False)
    provenance = json.loads((network / 'grid_model_provenance.json').read_text())
    shutil.copyfile(provenance['case_path'], network / 'kundur_full_source.xlsx')
    shutil.copyfile(provenance['license_path'], network / 'ANDES_LICENSE.txt')
    versions = {name: importlib.metadata.version(name) for name in
                ['andes', 'kvxopt', 'numpy', 'scipy', 'sympy', 'pandas', 'matplotlib',
                 'openpyxl', 'xlsxwriter', 'dill', 'multiprocess', 'pathos', 'numba']}
    (network / 'environment.json').write_text(json.dumps({'python': sys.version,
                                                         'platform': platform.platform(),
                                                         'versions': versions}, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, column, label in zip(axes, columns[:2], ['Max frequency deviation (Hz)', 'Max bus voltage deviation (pu)']):
        ax.bar(aggregate.index, aggregate[column], color=['#757575', '#a0a0a0', '#257c83', '#315d9b', '#af4968', '#817245'])
        ax.set_ylabel(label)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    fig.suptitle('Kundur benchmark: 10 s input replay, assumed actuators')
    fig.savefig(network / 'controller_comparison.png', dpi=300)
    fig.savefig(network / 'controller_comparison.pdf')
    plt.close(fig)
    chosen_run = sorted(m.run_id.unique())[len(m.run_id.unique()) // 2]
    plotted = trace[(trace.run_id == chosen_run) & trace.controller.isin(['C0', 'C3', 'C4'])]
    plotted.to_csv(network / 'example_trace_figure_data.csv', index=False)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for controller, group in plotted.groupby('controller'):
        axes[0].plot(group.time_s, group.coi_frequency_hz, label=controller)
        axes[1].plot(group.time_s, group.bus7_voltage_pu, label=controller)
    axes[0].set_ylabel('Generator COI frequency (Hz)')
    axes[1].set_ylabel('Bus 7 voltage (pu)')
    axes[1].set_xlabel('Simulation time (s)')
    axes[0].legend()
    fig.suptitle(f'Benchmark replay: lexicographic median run\n{chosen_run}')
    fig.savefig(network / 'example_trace.png', dpi=300)
    plt.close(fig)
    inventory = json.loads((out / 'reconciled_status.json').read_text())
    final = {'all_requested_measurements_complete': False,
             'audit_files': inventory['file_count'],
             'D3_response_analysis': 'implemented_and_tested; measurements unavailable',
             'D4_network_replay': state,
             'D5_numeric_import': 'implemented_and_tested; no facility source supplied',
             'remaining': ['paired GPU command/power/performance trials', 'facility PCC/ESS/UPS measurements',
                           'UPS/converter/weak-grid scenario models and validation',
                           'longer trace windows and additional operating points'],
             'original_outputs_modified': False}
    (out / 'completion_status.json').write_text(json.dumps(final, indent=2) + '\n')
    lines = ['# 미완료 항목 후속 실행 결과', '',
             f'- 기존 프롬프트 프로젝트 {inventory["file_count"]:,}개 파일을 조사하고 CSV 헤더/Parquet 스키마를 확인했다.',
             '- 상위 상태의 미구현 표시는 오래된 기록이다. 독립 튜닝 90/90, ablation 130/130, 이벤트 분류 40/40 및 별도 이벤트 매칭 산출물이 이미 존재한다.',
             f'- ANDES {versions["andes"]} 공개 Kundur 모델로 {state["runs"]}개 run, {state["scenarios_completed"]}개 C0-C5 시나리오를 새로 실행했다.',
             '- 100 MVA/60 Hz, 10 bus/4 generator 모델. 동일한 초기 운전점에서 bus 7의 P만 변경하고 Q는 고정했다.',
             '- 각 run에서 첫 적격 segment의 공통 10초 구간을 재생했다. 2초 평형, 10초 재생, 5초 원래 부하 복귀. 원본 전체 길이의 결과는 아니다.',
             '- node GPU 부하 편차의 1,000개 동기 복제를 가정했다. 입력은 기존 E1 가정 기반 actuator 결과이며 실제 시설 계측이 아니다.',
             '- 주파수는 발전기 관성 가중 회전속도 기반 COI이며 PCC 주파수 계측이 아니다. 전압은 동적 네트워크의 bus 전압이다.', '',
             '## C4 대 C3 비교', '', '| 지표 | C3 평균 | C4 평균 | 평균 감소율 | 개선/악화/동률 run |',
             '|---|---:|---:|---:|---:|']
    for row in comparisons:
        lines.append(f'| {row["metric"]} | {row["C3_mean"]:.7g} | {row["C4_mean"]:.7g} | {row["mean_reduction_percent"]:.3f}% | {row["improved_runs"]}/{row["worsened_runs"]}/{row["tied_runs"]} |')
    lines += ['', '음의 감소율은 악화다. 독립 session은 검증되지 않았으며 192개 시나리오는 192개 독립 실측이 아니다.', '',
              '## 검증 및 한계', '',
              '- 평형 주파수/전압, 양의 부하 계단 응답 부호, 20ms/10ms 적분 간격 수렴, 최종 대수 방정식 잔차를 검증했다. 검증 수치는 network_replay/validation.json에 있다.',
              '- 출력 중복/유한값/증거 유형/제어기 수/C0-C1 동일성/최종 잔차를 추가 검증했다. 모든 시각의 전력 수지나 모든 operating point의 안정성을 인증한 것은 아니다.',
              '- D3 관측 응답 분석에 baseline/plateau/90% 도달+dwell/결측·gap·실패 판정을 구현했다. 실제 명령 로그가 없어 분석한 실측 명령은 0개다. paired 인과 효과와 서비스 성능은 미식별이다.',
              '- D5 수치 계측 importer에 단위·시간대·경계·중복·결측·증거 유형 검증을 구현했다. 실제 시설 파일은 생성하지 않았다. UPS categorical 로그는 별도 입력이 필요하다.',
              '- 실제 GPU cap/clock, ESS, UPS 설정을 변경하지 않았다. 온도 API가 지원되지 않는 현재 GPU 환경의 live 실측은 실행하지 않았다.',
              '- benchmark 결과로 실제 시설 안정화, GPU 성능 보존, UPS/인버터 Q 제어, 약계통 안정도 또는 고유 감쇠 개선을 주장할 수 없다.', '',
              '## 재현', '', '프로젝트 폴더에서 실행:', '', '```bash',
              'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/run_network_replay.py --build-id NEW_BUILD_ID',
              'PYTHONPATH=src python -m pytest -q tests',
              'PYTHONPATH=src python scripts/analyze_phase0.py --source TRIAL_DIR --output NEW_ANALYSIS_DIR',
              'PYTHONPATH=src python scripts/import_facility.py --source METER_CSV --output NEW_IMPORT_DIR',
              '```', '', '기존 실험과 원본 데이터는 수정하지 않았다. 새 소스 코드로 기존 build를 resume하면 code hash가 달라지므로 새 build_id를 사용해야 한다.']
    (out / 'report_ko.md').write_text('\n'.join(lines) + '\n')
    artifacts = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*'))
                 if p.is_file() and p.name != 'artifact_hashes.json'}
    (out / 'artifact_hashes.json').write_text(json.dumps(artifacts, indent=2) + '\n')
    print(comparison.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-id', default='remaining_20260929')
    args = parser.parse_args()
    finish(PROJECT / 'outputs' / args.build_id)
