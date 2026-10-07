"""Read-only review of completed diagnostic artifacts; no tuning or retraining."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'outputs/full'
OUT = ROOT / 'outputs/controller_review_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = {'source': str(SOURCE), 'budget': 1., 'family': 'B1_fixed',
           'case_rule': 'rank run-balanced whole p95 P4 minus persistence; select min, middle, max; within run largest absolute paired window p95 difference',
           'scope': 'existing artifacts only; no new tuning, simulation or model training'}
    (OUT / 'config.json').write_text(json.dumps(cfg, indent=2))
    inputs = ['control_per_run.csv', 'control_summary.csv', 'evaluation_design.csv', 'window_metrics.csv']
    hashes = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in inputs}
    runs = pd.read_csv(SOURCE / 'control_per_run.csv')
    summary = pd.read_csv(SOURCE / 'control_summary.csv')
    selected = summary[(summary.budget == 1) & summary.family.isin(['B1_fixed', 'B2_tuned', 'D_mpc_trajectory'])].copy()
    selected.to_csv(OUT / 'phase_metrics.csv', index=False)
    deltas = []
    for (family, region), g in selected.groupby(['family', 'region']):
        base = g[g.input == 'persistence'].iloc[0]
        for _, row in g[g.input != 'persistence'].iterrows():
            deltas.append({'family': family, 'input': row.input, 'region': region,
                           **{m: row[m] - base[m] for m in ['p95_ramp', 'max_ramp', 'fast_throughput_wh', 'ess_throughput_wh']}})
    pd.DataFrame(deltas).to_csv(OUT / 'phase_paired_mean_differences.csv', index=False)
    design = pd.read_csv(SOURCE / 'evaluation_design.csv')
    windows = pd.read_csv(SOURCE / 'window_metrics.csv')
    rows = []
    # Every matching window in the fixed-policy comparison, no selection by error.
    fixed = design[(design.family == 'B1_fixed') & (design.budget == 1)]
    for _, d in fixed.iterrows():
        w = windows[(windows.task == d.task) & (windows.region == 'whole')]
        for _, x in w.iterrows():
            folder = SOURCE / 'control_cache' / d.task
            z = pd.read_parquet(folder / f'{x.run_id}_s{x.segment}_w{x.window}.parquet')
            dt = np.diff(z.issuance_s.to_numpy())
            load = np.diff(z.current_w.to_numpy()) / dt
            storage = np.diff((z.fast_w + z.ess_w).to_numpy()) / dt
            pcc = np.diff(z.pcc_w.to_numpy()) / dt
            np.testing.assert_allclose(pcc, load - storage, atol=1e-7)
            i = int(np.argmax(np.abs(pcc))) + 1
            rows.append({'input': d.input, 'run_id': x.run_id, 'segment': x.segment, 'window': x.window,
                         'max_ramp_region': 'recovery' if z.recovery.iloc[i] else 'operation',
                         'max_ramp_elapsed_s': z.issuance_s.iloc[i] - z.issuance_s.iloc[0],
                         'signed_load_ramp_w_s': load[i-1], 'signed_storage_ramp_w_s': storage[i-1],
                         'signed_pcc_ramp_w_s': pcc[i-1],
                         'storage_amplifies_load_at_peak': bool(abs(pcc[i-1]) > abs(load[i-1]) + 1e-7),
                         'recovery_start_fast_requested_w': z.fast_requested_w.iloc[200],
                         'recovery_start_fast_applied_w': z.fast_w.iloc[200],
                         'recovery_start_ess_requested_w': z.ess_requested_w.iloc[200],
                         'recovery_start_ess_applied_w': z.ess_w.iloc[200]})
    peaks = pd.DataFrame(rows)
    peaks.to_csv(OUT / 'peak_ramp_decomposition.csv', index=False)
    counts = peaks.groupby('input').agg(n_windows=('window', 'size'), recovery_peak_fraction=('max_ramp_region', lambda x: x.eq('recovery').mean()), storage_amplification_fraction=('storage_amplifies_load_at_peak', 'mean')).reset_index()
    counts.to_csv(OUT / 'peak_summary.csv', index=False)
    g = runs[(runs.family == 'B1_fixed') & (runs.budget == 1) & (runs.region == 'whole')]
    ranked = g.pivot(index='run_id', columns='input', values='p95_ramp')
    ranked['delta'] = ranked.P4 - ranked.persistence
    ranked = ranked.reset_index().sort_values(['delta', 'run_id'])
    cases = []
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    cache_hashes = {}
    for axrow, (label, position) in zip(axes, [('best', 0), ('middle', len(ranked)//2), ('worst', len(ranked)-1)]):
        run = ranked.iloc[position]
        fold = int(g[g.run_id == run.run_id].fold.iloc[0])
        ds = fixed[fixed.fold == fold].set_index('input')
        a = windows[(windows.task == ds.loc['P4', 'task']) & (windows.run_id == run.run_id) & (windows.region == 'whole')]
        b = windows[(windows.task == ds.loc['persistence', 'task']) & (windows.run_id == run.run_id) & (windows.region == 'whole')]
        paired = a.merge(b, on=['run_id', 'segment', 'window'], suffixes=('_p4', '_base'), validate='one_to_one')
        paired['delta'] = paired.p95_ramp_p4 - paired.p95_ramp_base
        chosen = paired.assign(magnitude=paired.delta.abs()).sort_values(['magnitude', 'segment', 'window'], ascending=[False, True, True]).iloc[0]
        traces = {}
        for inp in ['persistence', 'P4', 'perfect']:
            path = SOURCE / 'control_cache' / ds.loc[inp, 'task'] / f'{run.run_id}_s{chosen.segment}_w{chosen.window}.parquet'
            cache_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            traces[inp] = pd.read_parquet(path)
        z = traces['P4']; t = z.issuance_s - z.issuance_s.iloc[0]
        peak = int(np.argmax(abs(np.diff(z.pcc_w) / np.diff(z.issuance_s)))) + 1
        axrow[0].plot(t, z.current_w, color='gray', label='load')
        axrow[0].plot(t, z.forecast_w, ':', label='P4 endpoint')
        for inp, trace in traces.items(): axrow[0].plot(t, trace.pcc_w, label=inp + ' PCC', alpha=.8)
        axrow[1].plot(t, z.fast_w, label='fast applied W')
        axrow[1].plot(t, z.ess_w, label='ESS applied W')
        for ax in axrow:
            ax.axvspan(40, 60, color='orange', alpha=.1)
            ax.axvline(t.iloc[peak], color='red', linestyle='--', label='P4 max ramp')
            ax.set(xlabel='Elapsed seconds (recovery starts at 40 s)', ylabel='W')
            ax.legend(fontsize=7)
        axrow[0].set_title(f'{label}: run delta={run.delta:.1f}, window delta={chosen.delta:.1f} W/s')
        axrow[1].set_title(str(run.run_id))
        cases.append({'case': label, 'run_id': run.run_id, 'run_delta_p95_w_s': run.delta, 'segment': chosen.segment, 'window': chosen.window, 'window_delta_p95_w_s': chosen.delta, 'peak_elapsed_s': t.iloc[peak]})
        z.to_csv(OUT / f'case_{label}_plotting.csv', index=False)
    pd.DataFrame(cases).to_csv(OUT / 'case_selection.csv', index=False)
    fig.suptitle('Fixed policy B1 | E1 PCC proxy | reused exploratory evaluation | 23 runs / 4 assumed days\nRank labels do not imply success; no verified independent groups; no CI in this figure', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / 'fixed_policy_cases.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'fixed_policy_cases.svg', bbox_inches='tight')
    plt.close(fig)
    focus = pd.DataFrame(deltas)
    text = '''# 제어 활용 방식 후속 검토

기존 전체 진단 결과를 재분석했다. 새 학습·튜닝·controller 탐색은 수행하지 않았고, 기존 결과는 보존했다. Primary budget=1.0, E1 PCC proxy, 23개 run 및 가정 기반 4개 날짜다. 독립 수집 group 검증은 여전히 미완료다.

## 수식에서 확인한 동작

Endpoint controller는 `aim=d+w(f−d)`로 reference를 이동시킨 뒤 `request=d−reference`를 한 bin 지연 적용한다. 미래 상승값은 현재 reference를 높이므로 다른 상태가 같을 때 현재 방전 요청을 줄이거나 충전을 유도할 수 있다. 이 방향 자체를 오류로 단정할 수는 없지만, 미래값이 정확하다고 현재 PCC ramp가 자동으로 개선되는 구조는 아니다. Endpoint 하나로 상승 시점이나 중간 경로를 복원할 수도 없다.

복구는 40초 시점부터 매 bin에 `request=(energy−initial)×3600×efficiency_factor/dt`를 사용한다. 즉 한 bin 복원을 요청하고 출력·slew·예산 한도로 제한한다. 남은 20초에 복구량을 분산하는 목적함수는 없다. MPC도 이 공통 복구 정책을 사용하며 운전 horizon 끝에서 절단한다. 복구 직전 상태와 복구 경계의 PCC ramp를 동시에 최적화하지 않는다.

MPC persistence는 모든 미래 부하를 현재값으로 고정한다. 초기 PCC=부하, 장치 출력=0인 상태에서 0출력은 ramp 비용과 처리량 비용 모두 0으로 만들어 계속 유지된다. 따라서 이 비교군은 실제로 무제어와 같다. MPC perfect trajectory 대비 차이는 같은 MPC 입력의 차이로 읽어야 하며, 기존 반응형 대비 개선과 구분해야 한다.

## 구간별 입력 교체 차이

아래는 같은 family의 persistence를 뺀 run-balanced 평균 차이다. p95/max는 W/s, 처리량은 Wh다. 전체 p95는 구간 p95의 합이나 가중합이 아니다. 운전→복구 경계 ramp는 복구에 포함된다. B2는 입력별 inner 선택 parameter가 달라 입력의 순수 효과로 읽을 수 없다.

'''
    text += focus.to_markdown(index=False, floatfmt='.2f')
    text += '\n\n## 최대 ramp의 부하·장치 출력 분해\n\n각 창 최대 PCC ramp에서 `r_PCC=r_load−r_storage`를 실제 timestamp로 확인했다. 아래 비율은 window별 기술 통계이며 독립 표본 비율이나 인과 효과가 아니다. 전체 23개 run의 평균과는 다른 집계다.\n\n' + counts.to_markdown(index=False, floatfmt='.3f')
    text += '\n\n## 차이가 있는 고정 정책 사례\n\n기존 C_joint 사례는 P4 선택 weight가 모두 0이라 차이가 없었다. 이번에는 고정 정책 B1의 전체 run 차이를 최소·중앙·최대로 순위화하고, 해당 run에서 paired window p95 차이의 절대값이 가장 큰 창을 표시했다. 사후 기술적 사례이며 개발·튜닝 선택에 사용하지 않는다.\n\n' + pd.DataFrame(cases).to_markdown(index=False, floatfmt='.2f')
    text += '''

![고정 정책 사례](fixed_policy_cases.png)

## 다음 개발 판단

우선 검토 대상은 복구를 남은 시간에 분산하고 운전 마지막 계획과 연결하는 controller 설계, 그리고 actuator 적용 시각에 맞춘 200ms 경로 예측이다. 이번 결과만으로 새 설계의 개선을 입증하지 않았다. 수정 비교 시 동일 새 controller의 persistence·learned·perfect 경로, 동일 장치·예산·복구 종료 조건을 유지해야 한다. 목적함수·후보를 실행 전에 고정하고 inner 선택만 허용해야 한다.

완벽한 궤적도 기존 반응형의 전체 p95보다 높았으므로 새 모델 구조를 즉시 확대할 근거는 부족하다. 현재 평가셋에서 좋은 조합을 찾을 때까지 탐색하지 않는다. 먼저 timestamp 정의·실제 actuator delay 및 slew·효율·수집 session ID와 hardware/workload 교락을 확인할 자료를 확보하는 것이 타당하다.
'''
    (OUT / 'report_ko.md').write_text(text)
    body = markdown.markdown(text, extensions=['tables'])
    (OUT / 'report_ko.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>제어 활용 방식 후속 검토</title><style>body{font-family:sans-serif;max-width:1200px;margin:32px auto;padding:16px;line-height:1.6}img{max-width:100%}table{font-size:12px;display:block;overflow:auto}td,th{padding:6px;border:1px solid #ddd}</style>' + body + '</html>')
    assert hashes == {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in inputs}
    (OUT / 'completion_status.json').write_text(json.dumps({'status':'completed', 'new_training':False, 'new_tuning':False, 'source_hashes':hashes, 'plotted_cache_hashes':cache_hashes, 'ramp_identity_checked_windows':len(peaks), 'independent_group_verification':'blocked_missing_collection_evidence'}, indent=2))
    print('COMPLETED', OUT)
    print(counts.to_string(index=False))
    print(pd.DataFrame(cases).to_string(index=False))


if __name__ == '__main__':
    main()
