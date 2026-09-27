"""Run-balanced paired reporting; outer and reused-holdout assessments stay separate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from corrected_capacity import crps
from expanded import QCOL


def report(out, cfg, results):
    out = Path(out)
    rows, usage = [], []
    keys = ['assessment', 'resolution_ms', 'horizon_ms', 'outer_fold', 'pair', 'condition', 'seed', 'run_id']
    seen = set()
    pairing = {}
    for result in results:
        f = pd.read_parquet(result['prediction'])
        identity = result['identity']
        unit = tuple(identity[k] for k in keys[:-1])
        if unit in seen:
            raise ValueError(f'Duplicate prediction task {unit}')
        seen.add(unit)
        assert not f.duplicated(['target_index']).any()
        pairkey = tuple(identity[k] for k in ['assessment', 'resolution_ms', 'horizon_ms', 'outer_fold', 'pair', 'seed'])
        targets = f[['target_index', 'run_id', 'target_s', 'y']].reset_index(drop=True)
        if pairkey in pairing:
            pd.testing.assert_frame_equal(pairing.pop(pairkey), targets)
        else:
            pairing[pairkey] = targets
        q = cfg['quantiles']
        p = f[QCOL(q)].to_numpy()
        f['crps_w'] = crps(f.y.to_numpy(), p, q)
        f['mae_w'] = np.abs(f.y.to_numpy()-p[:, q.index(.5)])
        for level, lo, hi in [(80, .1, .9), (90, .05, .95)]:
            f[f'coverage_{level}'] = (f.y.to_numpy() >= p[:, q.index(lo)]) & (f.y.to_numpy() <= p[:, q.index(hi)])
            f[f'width_{level}_w'] = p[:, q.index(hi)]-p[:, q.index(lo)]
        # Fixed physical criterion, never selected using assessment scores.
        transition = np.abs(f.y-f.current) >= 100.0
        f['transition_crps_w'] = f.crps_w.where(transition)
        f['transition_mae_w'] = f.mae_w.where(transition)
        f['transition_fraction'] = transition.astype(float)
        numeric = ['crps_w', 'mae_w', 'coverage_80', 'coverage_90', 'width_80_w', 'width_90_w',
                   'transition_crps_w', 'transition_mae_w', 'transition_fraction']
        rows.append(f.groupby(keys, dropna=False)[numeric].mean().reset_index())
        u = pd.read_csv(result['usage'])
        for k, v in identity.items():
            if k not in ['parameters', 'selection']:
                u[k] = v
        usage.append(u)
    assert not pairing, 'Unpaired final predictions'
    metrics = pd.concat(rows, ignore_index=True)
    assert not metrics.duplicated(keys).any()
    metrics.to_csv(out/'per_run_seed_metrics.csv', index=False)
    # Average seeds and repeated assessments of a run before assigning equal run weights.
    runkeys = ['assessment', 'resolution_ms', 'horizon_ms', 'pair', 'condition', 'run_id']
    run = metrics.groupby(runkeys)[numeric].mean().reset_index()
    run.to_csv(out/'per_run_metrics.csv', index=False)
    agg = run.groupby(runkeys[:-1])[numeric].mean().reset_index()
    agg.to_csv(out/'aggregate_metrics.csv', index=False)
    rng = np.random.default_rng(cfg['seed'])
    comparisons = []
    for key, group in run.groupby(['assessment', 'resolution_ms', 'horizon_ms']):
        wide = group.pivot(index='run_id', columns=['pair', 'condition'], values='crps_w')
        for pair in ['legacy_budget', 'improved_controlled', 'independently_tuned']:
            hist = wide[pair, 'history_only'].to_numpy()
            meta = wide[pair, 'history_all_meta'].to_numpy()
            legacy = wide['legacy_budget', 'history_only'].to_numpy()
            indices = rng.integers(0, len(hist), size=(cfg['bootstrap_replicates'], len(hist)))
            d = meta-hist
            gain = 1-meta[indices].mean(axis=1)/hist[indices].mean(axis=1)
            overall = 1-meta[indices].mean(axis=1)/legacy[indices].mean(axis=1)
            comparisons.append(dict(zip(['assessment', 'resolution_ms', 'horizon_ms'], key)) | {
                'pair': pair, 'n_runs': len(hist), 'history_only': hist.mean(), 'history_all_meta': meta.mean(),
                'delta_crps_w': d.mean(), 'meta_gain': 1-meta.mean()/hist.mean(),
                'delta_ci_low_w': np.percentile(d[indices].mean(axis=1), 2.5),
                'delta_ci_high_w': np.percentile(d[indices].mean(axis=1), 97.5),
                'meta_gain_ci_low': np.percentile(gain, 2.5), 'meta_gain_ci_high': np.percentile(gain, 97.5),
                'history_improvement_vs_legacy': 1-hist.mean()/legacy.mean(),
                'meta_improvement_vs_legacy_history': 1-meta.mean()/legacy.mean(),
                'overall_gain_ci_low': np.percentile(overall, 2.5), 'overall_gain_ci_high': np.percentile(overall, 97.5),
                'improving_run_fraction': float((d<0).mean())})
    summary = pd.DataFrame(comparisons)
    summary.to_csv(out/'comparison_summary.csv', index=False)
    split = pd.concat(usage, ignore_index=True)
    split.to_csv(out/'feature_split_usage.csv', index=False)
    split['is_metadata'] = ~split.feature.str.startswith('power_')
    split.groupby(['assessment', 'pair', 'condition', 'is_metadata'])[['splits', 'gain']].sum().to_csv(out/'metadata_split_usage.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for assessment, s in summary.groupby('assessment'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for pair, g in s.groupby('pair'):
            label = g.resolution_ms.astype(str)+'/'+g.horizon_ms.astype(str)
            axes[0].plot(label, 100*g.meta_gain, marker='.', label=pair)
            axes[1].plot(label, 100*g.meta_improvement_vs_legacy_history, marker='.', label=pair)
        for ax, title in zip(axes, ['Metadata gain (%)', 'Overall improvement vs legacy history (%)']):
            ax.set_title(title)
            ax.axhline(0, color='gray', linewidth=.7)
            ax.tick_params(axis='x', rotation=90)
            ax.set_xlabel('Resolution / horizon (ms)')
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out/f'{assessment}_gains.png', dpi=160)
        plt.close(fig)
    label = 'Development-only integration results; not corrected main results.' if cfg.get('development_integration') else 'Corrected experiment results.'
    table = summary[['assessment', 'resolution_ms', 'horizon_ms', 'pair', 'meta_gain', 'meta_improvement_vs_legacy_history']].to_string(index=False)
    (out/'report_en.md').write_text('# Corrected capacity assessment\n\n'+label+'\n\n'
        'All final models are fresh fits. Grouped outer selections use only their own outer-training groups. '
        'Final development selections are frozen before scoring the four previously inspected holdout runs; '
        'those results are a reused-holdout assessment, not fresh confirmation.\n\n'
        'Legacy sampling takes at most 250 evenly spaced positions in the pooled eligible training rows, '
        'with equal row weights and 12 rounds. Its history is selected by history-only grouped inner validation. '
        'This is a newly fitted legacy-budget reference, not an exact replay of historical temporal training restrictions. '
        'Improved controlled fits share history-only-selected parameters, power features, sample indices, and weights. '
        'Independent fits use condition-specific selections. All pairs share targets and seeds 17, 42, and 2026.\n\n'
        'Scores first average within each run, then over seeds and repeated folds, then equally over runs. '
        'META gain is 1 minus metadata CRPS divided by paired history CRPS. Overall improvement uses the newly fitted '
        'legacy history-only CRPS denominator and is a distinct quantity. Uncertainty is a paired run bootstrap of '
        'fixed predictions; it excludes retraining and selection uncertainty. Small run counts and confounded static '
        'metadata limit interpretation; gains are not causal effects.\n\n'
        'Transition metrics use the fixed criterion |target power - issuance power| >= 100 W. '
        'Coverage and width are reported at 80% and 90%. Split usage describes fitted trees, not causal importance. '
        'Each task stores its exact indices, weights, encoder, requested/resolved/fitted parameters, and 15-quantile predictions.\n\n'
        '```text\n'+table+'\n```\n')
