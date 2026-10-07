#!/usr/bin/env python3
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

out=Path('outputs')/sys.argv[1];fig=out/'figures';fig.mkdir(exist_ok=True)
b=pd.read_parquet(out/'completed_bins_200ms.parquet');q=pd.read_csv(out/'quality_audit.csv')
run=b.run_id.iloc[0];g=b[b.run_id==run].head(500)
raw=pd.read_parquet(out/'canonical_runs'/f'{run}.parquet').head(5000)
plt.figure(figsize=(10,4));plt.plot(raw.elapsed_s,raw.gpu_power_w,label='native timestamp GPU sum',alpha=.5)
plt.plot(g.elapsed_s,g.gpu_power_w,label='completed 200ms',lw=1.5);plt.xlabel('elapsed s');plt.ylabel('8-GPU power W');plt.legend();plt.tight_layout();plt.savefig(fig/'raw_completed_power.png');plt.close()
plt.figure(figsize=(8,4));plt.hist(raw.elapsed_s.diff().dropna(),bins=100,range=(0,.1));plt.xlabel('native timestamp interval s');plt.ylabel('count');plt.tight_layout();plt.savefig(fig/'sampling_intervals.png');plt.close()
plt.figure(figsize=(8,4));plt.bar(['gap starts','bins excluded coverage'],[q.gap_starts.sum(),q.bins_excluded_coverage.sum()]);plt.ylabel('count');plt.tight_layout();plt.savefig(fig/'gaps_coverage.png');plt.close()
ev=pd.read_parquet(out/'events.parquet');one=ev[ev.run_id==run].head(1)
if len(one):
    e=one.iloc[0];lo=e.onset_time-pd.Timedelta(seconds=3);hi=e.detected_at+pd.Timedelta(seconds=5)
    h=b[(b.run_id==run)&(b.timestamp>=lo)&(b.timestamp<=hi)]
    plt.figure(figsize=(10,4));plt.plot(h.timestamp,h.gpu_power_w);plt.axvline(e.onset_time,color='orange',label='retrospective onset');plt.axvline(e.detected_at,color='red',label='online detected');plt.ylabel('8-GPU power W');plt.legend();plt.tight_layout();plt.savefig(fig/'event_onset_detected.png');plt.close()
meta=pd.read_parquet(out/'metadata.parquet');coverage=meta.groupby('field').value.apply(lambda x:x.notna().mean()).sort_values()
plt.figure(figsize=(8,4));coverage.plot.barh();plt.xlabel('fraction of canonical runs with value');plt.tight_layout();plt.savefig(fig/'metadata_coverage.png');plt.close()
p=out/'grid_frequency_traces.parquet'
if p.exists():
    grid=pd.read_parquet(p);s=grid[grid.scenario_id==grid.scenario_id.iloc[0]].head(500)
    fig_,ax=plt.subplots(2,1,figsize=(10,5),sharex=True)
    ax[0].plot(range(len(s)),s.p_load_scenario_w/1e6);ax[0].set_ylabel('scenario load ΔMW')
    ax[1].plot(range(len(s)),s.frequency_hz);ax[1].set_ylabel('simulated Hz');ax[1].set_xlabel('200ms bins')
    fig_.tight_layout();fig_.savefig(fig/'frequency_scenario.png');plt.close(fig_)
print(fig)
