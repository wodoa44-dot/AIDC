#!/usr/bin/env python3
"""Assumed E1 sensitivity on completed paired 1s fold-0 forecasts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from aidc_exp.control import rollout,summarize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',default='full_20260928T180455Z');ap.add_argument('--fold',type=int,default=0)
    args=ap.parse_args();project=Path(__file__).resolve().parents[1]
    mainout=project/'outputs'/args.id;out=mainout/'feasible_control_sensitivity_v1';out.mkdir(exist_ok=True)
    parts=mainout/'prediction_parts';f=args.fold;h=5;seed=2026
    paths={m:parts/f'f{f}_h{h}_s{seed}_{m}.parquet' for m in ('P2','P4')}
    if not all(p.exists() for p in paths.values()):raise RuntimeError('paired 1s predictions unavailable')
    p2=pd.read_parquet(paths['P2']);p4=pd.read_parquet(paths['P4'])
    key=['dataset','group_id','run_id','segment','fold','seed','horizon','issuance_s','target_time_s']
    paired=p2[key+['current_w','actual_w','q0.5']].merge(p4[key+['q0.5']],on=key,validate='one_to_one',suffixes=('_P2','_P4'))
    split=json.loads((mainout/'split_manifest.json').read_text())['folds'][f]
    prepared=pd.read_parquet(mainout/'prepared.parquet',columns=['run_id','current_w','eligible_history'])
    training=prepared.loc[prepared.run_id.isin(split['train_runs']) & prepared.eligible_history,'current_w'].dropna()
    p95=float(training.quantile(.95))
    rows=[];traces=[]
    for capacity_factor in (.1,.3):
        capacity=capacity_factor*p95
        for delay_bins in (0,2):
            for (run,segment),g in paired.groupby(['run_id','segment']):
                g=g.sort_values('issuance_s');d=g.current_w.to_numpy()
                if len(d)<2:continue
                for mode in ('C0','C2','C3','C4'):
                    forecast={'C3':g['q0.5_P2'].to_numpy(),'C4':g['q0.5_P4'].to_numpy()}.get(mode)
                    z=rollout(d,forecast,mode,capacity_w=capacity,ess_wh=capacity*2,delay_bins=delay_bins)
                    metrics=summarize(z,mode,run)
                    rows.append({**metrics,'segment':segment,'fold':f,'capacity_factor_training_p95':capacity_factor,
                        'capacity_w':capacity,'delay_s':delay_bins*.2,'n':len(z),
                        'soc_reserve_violation_count':int(np.sum(z.soc.to_numpy()<.2-1e-9))})
                    if capacity_factor==.3 and delay_bins==0:
                        z['run_id']=run;z['segment']=segment;z['controller']=mode;z['issuance_s']=g.issuance_s.to_numpy()
                        traces.append(z)
    result=pd.DataFrame(rows);result.to_csv(out/'control_sensitivity_per_segment.csv',index=False)
    run=result.groupby(['capacity_factor_training_p95','delay_s','controller','run_id'],as_index=False)[['pcc_ramp_p95_w_per_s','pcc_ramp_max_w_per_s','pcc_peak_to_peak_w','fast_abs_energy_wh','ess_abs_energy_wh','soc_min','soc_max','soc_reserve_violation_count']].mean()
    run.to_csv(out/'control_sensitivity_per_run.csv',index=False)
    run.groupby(['capacity_factor_training_p95','delay_s','controller'],as_index=False)[['pcc_ramp_p95_w_per_s','pcc_ramp_max_w_per_s','pcc_peak_to_peak_w','fast_abs_energy_wh','ess_abs_energy_wh','soc_min','soc_max','soc_reserve_violation_count']].mean().to_csv(out/'control_sensitivity_aggregate.csv',index=False)
    if traces:pd.concat(traces,ignore_index=True).to_parquet(out/'control_traces_base_case.parquet',index=False)
    (out/'assumptions.json').write_text(json.dumps({'evidence':'E1 assumed actuator PCC proxy, GPU control inactive',
        'scope':'fold 0, 1s paired P2/P4, seed 2026, 7 run groups; independent sessions unverified',
        'capacity':'0.1 and 0.3 times outer-train P95 GPU W','delay':'0 and 0.4s, integer 200ms bins',
        'ess_energy':'2h at assumed capacity','round_trip_efficiency':'0.95 each direction in existing assumed controller',
        'reference':'causal ramp-limited reference and storage split from aidc_exp.control.rollout',
        'limits':'No measured actuator, GPU cap performance, facility P/Q/V/f or grid dynamics. Metrics are proxy only.'},indent=2))
    print('segments',result[['run_id','segment']].drop_duplicates().shape[0],'runs',run.run_id.nunique(),'rows',len(result))

if __name__=='__main__':main()
