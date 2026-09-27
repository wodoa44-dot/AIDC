#!/usr/bin/env python
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,json,platform,sys,time,traceback,subprocess
from pathlib import Path
import numpy as np,pandas as pd,yaml
from src.data import audit_and_select,build
from src.experiment import run_cells
from src.reporting import report


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(Path(__file__).parent/'configs/default.yaml'))
    p.add_argument('--smoke',action='store_true');p.add_argument('--experiment-id');a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    if a.smoke:cfg.update(experiment_id='smoke_20260914',runs_per_cohort=1,resolutions_ms=[200],horizons_ms=[200],n_estimators=8,max_train_samples=600,max_validation_samples=200,bootstrap_replicates=50)
    if a.experiment_id:cfg['experiment_id']=a.experiment_id
    root=Path(__file__).resolve().parent;out=root/'outputs'/cfg['experiment_id']
    out.mkdir(parents=True,exist_ok=False)
    start=time.time()
    def log(s):
        msg=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())+' '+s
        print(msg,flush=True)
        with (out/'execution.log').open('a') as f:f.write(msg+'\n')
    status=dict(status='running',pid=os.getpid(),command=' '.join(sys.argv),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (out/'status.json').write_text(json.dumps(status,indent=2));(out/'effective_config.json').write_text(json.dumps(cfg,indent=2))
    import sklearn,lightgbm,pyarrow,matplotlib,joblib
    env=dict(python=sys.version,platform=platform.platform(),cpu_count=os.cpu_count(),seed=cfg['seed'],
      packages={k:v.__version__ for k,v in dict(numpy=np,pandas=pd,sklearn=sklearn,lightgbm=lightgbm,pyarrow=pyarrow,matplotlib=matplotlib,joblib=joblib,yaml=yaml).items()})
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    try:
        log('Audit actual continuous inputs and source equivalence')
        series=audit_and_select(cfg,out)
        grid=[]
        for r in cfg['resolutions_ms']:
            for h in cfg['horizons_ms']:
                grid.append(dict(resolution_ms=r,horizon_ms=h,selected=h>=r and h%r==0,
                    reason='compatible exact physical horizon' if h>=r and h%r==0 else 'horizon shorter than bin or not divisible; never rounded'))
        pd.DataFrame(grid).to_csv(out/'selected_grid.csv',index=False)
        log('Build complete causal bins and chronological samples')
        cells=build(series,cfg,out)
        run_cells(cells,cfg,out,log)
        log('Generate plots and Korean report');report(out,cfg)
        status.update(status='completed',elapsed_s=time.time()-start,cells=len(cells));log(f'COMPLETED in {time.time()-start:.1f} s')
    except Exception as e:
        status.update(status='failed',error=str(e),elapsed_s=time.time()-start);log(traceback.format_exc());raise
    finally:(out/'status.json').write_text(json.dumps(status,indent=2))
if __name__=='__main__':main()
