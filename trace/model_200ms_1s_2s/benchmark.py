#!/usr/bin/env python
"""Warm single-thread benchmark of the actual six-model metadata package."""
import argparse, json, os, platform, resource, sys, time
from pathlib import Path
import joblib, numpy as np
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from src.core import power_features, raw_predict, rearrange, transform_one
from src.online import ForecastService

def pct(v): return {f"p{p}":float(np.percentile(v,p)) for p in (50,95,99)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--package",required=True); ap.add_argument("--output",required=True); ap.add_argument("--trials",type=int,default=1000); a=ap.parse_args()
    start=time.perf_counter(); pkg=joblib.load(a.package); load_ms=(time.perf_counter()-start)*1000
    md={"gpu":"__UNSEEN_GPU__","workload":"llm_training","batch":16,"ai_model":"__UNSEEN_MODEL__"}; values=np.linspace(1800,2200,25)
    feature=[]; predict=[]; end=[]
    for _ in range(20):
        for item in pkg["horizons"].values():
            pf=power_features(values,200,item["history_ms"]); x=transform_one(pf,md,item["encoder"]); rearrange(raw_predict(item["models"],x))
    for _ in range(a.trials):
        t=time.perf_counter_ns(); xs=[]
        for item in pkg["horizons"].values():
            pf=power_features(values,200,item["history_ms"]); xs.append(transform_one(pf,md,item["encoder"]))
        t1=time.perf_counter_ns();
        for x,item in zip(xs,pkg["horizons"].values()): rearrange(raw_predict(item["models"],x))
        t2=time.perf_counter_ns(); feature.append((t1-t)/1000); predict.append((t2-t1)/1000); end.append((t2-t)/1000)
    result={"package":a.package,"package_bytes":os.path.getsize(a.package),"load_ms":load_ms,"trials":a.trials,
        "feature_construction_us":pct(feature),"prediction_only_six_models_us":pct(predict),"end_to_end_six_models_us":pct(end),
        "bin_collection_delay_ms":200,"model_threads":1,"process_workers":1,"maxrss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_method":"resource.getrusage(RUSAGE_SELF).ru_maxrss; process high-water mark, not incremental model RSS",
        "hardware":{"platform":platform.platform(),"processor":platform.processor(),"logical_cpu_count":os.cpu_count()},
        "note":"Local warmed benchmark; comparison with 200 ms cadence is not a real-time guarantee."}
    Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
