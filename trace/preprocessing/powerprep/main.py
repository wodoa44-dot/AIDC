from __future__ import annotations
import argparse
from pathlib import Path
from .config import load_config
from .pipeline import inventory,process,finalize_existing

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("stage",choices=["inventory","preprocess","detect","report","resolutions","windows","all"]); p.add_argument("--config",default=str(Path(__file__).parents[1]/"configs/default.yaml")); p.add_argument("--output",default=str(Path(__file__).parents[1]/"outputs")); p.add_argument("--datasets",nargs="+"); p.add_argument("--limit",type=int); p.add_argument("--no-resume",action="store_true")
 a=p.parse_args(argv); cfg=load_config(a.config); selected=a.datasets
 if str(cfg.get("schema_version", "")).startswith("2."):
  from .pipeline_v2 import run
  return run(a.stage,cfg,a.output,selected,a.limit,not a.no_resume)
 if a.stage in ["resolutions","windows"]: p.error("This stage requires configs/forecast_v2.yaml")
 if a.stage=="inventory": inventory(cfg,a.output,selected,a.limit)
 elif a.stage=="report": finalize_existing(cfg,a.output)
 else: process(cfg,a.output,selected,a.limit,not a.no_resume,a.stage)
if __name__=="__main__": main()
