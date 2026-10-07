"""Nohup launcher, verified process identity, resume and prediction-only report mode."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from data import atomic_json
ROOT=Path(__file__).resolve().parent


def proc(pid):
    try:
        p=Path('/proc')/str(pid);argv=(p/'cmdline').read_bytes().decode().split('\0');stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':int(pid),'argv':argv,'start_ticks':stat[19],'state':stat[0]}
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['launch','status','stop']);ap.add_argument('--config',type=Path,default=ROOT/'config.yaml');ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--report-only',action='store_true');args=ap.parse_args();out=args.output_dir.resolve();cfg=args.config.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.action in ['status','stop']:
        rec=json.loads((out/'process_identity.json').read_text());current=proc(rec['pid']);valid=current and current['state']!='Z' and current['start_ticks']==rec['start_ticks'] and str(ROOT/'pipeline.py') in current['argv'] and str(out) in current['argv']
        print(json.dumps({'alive_verified':bool(valid),'recorded':rec,'current':current},indent=2))
        if args.action=='stop':
            assert valid,'Refusing unrelated, reused or absent PID';fd=os.pidfd_open(rec['pid']);fresh=proc(rec['pid']);assert fresh and fresh['start_ticks']==rec['start_ticks'];signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd)
        return
    lockdir=ROOT/'outputs/.locks';lockdir.mkdir(parents=True,exist_ok=True)
    with open(lockdir/'launch.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            current=proc(p.name)
            if current and current['state']!='Z' and str(ROOT/'pipeline.py') in current['argv']:
                argv=current['argv'];otherconfig=Path(argv[argv.index('--config')+1])
                if otherconfig.read_bytes()==cfg.read_bytes() and ('--smoke' in argv)==args.smoke:raise SystemExit('Same pilot configuration already running: '+str(current['pid']))
        if (out/'identity.json').exists() and not (args.resume or args.report_only):raise SystemExit('Existing run needs --resume')
        stamp=str(time.time_ns())
        for name in ['pipeline.log','progress.json','completion_status.json','exit_code.txt','process_identity.json']:
            p=out/name
            if p.exists():p.rename(out/(name+'.'+stamp+'.previous'))
        command=['nohup',sys.executable,'-u',str(ROOT/'pipeline.py'),'--config',str(cfg),'--output-dir',str(out)]
        for flag,active in [('--smoke',args.smoke),('--resume',args.resume),('--report-only',args.report_only)]:
            if active:command.append(flag)
        env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
        with open('/dev/null','rb') as stdin,open(out/'pipeline.log','ab',buffering=0) as log:
            child=subprocess.Popen(command,cwd=str(ROOT),stdin=stdin,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,close_fds=True)
        identity=proc(child.pid);atomic_json(out/'process_identity.json',{**identity,'command':command,'started_utc':datetime.now(timezone.utc).isoformat(),'working_directory':str(ROOT),'python':sys.executable});(out/'pipeline.pid').write_text(str(child.pid)+'\n')
        time.sleep(2);print(json.dumps({'pid':child.pid,'current':proc(child.pid),'output_dir':str(out)},indent=2));print('INITIAL LOG\n'+(out/'pipeline.log').read_text()[-4500:])
        if child.poll() not in (None,0):raise SystemExit(child.returncode)


if __name__=='__main__':main()
