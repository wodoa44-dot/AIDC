"""Launch with nohup; inspect identity before signaling only this pipeline."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import yaml
from pipeline import ROOT,atomic_json,config_hash


def proc(pid):
    p=Path('/proc')/str(pid)
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':int(pid),'cmdline':cmd,'start_ticks':stat[19],'state':stat[0]}
    except (FileNotFoundError,PermissionError,ProcessLookupError): return None


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['launch','status','stop']); ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--resume',action='store_true'); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    out=args.output_dir.resolve(); cfg=args.config.resolve(); out.mkdir(parents=True,exist_ok=True)
    if args.action in ('status','stop'):
        rec=json.loads((out/'process_identity.json').read_text()); current=proc(rec['pid'])
        valid=current and current['state']!='Z' and current['start_ticks']==rec['start_ticks'] and str(out) in current['cmdline'] and str(ROOT/'pipeline.py') in current['cmdline']
        print(json.dumps({'alive_verified':bool(valid),'process':current,'recorded':rec},indent=2))
        if args.action=='stop':
            if not valid: raise SystemExit('Refusing to signal absent/reused/unrelated PID')
            fd=os.pidfd_open(rec['pid'])
            fresh=proc(rec['pid'])
            if not fresh or fresh['start_ticks']!=rec['start_ticks']: raise SystemExit('PID changed')
            signal.pidfd_send_signal(fd,signal.SIGTERM); os.close(fd)
        return
    lockdir=ROOT/'outputs'/'.locks'; lockdir.mkdir(parents=True,exist_ok=True)
    with open(lockdir/'launch.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        target=config_hash({**yaml.safe_load(cfg.read_text()),'smoke':args.smoke})
        for p in Path('/proc').iterdir():
            if not p.name.isdigit(): continue
            info=proc(p.name)
            if not info or info['state']=='Z': continue
            if str(ROOT/'pipeline.py') not in info['cmdline'] and str(ROOT/'run_pipeline.sh') not in info['cmdline']: continue
            argv=(p/'cmdline').read_bytes().decode().split('\0')
            if '--config' in argv:
                other=Path(argv[argv.index('--config')+1])
                otherhash=config_hash({**yaml.safe_load(other.read_text()),'smoke':'--smoke' in argv})
                if target==otherhash:
                    print('Same config already running; not launching:',json.dumps(info)); return
        if (out/'provenance.json').exists() and not args.resume: raise SystemExit('Existing output requires --resume')
        # Preserve earlier logs and final statuses before resuming.
        stamp=str(time.time_ns())
        for name in ('pipeline.log','completion_status.json','exit_code.txt','progress.json','process_identity.json'):
            p=out/name
            if p.exists(): p.rename(out/f'{name}.{stamp}.previous')
        argv=['nohup','bash',str(ROOT/'run_pipeline.sh'),'--config',str(cfg),'--output-dir',str(out)]
        if args.resume: argv.append('--resume')
        if args.smoke: argv.append('--smoke')
        with open(out/'pipeline.log','ab',buffering=0) as log,open('/dev/null','rb') as devnull:
            child=subprocess.Popen(argv,stdin=devnull,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        (out/'pipeline.pid').write_text(str(child.pid)+'\n')
        identity=proc(child.pid)
        atomic_json(out/'process_identity.json',{**identity,'config_hash':target,'launch_argv':argv})
        print(json.dumps({'pid':child.pid,'output_dir':str(out),'command':argv},indent=2))
        time.sleep(2)
        print('PROCESS',json.dumps(proc(child.pid))); print('INITIAL LOG\n'+(out/'pipeline.log').read_text()[-3000:])
        if child.poll() not in (None,0): raise SystemExit(child.returncode)

if __name__=='__main__': main()
