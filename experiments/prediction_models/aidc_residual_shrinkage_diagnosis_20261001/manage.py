"""Nohup launcher with duplicate prevention and process identity."""
import argparse, fcntl, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def proc(pid):
    try:
        d=Path('/proc')/str(pid);argv=(d/'cmdline').read_bytes().decode().split('\0');s=(d/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),argv=argv,start_ticks=s[19],state=s[0])
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['launch','status']);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    for flag in ['resume','report-only']:ap.add_argument('--'+flag,action='store_true')
    a=ap.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.action=='status':
        r=json.loads((out/'process_identity.json').read_text());p=proc(r['pid']);valid=p and p['state']!='Z' and p['start_ticks']==r['start_ticks'] and str(ROOT/'pipeline.py') in p['argv']
        print(json.dumps(dict(alive_verified=bool(valid),recorded=r,current=p,completion=json.loads((out/'completion_status.json').read_text()) if (out/'completion_status.json').exists() else None),indent=2));return
    with open(ROOT/'launch.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for p in Path('/proc').iterdir():
            if p.name.isdigit():
                q=proc(p.name)
                if q and q['state']!='Z' and str(ROOT/'pipeline.py') in q['argv']:raise SystemExit('Analysis already running: '+p.name)
        if (out/'identity.json').exists() and not (a.resume or a.report_only):raise SystemExit('Use --resume for existing output')
        stamp=str(time.time_ns())
        for n in ['pipeline.log','process_identity.json','completion_status.json','progress.json','exit_code.txt']:
            p=out/n
            if p.exists():p.rename(out/(n+'.'+stamp+'.previous'))
        cmd=['nohup',sys.executable,'-u',str(ROOT/'pipeline.py'),'--config',str(a.config.resolve()),'--output-dir',str(out)]
        for flag in ['resume','report-only']:
            if getattr(a,flag.replace('-','_')):cmd.append('--'+flag)
        with open('/dev/null','rb') as stdin, open(out/'pipeline.log','ab',buffering=0) as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdin=stdin,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONUNBUFFERED':'1'},close_fds=True)
        rec=proc(child.pid);rec.update(command=cmd,started_utc=datetime.now(timezone.utc).isoformat(),started_kst=datetime.now(timezone.utc).astimezone(timezone(__import__('datetime').timedelta(hours=9))).isoformat())
        (out/'process_identity.json').write_text(json.dumps(rec,indent=2));(out/'pipeline.pid').write_text(str(child.pid)+'\n')
        time.sleep(1);print(json.dumps(dict(pid=child.pid,current=proc(child.pid)),indent=2));print((out/'pipeline.log').read_text()[-2500:])
        if child.poll() not in (None,0):raise SystemExit(child.returncode)
if __name__=='__main__':main()
