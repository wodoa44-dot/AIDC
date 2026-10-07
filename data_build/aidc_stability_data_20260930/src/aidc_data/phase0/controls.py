from __future__ import annotations
from contextlib import contextmanager

@contextmanager
def temporary_power_limit(nvml,device,requested_mw,minimum_mw,maximum_mw):
    """Restore the pretrial cap on normal or exceptional exit. Caller must own device."""
    previous=nvml.nvmlDeviceGetPowerManagementLimit(device)
    if not minimum_mw<=requested_mw<=maximum_mw:raise ValueError('cap outside configured safe range')
    try:
        nvml.nvmlDeviceSetPowerManagementLimit(device,int(requested_mw))
        enforced=nvml.nvmlDeviceGetPowerManagementLimit(device)
        yield {'requested_mw':requested_mw,'enforced_mw':enforced,'previous_mw':previous}
    finally:nvml.nvmlDeviceSetPowerManagementLimit(device,previous)

def run_cap_trial(nvml,device,cfg,trial_id):
    """One guarded live trial. Service metrics need an instrumented workload wrapper."""
    import subprocess,time
    from datetime import datetime,timezone
    import pandas as pd
    from .collect import sample_nvml
    if not cfg.get('experiment_authorized') or not cfg.get('gpu_free_verified'):
        raise RuntimeError('authorized idle GPU verification required')
    if not cfg.get('allowed_cap_range_mw') or not cfg.get('requested_cap_mw'):
        raise RuntimeError('explicit cap range and requested cap required')
    low,high=cfg['allowed_cap_range_mw'];requested=int(cfg['requested_cap_mw'])
    device_low,device_high=nvml.nvmlDeviceGetPowerManagementLimitConstraints(device)
    low=max(int(low),int(device_low));high=min(int(high),int(device_high))
    if not low<=requested<=high:raise ValueError('cap outside device and user allowed range')
    original=nvml.nvmlDeviceGetPowerManagementLimit(device)
    start=time.monotonic_ns();utc=datetime.now(timezone.utc).isoformat();commands=[]
    if not isinstance(cfg['workload_command'],list) or not cfg['workload_command']:
        raise ValueError('workload_command must be an argv list')
    proc=subprocess.Popen(cfg['workload_command'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        def command(kind,value):
            intended=time.monotonic_ns();nvml.nvmlDeviceSetPowerManagementLimit(device,value);returned=time.monotonic_ns()
            enforced=nvml.nvmlDeviceGetPowerManagementLimit(device)
            commands.append({'command_id':f'{trial_id}:{kind}','trial_id':trial_id,'gpu_uuid':cfg['gpu_uuid'],
                'kind':kind,'intended_monotonic_ns':intended,'apply_time_monotonic_ns':None,
                'return_monotonic_ns':returned,'api_latency_ns':returned-intended,'requested_mw':value,
                'enforced_mw':enforced,'success':enforced==value,'evidence_type':'measured_new'})
        # Warmup then cap; sampling spans both windows. Restoration always runs.
        warmup=min(float(cfg['max_duration_s'])/3,5.)
        before=sample_nvml(nvml,device,trial_id,float(cfg['poll_interval_ms'])/1000,warmup,cfg['max_temperature_c'])
        if before.temperature_c.max()>=cfg['max_temperature_c']:raise RuntimeError('thermal stop during warmup')
        command('cap_down',requested)
        during=sample_nvml(nvml,device,trial_id,float(cfg['poll_interval_ms'])/1000,float(cfg['max_duration_s'])-warmup,cfg['max_temperature_c'])
        if during.temperature_c.max()>=cfg['max_temperature_c']:raise RuntimeError('thermal stop during cap interval')
        samples=pd.concat([before,during],ignore_index=True)
        workload={'trial_id':trial_id,'workload_start_utc':utc,'workload_start_monotonic_ns':start,'workload_pid':proc.pid,'workload_exit_code':proc.poll(),'evidence_type':'measured_new'}
    finally:
        try:command('restore',original)
        finally:
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
    return samples,pd.DataFrame(commands),workload
