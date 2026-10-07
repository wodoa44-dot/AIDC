from __future__ import annotations
import json, time
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import yaml
from ..inventory import PROJECT

FIELDS=['trial_id','gpu_uuid','monotonic_start_ns','monotonic_end_ns','wall_utc','poll_interval_s','api_duration_ns','power_w','energy_j','temperature_c','error','evidence_type']

def prepare(out:Path):
    out.mkdir(parents=True,exist_ok=True)
    combinations=[]
    for workload in ['training_compute','training_communication','training_checkpoint','inference_steady','inference_burst']:
        for treatment in ['baseline','cap_down_up','filler_on_off','gradual_release','staggered_release']:
            combinations.append({'workload':workload,'treatment':treatment,'independent_trials_planned':3,
                'actual_trials':0,'status':'blocked_hardware','requires':'authorized GPU UUID, instrumented workload, no competing process'})
    pd.DataFrame(combinations).to_csv(out/'phase0_matrix.csv',index=False)
    (out/'phase0_protocol.md').write_text('''# Phase 0 measurement protocol\n\nLive collection is disabled by default. Record GPU UUID, authorization, occupancy, workload command, allowed cap range, thermal stop, max duration, and paired no-treatment trials. Sample with 10/20/50/100 ms polling; polling frequency is not independent sensor bandwidth. Log monotonic API start/end and UTC wall time. Log commands and workload phase/performance separately with their own availability times. Use three independently restarted trials per feasible condition, randomized order, warmup/cooldown, and restore cap/clock on exit. Missing PDU/PCC meters remain disabled with a reason. No UPS/protection commands.\n''')
    (out/'d5_collection_schema.json').write_text(json.dumps({'evidence_type':'measured_new','required_fields':['timestamp_utc','available_at_utc','meter_id','measurement_boundary','pcc_active_power_w','pcc_reactive_power_var','voltage_v','frequency_hz','ess_soc','ups_mode','clock_offset_s','clock_uncertainty_s'],'status':'blocked_source','disabled_reason':'no facility PCC/ESS/UPS meter source or authorization supplied'},indent=2))

def sample_nvml(nvml,device,trial_id,poll_s,max_duration_s,max_temperature_c=None):
    rows=[];t0=time.monotonic();last=None
    while time.monotonic()-t0<max_duration_s:
        start=time.monotonic_ns();wall=datetime.now(timezone.utc).isoformat();error=[]
        try:power=nvml.nvmlDeviceGetPowerUsage(device)/1000.0
        except Exception as e:power=None;error.append('power:'+type(e).__name__)
        try:energy=nvml.nvmlDeviceGetTotalEnergyConsumption(device)/1000.0
        except Exception as e:energy=None;error.append('energy_unsupported_or_error:'+type(e).__name__)
        try:temp=nvml.nvmlDeviceGetTemperature(device,nvml.NVML_TEMPERATURE_GPU)
        except Exception as e:temp=None;error.append('temperature:'+type(e).__name__)
        end=time.monotonic_ns()
        rows.append({'trial_id':trial_id,'gpu_uuid':str(nvml.nvmlDeviceGetUUID(device)),
            'monotonic_start_ns':start,'monotonic_end_ns':end,'wall_utc':wall,
            'poll_interval_s':(start-last)/1e9 if last else None,'api_duration_ns':end-start,
            'power_w':power,'energy_j':energy,'temperature_c':temp,'error':';'.join(error), 'evidence_type':'measured_new'})
        last=start
        if max_temperature_c is not None and temp is not None and temp>=max_temperature_c:
            break
        time.sleep(max(0,poll_s-(time.monotonic_ns()-start)/1e9))
    return pd.DataFrame(rows,columns=FIELDS)

def collect(out:Path,live:bool,config_path:str|None):
    prepare(out)
    if not live:raise RuntimeError('Use collect --live --config with a fully specified authorized device and workload')
    cfg=yaml.safe_load(Path(config_path).read_text()) if config_path else yaml.safe_load((PROJECT/'configs/phase0.yaml').read_text())
    required=['gpu_uuid','experiment_authorized','workload_command','max_duration_s','max_temperature_c']
    missing=[k for k in required if not cfg.get(k)]
    if missing or not cfg.get('live_enabled'):
        raise RuntimeError(f'Live collection disabled or missing: {missing}; supply an authorized device/workload config')
    import pynvml as nvml
    nvml.nvmlInit()
    try:
        device=nvml.nvmlDeviceGetHandleByUUID(cfg['gpu_uuid'].encode())
        # Passive collection only. Active cap controls are separate explicit trials.
        trial=datetime.now(timezone.utc).strftime('passive_%Y%m%dT%H%M%SZ')
        if cfg.get('mode','passive')=='cap_trial':
            from .controls import run_cap_trial
            frame,commands,workload=run_cap_trial(nvml,device,cfg,trial)
            commands.to_parquet(out/f'{trial}_commands.parquet',index=False)
            pd.DataFrame([workload]).to_parquet(out/f'{trial}_workload_events.parquet',index=False)
        else:
            frame=sample_nvml(nvml,device,trial,cfg['poll_interval_ms']/1000,float(cfg['max_duration_s']),cfg['max_temperature_c'])
        frame.to_parquet(out/f'{trial}_nvml.parquet',index=False)
    finally:nvml.nvmlShutdown()
