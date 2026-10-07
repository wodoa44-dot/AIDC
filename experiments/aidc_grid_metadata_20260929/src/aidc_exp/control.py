from __future__ import annotations
import numpy as np
import pandas as pd


def rollout(demand_w, forecast_w=None, mode="C2", dt=.2, capacity_w=300., delay_bins=0,
            ess_wh=1000., reserve=.2, eta=.95, slew_w_per_s=500.):
    """Assumed E1 power balance. Positive storage power discharges to offset load."""
    d=np.asarray(demand_w,float); f=np.asarray(forecast_w,float) if forecast_w is not None else d
    n=len(d); pcc=np.empty(n); fast=np.zeros(n); ess=np.zeros(n); soc=np.empty(n)
    reference=d[0]; slow=0.; energy=ess_wh*.5; queue=[0.]*(delay_bins+1)
    for i in range(n):
        if mode in ("C0","C1"):
            reference=d[i]; desired=0.
        else:
            # Forecast influences the desired reference. The causal feedback term is current demand.
            aim=d[i] if mode=="C2" else d[i]+.5*(f[i]-d[i])
            step=np.clip(aim-reference,-slew_w_per_s*dt,slew_w_per_s*dt)
            reference+=step
            desired=np.clip(d[i]-reference,-capacity_w,capacity_w)
        queue.append(desired); delayed=queue.pop(0)
        slow+=.15*(delayed-slow)
        fast_cmd=np.clip(delayed-slow,-capacity_w*.5,capacity_w*.5)
        ess_cmd=np.clip(delayed-fast_cmd,-capacity_w*.5,capacity_w*.5)
        # Wh energy account: positive power is discharge, negative power is charge.
        if ess_cmd >= 0:
            ess_cmd=min(ess_cmd,max(0.,(energy-reserve*ess_wh)*eta*3600/dt))
            energy-=ess_cmd*dt/(3600*eta)
        else:
            ess_cmd=max(ess_cmd,-max(0.,(ess_wh-energy))*3600/(dt*eta))
            energy-=ess_cmd*dt*eta/3600
        fast[i]=fast_cmd;ess[i]=ess_cmd;soc[i]=energy/ess_wh
        pcc[i]=d[i]-fast[i]-ess[i]
    return pd.DataFrame({"baseline_demand_w":d,"gpu_after_control_w":d,"gpu_command_w":np.zeros(n),
        "fast_storage_w":fast,"ess_w":ess,"pcc_w":pcc,"soc":soc,
        "power_balance_error_w":pcc-(d-fast-ess)})


def summarize(frame, mode, run_id):
    p=frame.pcc_w.to_numpy();r=np.diff(p)/.2
    return {"run_id":run_id,"controller":mode,"evidence":"E1 assumed actuator, no measured GPU cap response",
            "pcc_ramp_p95_w_per_s":float(np.percentile(abs(r),95)),
            "pcc_ramp_max_w_per_s":float(abs(r).max()),"pcc_peak_to_peak_w":float(p.max()-p.min()),
            "fast_abs_energy_wh":float(np.sum(abs(frame.fast_storage_w))*.2/3600),
            "ess_abs_energy_wh":float(np.sum(abs(frame.ess_w))*.2/3600),
            "soc_min":float(frame.soc.min()),"soc_max":float(frame.soc.max()),
            "power_balance_max_error_w":float(abs(frame.power_balance_error_w).max())}
