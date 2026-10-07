"""AC-boundary E1 simulation. No hardware interface. Fixed online budget windows."""
import numpy as np
import pandas as pd


def simulate(demand, forecast, mode, cfg, device, rates, multiplier=1., budgeted=True, trajectory=None):
    dt=cfg['dt']; n=len(demand); active=int((cfg['window_s']-cfg['recovery_s'])/dt)
    pmax=np.array([device['fast_max_w'],device['ess_max_w']])
    cap=np.array([device['fast_capacity_wh'],device['ess_capacity_wh']])
    initial=cap*cfg['initial_soc']; energy=initial.copy(); eta=cfg['eta']
    lower=cap*cfg['soc_min']; upper=cap*cfg['soc_max']
    # Fixed-window budget, fully credited at start. No carryover; no future run length.
    budget=np.asarray(rates)*cfg['window_s']/3600*multiplier
    credit=budget.copy() if budgeted else np.full(2,np.inf)
    prev=np.zeros(2); reference=float(demand[0]); slow=0.
    queue=[0.]*cfg['delay_bins']; rows=[]; pending=np.zeros(2)
    def change(p): return -p*dt/(3600*eta) if p>=0 else -p*dt*eta/3600
    def liability(e,j):
        return (initial[j]-e)/eta if e<initial[j] else (e-initial[j])*eta
    for i,d in enumerate(demand):
        recovery=i>=active
        aim=d if mode=='C2' else d+cfg['forecast_weight']*(forecast[i]-d)
        reference+=float(np.clip(aim-reference,-cfg['reference_slew_w_s']*dt,cfg['reference_slew_w_s']*dt))
        requested=float(np.clip(d-reference,-pmax.sum(),pmax.sum())) if mode not in ('C0','C1') else 0.
        queue.append(requested); delayed=queue.pop(0)
        slow+=cfg['slow_alpha']*(delayed-slow)
        commands=np.array([delayed-slow,slow])
        if trajectory is not None: commands=pending.copy()
        if recovery:
            # Common clock-triggered restoration, including its ramps and throughput.
            commands=np.array([(energy[j]-initial[j])*3600*(eta if energy[j]>=initial[j] else 1/eta)/dt for j in range(2)])
        applied=np.zeros(2); flags=[]; losses=[]; residual=[]
        for j in range(2):
            # Physical slew Pmax/dt; zero is always reachable. This is an explicit E1 assumption.
            lo=max(-pmax[j],prev[j]-pmax[j],-(upper[j]-energy[j])*3600/(dt*eta))
            hi=min(pmax[j],prev[j]+pmax[j],(energy[j]-lower[j])*3600*eta/dt)
            # Reserve enough time to restore SoC with two idle/slew margin bins.
            recover_s=cfg['recovery_s']-2*dt
            elo=max(lower[j],initial[j]-pmax[j]*recover_s*eta/3600)
            ehi=min(upper[j],initial[j]+pmax[j]*recover_s/(3600*eta))
            if not recovery:
                lo=max(lo,-(ehi-energy[j])*3600/(dt*eta))
                hi=min(hi,(energy[j]-elo)*3600*eta/dt)
            p=float(np.clip(commands[j],lo,hi))
            physical=p
            if budgeted and abs(p)*dt/3600+liability(energy[j]+change(p),j)>credit[j]+1e-9:
                a,b=0.,1.
                for _ in range(35):
                    m=(a+b)/2; q=p*m
                    if abs(q)*dt/3600+liability(energy[j]+change(q),j)<=credit[j]+1e-9: a=m
                    else: b=m
                p*=a
            old=energy[j]; energy[j]+=change(p); applied[j]=p
            used=abs(p)*dt/3600
            if budgeted: credit[j]=max(0.,credit[j]-used)
            loss=used*(1/eta-1) if p>=0 else used*(1-eta)
            losses.append(loss); residual.append(energy[j]-old+p*dt/3600+loss)
            flags.extend([abs(commands[j])>pmax[j]+1e-8,abs(physical-p)>1e-8,
                          energy[j]<=lower[j]+1e-9 or energy[j]>=upper[j]-1e-9])
        row=dict(step=i,recovery=recovery,reference_w=reference,reference_used=trajectory is None,baseline_demand_w=d,forecast_w=forecast[i],pcc_w=d-applied.sum(),
                 power_balance_residual_w=(d-applied.sum())+applied.sum()-d)
        for j,name in enumerate(('fast','ess')):
            row.update({f'{name}_w':applied[j],f'{name}_requested_w':commands[j],f'{name}_soc':energy[j]/cap[j],
                        f'{name}_credit_wh':credit[j],f'{name}_budget_wh':budget[j],f'{name}_loss_wh':losses[j],
                        f'{name}_energy_balance_residual_wh':residual[j],f'{name}_output_active':flags[3*j],
                        f'{name}_budget_active':flags[3*j+1],f'{name}_soc_active':flags[3*j+2],
                        f'{name}_slew_w_s':abs(applied[j]-prev[j])/dt})
        row.update(solver_status='not_called',solver_s=0.,solver_fallback=False)
        rows.append(row); prev=applied
        if trajectory is not None and not recovery:
            from mpc import step
            if cfg.get('_smoke_mpc_steps') is not None and i>=cfg['_smoke_mpc_steps']:
                pending=np.zeros(2);info=dict(solver_status='smoke_solve_cap',solver_s=0.,solver_fallback=True)
            else:
                pending,info=step(trajectory[i],row['pcc_w'],energy.copy(),applied.copy(),credit.copy(),cfg,device,active-i-1)
            row.update(info)
    return pd.DataFrame(rows)


def metrics(z,cfg,device):
    dt=cfg['dt']; r=np.diff(z.pcc_w)/dt; a=abs(r); exceed=a>cfg['ramp_threshold_w_s']
    streak=best=0
    for b in exceed:
        streak=streak+1 if b else 0; best=max(best,streak)
    m=dict(n=len(z),duration_s=len(z)*dt,p95_ramp=np.quantile(a,.95),p99_ramp=np.quantile(a,.99),
           max_ramp=a.max(),peak_to_peak=np.ptp(z.pcc_w),threshold_exceed_fraction=exceed.mean(),
           threshold_exceed_s=exceed.sum()*dt,threshold_max_duration_s=best*dt,
           rise_p95=np.quantile(r[r>0],.95) if (r>0).any() else 0.,
           fall_p95=np.quantile(-r[r<0],.95) if (r<0).any() else 0.,
           rise_max=max(0.,r.max()),fall_max=max(0.,-r.min()),
           power_balance_residual_w=abs(z.power_balance_residual_w).max())
    for name in ('fast','ess'):
        p=z[f'{name}_w']; soc=z[f'{name}_soc']; cap=device[f'{name}_capacity_wh']; limit=device[f'{name}_max_w']
        used=abs(p).sum()*dt/3600; budget=z[f'{name}_budget_wh'].iloc[0]
        m.update({f'{name}_throughput_wh':used,f'{name}_loss_wh':z[f'{name}_loss_wh'].sum(),
                  f'{name}_soc_min':soc.min(),f'{name}_soc_max':soc.max(),f'{name}_soc_end':soc.iloc[-1],
                  f'{name}_end_energy_delta_wh':(soc.iloc[-1]-cfg['initial_soc'])*cap,
                  f'{name}_budget_wh':budget,f'{name}_unused_wh':budget-used,
                  f'{name}_budget_active_fraction':z[f'{name}_budget_active'].mean(),
                  f'{name}_output_active_fraction':z[f'{name}_output_active'].mean(),
                  f'{name}_soc_active_fraction':z[f'{name}_soc_active'].mean(),
                  f'{name}_request_applied_mae_w':abs(z[f'{name}_requested_w']-p).mean(),
                  f'{name}_max_slew_w_s':z[f'{name}_slew_w_s'].max(),
                  f'{name}_energy_balance_residual_wh':abs(z[f'{name}_energy_balance_residual_wh']).max(),
                  f'{name}_power_violation':max(0.,abs(p).max()-limit),
                  f'{name}_soc_violation':max(0.,cfg['soc_min']-soc.min(),soc.max()-cfg['soc_max']),
                  f'{name}_budget_violation_wh':max(0.,used-budget),
                  f'{name}_slew_violation':max(0.,z[f'{name}_slew_w_s'].max()-limit/dt)})
    return m


def validate(m):
    for k,v in m.items():
        if any(s in k for s in ('violation','residual','end_energy_delta')):
            assert abs(v)<1e-6,(k,v)
