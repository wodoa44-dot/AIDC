"""Five-bin LP receding horizon diagnostic; not a certified optimal upper bound."""
import time
import numpy as np
from scipy.optimize import linprog


def step(future,pcc,energy,previous,credit,cfg,device,remaining):
    started=time.perf_counter();dt=cfg['dt'];eta=cfg['eta'];H=min(len(future),remaining)
    if H<=0:return np.zeros(2),{'solver_status':'inactive','solver_s':0.,'solver_fallback':False}
    # x = discharge[H,2], charge[H,2], abs(PCC difference)[H].
    nv=5*H;cost=np.zeros(nv);cost[:4*H]=cfg['mpc']['throughput_weight'];cost[4*H:]=cfg['mpc']['ramp_weight']
    pmax=np.array([device['fast_max_w'],device['ess_max_w']]);cap=np.array([device['fast_capacity_wh'],device['ess_capacity_wh']]);initial=cap*cfg['initial_soc']
    limits=[];rhs=[]
    def add(a,b):limits.append(a);rhs.append(b)
    def net(t,j):
        a=np.zeros(nv);a[t*2+j]=1;a[2*H+t*2+j]=-1;return a
    for t in range(H):
        demand_delta=future[t]-(pcc if t==0 else future[t-1])
        ramp=-sum((net(t,j) for j in range(2)),np.zeros(nv))
        if t>0:ramp+=sum((net(t-1,j) for j in range(2)),np.zeros(nv))
        a=ramp.copy();a[4*H+t]=-1;add(a,-demand_delta)
        a=-ramp;a[4*H+t]=-1;add(a,demand_delta)
        for j in range(2):
            combined=np.zeros(nv);combined[t*2+j]=1;combined[2*H+t*2+j]=1;add(combined,pmax[j])
            slew=net(t,j)-(net(t-1,j) if t else 0)
            prev=previous[j] if t==0 else 0
            add(slew,pmax[j]+prev);add(-slew,pmax[j]-prev)
            de=np.zeros(nv);used=np.zeros(nv)
            for k in range(t+1):
                de[2*k+j]=-dt/(3600*eta);de[2*H+2*k+j]=dt*eta/3600
                used[2*k+j]=used[2*H+2*k+j]=dt/3600
            restore_s=cfg['recovery_s']-2*dt
            low=max(cap[j]*cfg['soc_min'],initial[j]-pmax[j]*restore_s*eta/3600)
            high=min(cap[j]*cfg['soc_max'],initial[j]+pmax[j]*restore_s/(3600*eta))
            add(de,high-energy[j]);add(-de,energy[j]-low)
            # Every prefix reserves the AC energy needed for the common recovery.
            add(used-de/eta,credit[j]-(initial[j]-energy[j])/eta)
            add(used+de*eta,credit[j]-(energy[j]-initial[j])*eta)
    try:
        result=linprog(cost,A_ub=np.asarray(limits),b_ub=np.asarray(rhs),bounds=[(0,None)]*nv,method='highs',options={'time_limit':cfg['mpc']['time_limit_s']})
        if not result.success:raise RuntimeError('solver_'+str(result.status))
        u=result.x[:2*H].reshape(H,2);c=result.x[2*H:4*H].reshape(H,2)
        # Reject a relaxed solution that dissipates energy through simultaneous charge/discharge.
        if np.max(np.minimum(u,c))>cfg['mpc']['simultaneous_tolerance_w']:raise RuntimeError('simultaneous_flow_relaxation')
        command=u[0]-c[0]
        if not np.isfinite(command).all():raise RuntimeError('nonfinite')
        return command,{'solver_status':'optimal_lp_diagnostic','solver_s':time.perf_counter()-started,'solver_fallback':False}
    except Exception as e:
        # Zero request is causal and always slew-feasible in the shared physical core.
        return np.zeros(2),{'solver_status':str(e),'solver_s':time.perf_counter()-started,'solver_fallback':True}
