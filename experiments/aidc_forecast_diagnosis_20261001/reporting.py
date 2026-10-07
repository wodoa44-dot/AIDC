import json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown
from nested_report import paired_cluster

METRICS=['p95_ramp','p99_ramp','max_ramp','fast_throughput_wh','ess_throughput_wh','fast_mean_abs_w','ess_mean_abs_w']


def region_stats(z,cfg,region):
    point=np.ones(len(z),dtype=bool) if region=='whole' else (z.recovery.to_numpy()==(region=='recovery'))
    dt=np.diff(z.issuance_s.to_numpy());r=np.diff(z.pcc_w.to_numpy())/dt;rm=point[1:]
    rr=r[rm];aa=abs(rr);idx=np.flatnonzero(rm)[np.argmax(aa)]+1
    exceed=aa>cfg['ramp_threshold_w_s'];longest=streak=0
    for x in exceed:streak=streak+1 if x else 0;longest=max(longest,streak)
    used=z.iloc[np.flatnonzero(point)];dur=point.sum()*cfg['dt']
    m={'n_samples':int(point.sum()),'duration_s':dur,'p95_ramp':np.quantile(aa,.95),'p99_ramp':np.quantile(aa,.99),'max_ramp':aa.max(),
       'threshold_exceed_fraction':exceed.mean(),'threshold_exceed_s':dt[rm][exceed].sum(),'threshold_max_duration_s':longest*cfg['dt'],
       'max_ramp_time_s':z.issuance_s.iloc[idx],'max_ramp_window':z.window.iloc[0],'max_ramp_segment':z.segment.iloc[0],
       'power_balance_residual_w':abs(used.power_balance_residual_w).max(),'solver_s_sum':used.solver_s.sum(),'solver_s_max':used.solver_s.max(),
       'solver_calls':int((~used.solver_status.isin(['not_called','inactive','smoke_solve_cap'])).sum()),'solver_fallbacks':int(used.solver_fallback.sum()),
       'deadline_exceed_count':int((used.solver_s>cfg['dt']).sum())}
    causes=[]
    if abs(z.baseline_demand_w.iloc[idx]-z.baseline_demand_w.iloc[idx-1])/dt[idx-1]>cfg['ramp_threshold_w_s']:causes.append('load_transition')
    if z.reference_used.iloc[idx] and abs(z.reference_w.iloc[idx]-z.reference_w.iloc[idx-1])>1e-8:causes.append('reference_change')
    for name in ['fast','ess']:
        p=used[f'{name}_w'];energy=abs(p).sum()*cfg['dt']/3600
        m.update({f'{name}_throughput_wh':energy,f'{name}_mean_abs_w':energy*3600/dur,
                  f'{name}_loss_wh':used[f'{name}_loss_wh'].sum(),f'{name}_soc_min':used[f'{name}_soc'].min(),f'{name}_soc_max':used[f'{name}_soc'].max(),
                  f'{name}_soc_at_region_end':used[f'{name}_soc'].iloc[-1],f'{name}_window_terminal_soc':z[f'{name}_soc'].iloc[-1],
                  f'{name}_window_terminal_w':z[f'{name}_w'].iloc[-1],f'{name}_output_active_fraction':used[f'{name}_output_active'].mean(),
                  f'{name}_budget_active_fraction':used[f'{name}_budget_active'].mean(),f'{name}_soc_active_fraction':used[f'{name}_soc_active'].mean(),
                  f'{name}_request_applied_mae_w':abs(used[f'{name}_requested_w']-p).mean(),
                  f'{name}_energy_balance_residual_wh':abs(used[f'{name}_energy_balance_residual_wh']).max(),
                  f'{name}_unused_credit_at_region_end_wh':used[f'{name}_credit_wh'].iloc[-1]})
        if z[f'{name}_output_active'].iloc[idx] or z[f'{name}_output_active'].iloc[idx]!=z[f'{name}_output_active'].iloc[idx-1]:causes.append(name+'_saturation_or_release')
        if z[f'{name}_budget_active'].iloc[idx]:causes.append(name+'_budget_limit')
        if z.recovery.iloc[idx] and abs(z[f'{name}_w'].iloc[idx])<1e-8 and abs(z[f'{name}_w'].iloc[idx-1])>1e-8:causes.append(name+'_output_return_to_zero')
    if z.recovery.iloc[idx]:causes.append('soc_recovery')
    m['max_ramp_coincident_candidates']=json.dumps(causes);return m,rr


def aggregate(cfg,out,log):
    design=pd.concat([pd.read_csv(out/'endpoint_design.csv'),pd.read_csv(out/'trajectory_design.csv')],ignore_index=True)
    design['weight_axis']=np.where(design.family.eq('C_weight'),design.weight,-1.)
    design.to_csv(out/'evaluation_design.csv',index=False)
    task_runs=[];windows=[];constraints=[]
    for task in design.task.unique():
        folder=out/'control_cache'/task;record=json.loads((folder/'complete.json').read_text());grouped={}
        for name in record['files']:
            z=pd.read_parquet(folder/name);first=z.iloc[0]
            cp=json.loads((folder/(name+'.json')).read_text())['constraint_metrics']
            constraints.append({'task':task,'run_id':first.run_id,'segment':first.segment,'window':first.window,**cp})
            for region in ['operation','recovery','whole']:
                m,r=region_stats(z,cfg,region);ids={'task':task,'run_id':first.run_id,'group_id':first.group_id,'region':region}
                windows.append({**ids,'segment':first.segment,'window':first.window,**m})
                entry=grouped.setdefault((first.run_id,region),{'group':first.group_id,'ramp':[],'m':[]})
                entry['ramp'].append(r);entry['m'].append(m)
        for (run,region),x in grouped.items():
            a=abs(np.concatenate(x['ramp']));m=pd.DataFrame(x['m']);worst=m.loc[m.max_ramp.idxmax()]
            row={'task':task,'run_id':run,'group_id':x['group'],'region':region,'p95_ramp':np.quantile(a,.95),'p99_ramp':np.quantile(a,.99),'max_ramp':a.max(),
                 'n_windows':len(m),'duration_s':m.duration_s.sum(),'threshold_exceed_fraction':float(np.mean(a>cfg['ramp_threshold_w_s'])),
                 'threshold_exceed_s':m.threshold_exceed_s.sum(),'threshold_max_duration_s':m.threshold_max_duration_s.max(),
                 'max_ramp_time_s':worst.max_ramp_time_s,'max_ramp_window':worst.max_ramp_window,'max_ramp_segment':worst.max_ramp_segment,
                 'max_ramp_coincident_candidates':worst.max_ramp_coincident_candidates,'solver_s_sum':m.solver_s_sum.sum(),'solver_s_max':m.solver_s_max.max(),
                 'solver_calls':m.solver_calls.sum(),'solver_fallbacks':m.solver_fallbacks.sum(),'deadline_exceed_count':m.deadline_exceed_count.sum()}
            for dev in ['fast','ess']:
                for suffix in ['throughput_wh','loss_wh']:row[f'{dev}_{suffix}']=m[f'{dev}_{suffix}'].sum()
                row[f'{dev}_mean_abs_w']=row[f'{dev}_throughput_wh']*3600/row['duration_s']
                for suffix in ['output_active_fraction','budget_active_fraction','soc_active_fraction','request_applied_mae_w']:row[f'{dev}_{suffix}']=m[f'{dev}_{suffix}'].mean()
                row[f'{dev}_max_terminal_soc_error']=abs(m[f'{dev}_window_terminal_soc']-.5).max()
            task_runs.append(row)
        log('aggregate_task',task=task)
    pd.DataFrame(task_runs).to_csv(out/'task_per_run.csv',index=False);pd.DataFrame(windows).to_csv(out/'window_metrics.csv',index=False)
    pd.DataFrame(constraints).to_csv(out/'constraint_checks.csv',index=False)
    runs=design.merge(pd.DataFrame(task_runs),on='task',validate='many_to_many');runs.to_csv(out/'control_per_run.csv',index=False)
    keys=['family','input','weight_axis','budget','region']
    summary=runs.groupby(keys)[METRICS+['threshold_exceed_fraction','threshold_exceed_s','fast_budget_active_fraction','ess_budget_active_fraction','fast_request_applied_mae_w','ess_request_applied_mae_w']].mean().reset_index()
    counts=runs.groupby(keys).agg(n_runs=('run_id','nunique'),n_assumed_day_groups=('group_id','nunique')).reset_index();summary=summary.merge(counts,on=keys)
    summary['verified_independent_groups']=0;summary.to_csv(out/'control_summary.csv',index=False)
    rng=np.random.default_rng(cfg['bootstrap_seed']);comparisons=[];logo=[];individual=[]
    for (family,weight,budget,region),g in runs.groupby(['family','weight_axis','budget','region']):
        for left,right in [('P2','persistence'),('P4','persistence'),('perfect','persistence'),('perfect','P4'),('P4','P2')]:
            a=g[g.input==left].set_index('run_id');b=g[g.input==right].set_index('run_id').reindex(a.index)
            assert not a.index.duplicated().any() and not b.p95_ramp.isna().any()
            for metric in ['p95_ramp','max_ramp','fast_throughput_wh','ess_throughput_wh']:
                ids={'family':family,'weight_axis':weight,'budget':budget,'region':region,'comparison':left+'-'+right,'metric':metric}
                row,ll,gg=paired_cluster(a,b,metric,a.group_id,rng,cfg['bootstrap_replicates']);comparisons.append({**ids,**row})
                logo.extend([{**ids,**x} for x in ll]);individual.extend([{**ids,**x} for x in gg])
    # Same MPC structure and same models; only endpoint-flat versus full path information changes.
    for (budget,region,inp),g in runs[runs.family.str.startswith('D_')].groupby(['budget','region','input']):
        a=g[g.family=='D_mpc_trajectory'].set_index('run_id');b=g[g.family=='D_mpc_endpoint'].set_index('run_id').reindex(a.index)
        for metric in ['p95_ramp','max_ramp','fast_throughput_wh','ess_throughput_wh']:
            ids={'family':'D_information','weight_axis':-1,'budget':budget,'region':region,'comparison':inp+'_trajectory-minus-endpoint','metric':metric}
            row,ll,gg=paired_cluster(a,b,metric,a.group_id,rng,cfg['bootstrap_replicates']);comparisons.append({**ids,**row});logo.extend([{**ids,**x} for x in ll]);individual.extend([{**ids,**x} for x in gg])
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False);pd.DataFrame(logo).to_csv(out/'leave_one_day_out.csv',index=False);pd.DataFrame(individual).to_csv(out/'individual_day_differences.csv',index=False)
    selected=design[design.family.isin(['B2_tuned','C_joint'])].drop(columns='task');selected.to_csv(out/'selected_weights.csv',index=False)
    selected.assign(weight_zero=selected.weight.eq(0)).groupby(['family','input']).weight_zero.mean().reset_index().to_csv(out/'weight_zero_selection_fraction.csv',index=False)
    scores=pd.concat([pd.read_csv(p) for p in sorted((out/'selection').glob('*_scores.csv'))]);scores.to_csv(out/'validation_scores.csv',index=False)
    actual=len(list((out/'tuning_cache').glob('*.json')));reused=sum('path' in json.loads(p.read_text())['source'] for p in (out/'tuning_cache').glob('*.json'))
    (out/'search_budget.json').write_text(json.dumps({'unique_physical_candidates_per_partition_budget':78,'candidate_options_per_learned_or_perfect_input':30,'reactive_or_persistence_candidates':6,
        'global_weight_zero_candidates_shared':6,'candidate_checkpoint_count':actual,'reused_nested_validation_checkpoints':reused,'selection':'feasible; run-balanced p95, max ramp, throughput, lower weight, candidate index'},indent=2))


def visualize_report(cfg,out,smoke,log):
    figs=out/'figures';figs.mkdir(exist_ok=True);plotdata=out/'plotting_csv';plotdata.mkdir(exist_ok=True)
    summary=pd.read_csv(out/'control_summary.csv');runs=pd.read_csv(out/'control_per_run.csv');paired=pd.read_csv(out/'paired_comparisons.csv');nr=runs.run_id.nunique();ng=runs.group_id.nunique()
    caption=f'E1 PCC proxy | reused exploratory evaluation | run-balanced | n={nr} runs, {ng} assumed day groups\n95% CI where shown: 2000 day-cluster resamples; verified independent groups=0; same budget does not imply same usage'
    def save(fig,name,data):
        fig.text(.5,.005,caption,ha='center',fontsize=7);fig.tight_layout(rect=(0,.07,1,1));fig.canvas.draw()
        fig.savefig(figs/f'{name}.png',dpi=300,bbox_inches='tight',pad_inches=.12);fig.savefig(figs/f'{name}.svg',bbox_inches='tight',pad_inches=.12);plt.close(fig);data.to_csv(plotdata/f'{name}.csv',index=False)
    colors={'persistence':'black','P2':'tab:blue','P4':'tab:orange','perfect':'tab:green'}
    s=summary[(summary.family=='C_weight')&(summary.budget==1)&(summary.region=='whole')]
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,metric in zip(axs,['p95_ramp','max_ramp']):
        for model,g in s.groupby('input'):ax.plot(g.weight_axis,g[metric],'o-',label=model,color=colors[model])
        ax.set(xlabel='Forecast weight (other parameters fixed)',ylabel=metric+' W/s');ax.legend()
    save(fig,'weight_sensitivity',s)
    s=summary[(summary.family.isin(['B1_fixed','B2_tuned','C_joint','D_mpc_trajectory']))&(summary.region=='whole')]
    fig,axs=plt.subplots(2,2,figsize=(12,9))
    for ax,family in zip(axs.flat,['B1_fixed','B2_tuned','C_joint','D_mpc_trajectory']):
        for model,g in s[s.family==family].groupby('input'):ax.plot(g.budget,g.p95_ramp,'o-',label=model,color=colors[model])
        ax.set(title=family,xlabel='Separate-device budget multiplier',ylabel='p95 |ramp| W/s');ax.legend(fontsize=8)
    save(fig,'budget_performance',s)
    for metric in ['p95_ramp','max_ramp']:
        s=summary[(summary.family.isin(['C_joint','D_mpc_trajectory']))&(summary.region=='whole')]
        fig,axs=plt.subplots(1,2,figsize=(12,5))
        for ax,dev in zip(axs,['fast','ess']):
            for (family,model),g in s.groupby(['family','input']):ax.plot(g[f'{dev}_mean_abs_w'],g[metric],marker='o',linestyle='-' if family=='C_joint' else '--',color=colors[model],label=family+' '+model)
            ax.set(xlabel=f'{dev} actual Wh/h (mean absolute W)',ylabel=metric+' W/s');ax.legend(fontsize=6,ncol=2)
        save(fig,'throughput_'+metric,s)
    h=pd.read_csv(out/'horizon_forecast_metrics.csv');hs=h.groupby(['model','horizon_s']).mae.mean().reset_index()
    fig,ax=plt.subplots(figsize=(8,5))
    for model,g in hs.groupby('model'):ax.plot(g.horizon_s,g.mae,'o-',label=model,color=colors[model])
    ax.set(xlabel='Target horizon seconds',ylabel='Run-balanced MAE W',title=f'All forecast runs n={h.run_id.nunique()}');ax.legend();save(fig,'horizon_prediction_error',hs)
    forecast=pd.read_csv(out/'forecast_per_run.csv');f=forecast[forecast.scope=='control_common_timestamps'].pivot(index='run_id',columns='model',values='mae')
    a=runs[(runs.family=='C_joint')&(runs.budget==1)&(runs.region=='whole')].pivot(index='run_id',columns='input',values='p95_ramp')
    scatter=pd.DataFrame({'forecast_gain_P2_minus_P4_W':f.P2-f.P4,'control_gain_P2_minus_P4_W_s':a.P2-a.P4}).dropna().reset_index()
    fig,ax=plt.subplots(figsize=(8,5));ax.scatter(scatter.forecast_gain_P2_minus_P4_W,scatter.control_gain_P2_minus_P4_W_s);ax.axhline(0,color='gray');ax.axvline(0,color='gray');ax.set(xlabel='P2 MAE - P4 MAE (W)',ylabel='P2 p95 - P4 p95 (W/s)',title='Each point is one run; not independent sessions');save(fig,'forecast_control_gain_scatter',scatter)
    s=summary[(summary.family=='C_joint')&(summary.budget==1)]
    fig,ax=plt.subplots(figsize=(10,5));positions={'operation':0,'recovery':1,'whole':2}
    for k,model in enumerate(INPUT_ORDER):
        g=s[s.input==model];ax.bar(np.array([positions[x] for x in g.region])+k*.18,g.p95_ramp,width=.18,label=model,color=colors[model])
    ax.set_xticks([.27,1.27,2.27]);ax.set_xticklabels(['operation 0-40s','recovery 40-60s','whole 60s']);ax.set_ylabel('Run-balanced p95 ramp W/s');ax.legend();save(fig,'operation_recovery_whole',s)
    # Predefined run selection: best, median, worst P4-versus-persistence whole-run delta.
    cr=runs[(runs.family=='C_joint')&(runs.budget==1)&(runs.region=='whole')]
    a=cr[cr.input=='P4'].set_index('run_id');b=cr[cr.input=='persistence'].set_index('run_id').reindex(a.index)
    order=(a.p95_ramp-b.p95_ramp).sort_values();chosen=[('best',order.index[0]),('middle',order.index[len(order)//2]),('worst',order.index[-1])];cases=[]
    wm=pd.read_csv(out/'window_metrics.csv')
    for label,run in chosen:
        # Select the window with the largest P4 max ramp in that run, report both run and window numbers.
        task=a.loc[run,'task'];w=wm[(wm.task==task)&(wm.run_id==run)&(wm.region=='whole')].sort_values(['max_ramp','window'],ascending=[False,True]).iloc[0]
        paths={};traces={}
        for model in INPUT_ORDER:
            row=cr[(cr.run_id==run)&(cr.input==model)].iloc[0];folder=out/'control_cache'/row.task
            path=next(p for p in folder.glob('*.parquet') if p.name.startswith(run+'_') and f'_s{w.segment}_w{w.window}.parquet' in p.name)
            paths[model]=path;traces[model]=pd.read_parquet(path)
        z=traces['P4'];t=z.issuance_s.to_numpy();max_t=float(w.max_ramp_time_s)
        fig,axs=plt.subplots(6,1,figsize=(12,14),sharex=True)
        axs[0].plot(t,z.current_w,label='load');axs[0].plot(t+1,z.P2,label='P2 at target');axs[0].plot(t+1,z.P4,label='P4 at target');axs[0].plot(t+1,z.actual_w,label='actual future',alpha=.6)
        for model,q in traces.items():
            axs[1].plot(t,q.pcc_w,label=model);axs[2].plot(t,q.fast_w,label=model);axs[3].plot(t,q.ess_w,label=model)
            axs[4].plot(t,q.fast_soc,label=model+' fast');axs[4].plot(t,q.ess_soc,'--',label=model+' ESS')
            axs[5].plot(t,q.fast_credit_wh,label=model+' fast');axs[5].plot(t,q.ess_credit_wh,'--',label=model+' ESS')
        for ax,y in zip(axs,['Load / forecast W','PCC W','fast W','ESS W','SoC','Credit Wh']):
            ax.set_ylabel(y);ax.legend(ncol=4,fontsize=6);ax.axvspan(t[0]+40,t[-1],color='gray',alpha=.08);ax.axvline(max_t,color='red',ls=':',lw=.7)
        axs[0].set_title(f'{label}: {run}; max-ramp window, not assumed representative');axs[-1].set_xlabel('Time seconds')
        casecsv=pd.concat([q.assign(plotted_input=m) for m,q in traces.items()],ignore_index=True);save(fig,'case_'+label,casecsv)
        fig,ax=plt.subplots(figsize=(11,5))
        for model,q in traces.items():ax.plot(t,q.pcc_w,label=model)
        ax.axvline(max_t,color='red',ls=':',label='P4 maximum ramp');ax.axvline(t[0]+40,color='gray',ls='--',label='recovery begins');ax.axvspan(t[0]+40,t[-1],alpha=.08,color='gray')
        ax.set_xlim(max(t[0],min(max_t,t[0]+40)-2),min(t[-1],max(max_t,t[0]+40)+2));ax.set(xlabel='Time seconds',ylabel='PCC W',title='Maximum ramp and operation/recovery boundary');ax.legend(fontsize=7);save(fig,'zoom_'+label,casecsv)
        baseline=wm[(wm.task==b.loc[run,'task'])&(wm.run_id==run)&(wm.region=='whole')&(wm.window==w.window)&(wm.segment==w.segment)].iloc[0]
        cases.append({'case':label,'run_id':run,'run_P4_minus_reactive_p95':order[run],'plotted_window':w.window,'segment':w.segment,'window_P4_minus_reactive_p95':w.p95_ramp-baseline.p95_ramp,
                      'window_P4_max_ramp':w.max_ramp,'max_ramp_time_s':max_t,'coincident_candidates':w.max_ramp_coincident_candidates,'rule':'best/median/worst whole-run P4-persistence delta; within run largest P4 max-ramp window, deterministic tie break'})
    pd.DataFrame(cases).to_csv(out/'case_selection.csv',index=False)
    def table(d):return d.to_markdown(index=False,floatfmt='.5g')
    def value(family,inp,metric='p95_ramp',region='whole'):
        return float(summary[(summary.family==family)&(summary.input==inp)&(summary.budget==1)&(summary.region==region)][metric].iloc[0])
    fs=pd.read_csv(out/'forecast_summary.csv');allf=fs[fs.scope=='all_forecast'].set_index('model')
    forecast_delta=float(allf.loc['P4','mae']-allf.loc['P2','mae'])
    perfect_delta=value('B1_fixed','perfect')-value('B1_fixed','P4');endpoint_vsreactive=value('B2_tuned','perfect')-value('B2_tuned','persistence')
    trajectory_delta=value('D_mpc_trajectory','perfect')-value('D_mpc_endpoint','perfect')
    joint_delta=value('C_joint','P4')-value('C_joint','persistence')
    decisions=[{'type':1,'judgment':'현재 endpoint core에서 개선 여지 관측' if perfect_delta<0 else '판정 불충분','evidence':f'B1 perfect−P4 whole p95={perfect_delta:.5g} W/s','limit':'Perfect diagnostic; not deployable or an upper bound'},
               {'type':2,'judgment':'현재 endpoint 활용 방식 개선 필요 근거' if endpoint_vsreactive>=0 else 'perfect endpoint 활용 가능성 관측','evidence':f'B2 perfect−reactive whole p95={endpoint_vsreactive:.5g} W/s','limit':'Does not establish uselessness of all forecasting'},
               {'type':3,'judgment':'같은 MPC 내 궤적 정보 이점 관측' if trajectory_delta<0 else '궤적 정보 이점 미확인','evidence':f'MPC perfect trajectory−endpoint whole p95={trajectory_delta:.5g} W/s','limit':'Fixed LP objective and budget; actual usage may differ'},
               {'type':4,'judgment':'전체 run 평균 MAE에서 metadata 추가 가치 미확인' if forecast_delta>=0 else '전체 run 평균 MAE에서 metadata 이점 관측','evidence':f'P4−P2 MAE={forecast_delta:.5g} W','limit':'Date/hardware/workload confounding; inspect tails and regional errors'},
               {'type':5,'judgment':'joint weight 선택 후 P4 추가 가치 미확인' if joint_delta>=0 else 'joint weight 선택 후 P4 이점 관측','evidence':f'C joint P4−reactive whole p95={joint_delta:.5g} W/s','limit':'Only current learned models/controllers and device assumptions'},
               {'type':6,'judgment':'정식 독립 일반화 판정 불충분','evidence':'4 assumed collection dates; 0 verified independent groups','limit':'Exploratory reused evaluation, not confirmatory significance'}]
    pd.DataFrame(decisions).to_csv(out/'final_judgments.csv',index=False)
    if endpoint_vsreactive>=0 and trajectory_delta<0:recommendation='추가 endpoint 모델 구조 탐색보다 예측을 사용하는 제어 방식과 궤적 target을 먼저 검토할 근거가 있다. 기존 데이터에서는 이번 사전 제한된 진단에서 멈추고, 독립 수집과 deadline을 검증한 뒤 확대하는 것이 타당하다.'
    elif perfect_delta<0:recommendation='같은 endpoint 정책에서 perfect 입력의 개선 여지가 관측돼 예측오차 개선을 제한적으로 검토할 근거가 있다. 다만 metadata 확대나 새 구조 탐색은 독립 수집 자료와 실제 사용량·최대 ramp 검증 후 결정해야 한다.'
    else:recommendation='현재 endpoint 정책에서 모델 오차만 줄이는 추가 개발의 근거가 부족하다. 무제한 모델 탐색을 이어가기보다 제어 목적·target·데이터 독립성 및 물리 가정을 먼저 점검하고 현재 데이터의 탐색을 멈추는 것이 타당하다.'
    constraints=pd.read_csv(out/'constraint_checks.csv');vcols=[c for c in constraints if 'violation' in c or 'end_energy_delta' in c];bad=int((constraints[vcols].abs()>1e-6).any(axis=1).sum())
    primary=summary[(summary.budget==1)&(summary.region=='whole')&~summary.family.eq('C_weight')]
    pc=paired[(paired.budget==1)&(paired.region=='whole')&paired.metric.isin(['p95_ramp','max_ramp'])&~paired.family.eq('C_weight')]
    solver=pd.read_csv(out/'task_per_run.csv');solver=solver[solver.region=='whole'];solver=solver[solver.task.str.startswith('mpc_')]
    solver[['task','run_id','solver_calls','solver_s_sum','solver_s_max','solver_fallbacks','deadline_exceed_count']].to_csv(out/'solver_diagnostics.csv',index=False)
    inc=pd.read_csv(out/'run_inclusion_manifest.csv');weights=pd.read_csv(out/'weight_zero_selection_fraction.csv')
    report=f'''# AIDC 예측모델과 제어 활용 방식 최종 진단

**기존 평가셋 재사용 탐색적 진단 · E1 actuator 가정 · {'smoke (제한 표본·solver 호출 제한, 과학적 결론 아님)' if smoke else 'full 실행'}**

## 최종 판단

{recommendation}

{table(pd.DataFrame(decisions))}

관측 범위는 이 데이터·장치·예산·선택 절차에 한정된다. 모든 가능한 모델의 한계나 예측 불가능성을 입증하지 않았다. Perfect 입력은 배포 불가능한 진단용 미래정보이며 최적 성능 상한이 아니다.

## 평가 감사와 재사용

기존 nested 원본 CSV primary C2/C3/C4 p95는 각각 2508.5107 / 2640.1045 / 2686.8911 W/s였다. 원본 재확인 표는 verified_previous_primary.csv. 기존 결과는 수정하지 않았다. 수정된 4개 날짜 outer split과 12개 inner partition, 학습표본, 모델 checksum과 복원 q50 일치를 확인해 재사용했다. 이번 장치 조건은 각 fold의 기존 calibration을 그대로 사용하며 모든 paired 비교는 같은 fold/run/timestamp에서 수행한다.

원본 고유 run {len(inc)}개 중 feature/target 가능한 run {int(inc.forecast_included.sum())}개, 60초 control window가 있는 run {int(inc.control_included.sum())}개. 제외 규칙을 성능에 맞춰 변경하지 않았다. 예측 평가는 제어 제외 run도 포함했다. 자세한 segment 길이와 metadata는 run_inclusion_manifest.csv 및 segment_manifest.csv에 있다.

{table(inc[['run_id','raw_duration_s','max_segment_s','feature_target_samples','valid_control_windows','control_included','hardware','workload','model_size_token','batch_size','collection_day','exclusion_reason']])}

## A. 같은 최신 split에서 예측이 좋은가?

아래는 run-balanced 평균이다. all_forecast는 모든 예측 가능 run/timestamp, control_common_timestamps는 동일 제어 평가 window의 발행 시각만 포함한다. Persistence는 같은 point metric으로 비교하고, 분포 평가는 15개 동일 값의 퇴화분포를 같은 quantile grid + constant tail pinball 적분으로 평가했다. CRPS approximation을 섞지 않았다. 상승/하강/정상은 1초 endpoint 평균 변화가 ±500 W/s를 넘는지로 분류하며, 큰 전이는 실제 200ms 경로의 최대 absolute ramp가 500 W/s를 넘는 경우다. 후자는 endpoint 차이를 최대 ramp로 대체하지 않는다.

{table(fs[['scope','model','mae','rmse','signed_bias','p95_abs_error','p99_abs_error','crps_grid','q10_q90_coverage','q10_q90_width_w']])}

Quantile별 pinball, 구간별 과소·과대 예측, 날짜/hardware/workload 표는 forecast_per_run.csv, forecast_regions.csv, forecast_by_*.csv에 보존했다. 계절성 baseline은 training에서 주기를 확인하는 별도 근거가 없어 추가하지 않았다. 이전 5-fold 개선율을 이 결과로 가져오지 않았다.

## B/C. 입력 교체와 weight 선택

B1_fixed는 reference slew 500 W/s, alpha .15, weight .5를 고정하고 입력만 교체했다. B2_tuned는 weight .5에서 기존 6개 후보를 동일 inner 절차로 선택했다. C_weight는 나머지 parameter를 고정하고 weight 0/.25/.5/.75/1을 비교한다. C_joint는 weight와 기존 6개 후보를 함께 선택했다. Weight=0은 같은 parameter의 reactive와 수치적으로 일치하는 공통 core를 사용한다. Persistence 입력은 모든 weight에서 같은 행동이어서 공유했다. 각 learned/perfect 입력은 30개의 고유 선택 후보를 갖고 reactive/persistence는 6개다. 공통 weight=0 후보를 포함한 실제 물리 후보는 partition/budget당 78개로 중복 실행하지 않았다. 재사용된 validation cache 수는 search_budget.json에 기록했다.

{table(primary[['family','input','p95_ramp','max_ramp','fast_throughput_wh','ess_throughput_wh','n_runs','n_assumed_day_groups']])}

{table(weights)}

Outer fold/budget별 선택 weight와 score는 selected_weights.csv, validation_scores.csv. Reactive 후보를 포함해도 outer-test에서 더 좋아짐이 보장되지 않는다. Inner 선택과 일반화 실패를 숨기지 않았다.

## D. Endpoint와 경로 정보

실행 근거는 사전 정의한 구조적 모호성: 동일 endpoint에도 중간 부하 경로는 다를 수 있다. 이 이유로 B/C의 좋은 조합이 나올 때까지 탐색하지 않고 제한된 궤적 진단을 수행했다. 기존 q50 재사용 가능한 1초를 제외한 h=0.2/.4/.6/.8초 P2/P4만 동일 history·feature·고정 LightGBM q50로 학습했다. 불필요한 quantile 재학습은 없다. 1초는 기존 15-quantile 정렬 median을 그대로 재사용했다. 예측 대상과 학습/validation partition은 바꾸지 않았다. 독립 horizon median을 joint confidence band로 취급하지 않았다.

MPC는 5개 bin의 PCC 차이 절대합 + .01×장치 절대출력 합을 최소화한다. 매 계획 prefix에서 출력·slew·효율 에너지·SoC·장치별 처리량 및 복구예약 제약을 포함한다. 계산 실패/시간초과/동시 charge-discharge relaxation 해는 causal zero-request fallback을 사용하며 동일 executor가 최종 제약을 집행한다. 40초 이후 공통 SoC 복구는 바꾸지 않았다. 목적함수·가중치 추가 탐색은 하지 않았다.

D_mpc_trajectory와 D_mpc_endpoint는 같은 새 controller 안에서 실제 5-step 경로와 endpoint 반복 경로를 비교한다. Persistence는 두 경로가 동일하여 checkpoint를 공유한다. 새 learned controller와 기존 reactive만 비교해 이득을 모두 예측 효과로 부르지 않는다. MPC 내부 persistence 대비 차이, 같은 MPC에서 trajectory−endpoint 차이를 따로 제시한다.

Solver 진단: calls={int(solver.solver_calls.sum())}, fallback sample 수={int(solver.solver_fallbacks.sum())}, 최대 solve 시간={solver.solver_s_max.max():.6g}s, 200ms 초과={int(solver.deadline_exceed_count.sum())}. Solver 시간은 실제 배포 latency 증명이 아니다. 초과가 있으면 기본 delay를 유지한 E1 결과를 realtime 실행 가능성으로 해석하지 않는다. 상세는 solver_diagnostics.csv.

## 운전·복구·전체 구간

{table(summary[(summary.family.isin(['C_joint','D_mpc_trajectory']))&(summary.budget==1)][['family','input','region','p95_ramp','p99_ramp','max_ramp','fast_throughput_wh','ess_throughput_wh','fast_budget_active_fraction','ess_budget_active_fraction']])}

실제 timestamp 차이로 ramp를 계산했다. 운전→복구 경계 ramp는 도착 sample을 기준으로 복구에 배정했다. 전체 p95를 구간별 분위수 가중합으로 만들지 않았다. 최대 ramp 시각과 부하 전이/reference 변경/포화·해제/budget/복구/출력 zero 복귀의 동시 발생은 window_metrics.csv와 task_per_run.csv에 기록했다. 동시 발생은 인과 증명이 아니다.

## 제약·처리량과 통계 한계

출력·SoC·budget·energy balance·종료 조건 허용오차 1e-6를 넘는 위반 행 {bad}개. SoC와 출력을 마지막에 덮어쓰지 않았다. 처리량은 장치별 AC 절대 충방전 Wh이며 손실·순 소비·열화와 다르다. 동일 예산이어도 실제 사용량은 다를 수 있다. 사용량을 맞추기 위한 불필요한 충방전을 추가하지 않았다. p95 개선과 최대 ramp 악화·사용량 증가는 별도로 읽어야 한다.

{table(pc[['family','comparison','metric','mean_difference','ci_low','ci_high','n_runs','n_assumed_groups','status']])}

확인된 독립 group은 0개다. 수집일 4개를 가정한 2,000회 cluster bootstrap으로 group 합계/표집된 run 수를 사용하여 run-balanced estimand를 유지했다. Timestamp/segment/seed를 독립 표본으로 부풀리지 않았다. 이 CI는 고정 예측·선택 정책과 날짜 독립 가정에 조건부이며 정식 독립 검증/확인적 유의성 검정이 아니다. 개별 날짜와 leave-one-day-out 결과는 individual_day_differences.csv, leave_one_day_out.csv에서 확인한다. 날짜와 hardware/workload 교락은 해소되지 않았다.

## 사례와 시각화

{table(pd.DataFrame(cases))}

사전 규칙은 whole-run P4−reactive p95 차이의 최소/중앙/최대 run이다. 이름 best/middle/worst는 순위이며 부호에 따라 실제 성공/실패 여부가 달라진다. 각 run에서 P4 최대 ramp가 가장 큰 window를 표시하고 전체 run과 해당 window의 차이를 모두 기록했다. 첫 window가 전체 run을 대표한다고 가정하지 않았다. 시간 꺾은선과 운전/복구 경계·최대 ramp 확대 그림은 figures/에 PNG 300 dpi 및 SVG로 저장했고 plotting_csv/에 원자료를 저장했다. 순서 없는 모델명에는 꺾은선을 사용하지 않았다.

## 최종 개발 방향과 남은 미완료

{recommendation}

정식 독립 수집 group 확인은 자료 부재로 미완료다. 보고서 실제 HTML 렌더링 결과는 render_verification.json에서 별도로 확인한다. 독립성 미확인을 계산 단계 실패와 혼동하지 않았고, background 시작을 완료로 표시하지 않았다. 모든 단계 상태와 최종 exit code는 completion_status.json 및 exit_code.txt를 따른다.
'''
    (out/'report_ko.md').write_text(report)
    gallery=''.join(f'<figure><img src="figures/{p.name}" alt="{p.stem}"><figcaption>{p.stem}</figcaption></figure>' for p in sorted(figs.glob('*.png')))
    html='<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>AIDC 예측·제어 최종 진단</title><style>body{font-family:"NanumGothic",sans-serif;max-width:1200px;margin:24px auto;padding:0 24px;line-height:1.7;overflow-wrap:anywhere}table{border-collapse:collapse;display:block;overflow-x:auto;max-width:100%;font-size:12px}th,td{border:1px solid #ddd;padding:6px}img{max-width:100%;height:auto}figure{margin:32px 0}pre{overflow:auto}h1,h2{line-height:1.35}</style></head><body>'+markdown.markdown(report,extensions=['tables','fenced_code'])+'<h2>그림</h2>'+gallery+'</body></html>'
    (out/'report_ko.html').write_text(html)
    log('rendering_report')
    try:
        with open(out/'render.log','w') as f:r=subprocess.run([sys.executable,str(Path(__file__).with_name('render_report.py')),str(out)],stdout=f,stderr=subprocess.STDOUT,timeout=55)
        if r.returncode:raise RuntimeError('renderer exit '+str(r.returncode))
        info=json.loads((out/'render_verification.json').read_text());assert info['all_images_loaded'] and not info['horizontal_overflow']
        return True
    except Exception as e:
        (out/'render_verification.json').write_text(json.dumps({'status':'blocked_or_failed','reason':str(e),'log':'render.log'}));return False

INPUT_ORDER=['persistence','P2','P4','perfect']
