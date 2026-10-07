import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import diagnosis
from core import simulate,metrics,validate
from reporting import region_stats
from mpc import step

class DiagnosisChecks(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
        self.dev={'fast_max_w':150.,'ess_max_w':150.,'fast_capacity_wh':6.,'ess_capacity_wh':600.}
        self.d=1000+500*(np.arange(300)%17<8)
    def test_zero_weight_matches_reactive(self):
        cfg={**self.cfg,'forecast_weight':0.}
        a=simulate(self.d,self.d,'C2',cfg,self.dev,[50.,50.]);b=simulate(self.d,self.d[::-1],'C4',cfg,self.dev,[50.,50.])
        pd.testing.assert_frame_equal(a.drop(columns='forecast_w'),b.drop(columns='forecast_w'))
        self.assertEqual(diagnosis.canonical('P2',0,2),diagnosis.canonical('P4',0,2))
        self.assertEqual(diagnosis.canonical('persistence',1,2),diagnosis.canonical('perfect',0,2))
    def test_causal_prefix(self):
        f=self.d.astype(float);b=f.copy();b[80:]+=4000
        a=simulate(self.d,f,'C4',self.cfg,self.dev,[50.,50.]);z=simulate(self.d,b,'C4',self.cfg,self.dev,[50.,50.])
        pd.testing.assert_frame_equal(a.iloc[:80],z.iloc[:80])
    def test_planner_and_terminal_physics(self):
        path=np.repeat(self.d[:,None],5,axis=1)
        z=simulate(self.d,self.d,'C2',self.cfg,self.dev,[50.,50.],trajectory=path)
        validate(metrics(z,self.cfg,self.dev))
        self.assertLess(abs(z.fast_w.iloc[-1])+abs(z.ess_w.iloc[-1]),1e-6)
        self.assertTrue((z.solver_status=='optimal_lp_diagnostic').any())
    def test_infeasible_planner_fallback_is_zero(self):
        command,info=step(np.ones(5)*1000,1000,np.array([3.,300.]),np.zeros(2),np.array([-1.,-1.]),self.cfg,self.dev,5)
        self.assertTrue(info['solver_fallback']);np.testing.assert_allclose(command,0)
    def test_recovery_boundary_not_dropped(self):
        z=simulate(np.ones(300)*1000,np.ones(300)*1000,'C2',self.cfg,self.dev,[50.,50.]);z['issuance_s']=np.arange(300)*.2
        z['window']='0';z['segment']=1;z.loc[200:,'pcc_w']=1100
        _,a=region_stats(z,self.cfg,'operation');_,r=region_stats(z,self.cfg,'recovery');_,w=region_stats(z,self.cfg,'whole')
        self.assertEqual(len(a)+len(r),len(w));self.assertEqual(np.max(abs(a)),0);self.assertAlmostEqual(np.max(abs(r)),500)
    def test_identical_endpoint_different_path(self):
        a=np.array([0.,0.,0.,0.,100.]);b=np.array([100.,0.,100.,0.,100.]);self.assertEqual(a[-1],b[-1]);self.assertFalse(np.array_equal(a,b))

if __name__=='__main__':unittest.main()
