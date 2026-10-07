import unittest
import numpy as np
import yaml
from pathlib import Path
from control import simulate,metrics,validate


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
        self.device=dict(fast_max_w=150.,ess_max_w=150.,fast_capacity_wh=6.,ess_capacity_wh=600.)
        self.demand=1000+800*(np.arange(300)%17<8)
    def test_constraints_recovery_budget(self):
        for mode in ('C2','C3','C4','C5'):
            for rates in ([0.,0.],[.01,.01],[20.,50.],[500.,500.]):
                z=simulate(self.demand,self.demand[::-1],mode,self.cfg,self.device,rates)
                validate(metrics(z,self.cfg,self.device))
                self.assertLess(abs(z.fast_w.iloc[-1])+abs(z.ess_w.iloc[-1]),1e-6)
    def test_causal_prefix(self):
        d=self.demand.copy(); f=d.astype(float); altered=d.copy(); altered[80:]+=9000; other=f.copy(); other[80:]-=8000
        a=simulate(d,f,'C4',self.cfg,self.device,[100.,100.])
        b=simulate(altered,other,'C4',self.cfg,self.device,[100.,100.])
        np.testing.assert_allclose(a.iloc[:80][['pcc_w','fast_w','ess_w']],b.iloc[:80][['pcc_w','fast_w','ess_w']])
    def test_delay_and_zero_budget(self):
        d=np.ones(300)*1000; d[10:]=1800
        a=simulate(d,d,'C2',self.cfg,self.device,[100.,100.])
        self.assertEqual(a.fast_w.iloc[10]+a.ess_w.iloc[10],0)
        self.assertGreater(a.fast_w.iloc[11]+a.ess_w.iloc[11],0)
        z=simulate(d,d,'C4',self.cfg,self.device,[0.,0.])
        np.testing.assert_allclose(z.pcc_w,d,atol=1e-8)
    def test_large_capacity_tight_recovery_credit(self):
        device=dict(fast_max_w=1000.,ess_max_w=1000.,fast_capacity_wh=40.,ess_capacity_wh=4000.)
        rng=np.random.default_rng(2)
        d=5000+rng.normal(0,1000,300)
        for rate in (1.,50.,100.,200.):
            z=simulate(d,d[::-1],'C4',self.cfg,device,[rate,rate],.5)
            validate(metrics(z,self.cfg,device))

    def test_no_control(self):
        a=simulate(self.demand,self.demand,'C0',self.cfg,self.device,[100.,100.])
        b=simulate(self.demand,self.demand,'C1',self.cfg,self.device,[100.,100.])
        np.testing.assert_allclose(a.pcc_w,b.pcc_w)
        np.testing.assert_allclose(a.pcc_w,self.demand)

if __name__=='__main__': unittest.main()
