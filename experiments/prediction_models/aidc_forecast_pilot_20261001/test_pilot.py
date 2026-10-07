import unittest
import numpy as np
import pandas as pd
from data import continuous_features,columns,sample,spaced,weights,encode,prediction_frame,metrics,summarize


class PilotChecks(unittest.TestCase):
    def raw(self):
        p=1000+np.arange(200,dtype=float)
        return pd.DataFrame({'run_id':'r','segment':0,'bin':np.arange(200),'issuance_s':(np.arange(200)+1)*.2,'current_w':p,'y_5':pd.Series(p).shift(-5),'collection_day':'d'})
    def test_causal_features_and_common_validity(self):
        raw=self.raw();a=continuous_features(raw);other=raw.copy();other.loc[100:,'current_w']+=10000;other['y_5']=other.current_w.shift(-5);b=continuous_features(other)
        pd.testing.assert_frame_equal(a.loc[:99,columns(10)],b.loc[:99,columns(10)])
        self.assertEqual(int(a.valid_anchor.sum()),146)
        self.assertFalse(a.valid_anchor.iloc[48]);self.assertTrue(a.valid_anchor.iloc[49])
        self.assertNotIn('lag_5',columns(1));self.assertNotIn('lag_25',columns(5));self.assertNotIn('lag_50',columns(10))
    def test_missing_bin_and_segment_boundary(self):
        raw=self.raw();raw.loc[80,'current_w']=np.nan;raw['y_5']=raw.current_w.shift(-5);d=continuous_features(raw)
        self.assertFalse(d.valid_anchor.iloc[75:130].any())
        raw=self.raw();raw.loc[100:,'segment']=1;raw.loc[95:99,'y_5']=np.nan;d=continuous_features(raw)
        self.assertFalse(d.valid_anchor.iloc[95:149].any())
    def test_sampling_caps_stable_independent_of_labels(self):
        g=pd.concat([self.raw().assign(issuance_s=lambda d:d.issuance_s+40*k,bin=lambda d:d.bin+200*k) for k in range(20)],ignore_index=True)
        cand=spaced(g);self.assertTrue((np.diff(cand.issuance_s)>=1-1e-8).all())
        a=sample(cand,300,True,17);b=sample(cand.assign(actual_w=100000),300,True,17)
        np.testing.assert_array_equal(a.bin,b.bin);self.assertEqual(len(a),300)
        z=sample(cand,600,False,17);self.assertEqual(len(z),600);self.assertFalse(z.bin.duplicated().any())
    def test_weights_and_training_only_encoding(self):
        d=pd.DataFrame({'run_id':['a','a','a','b'],'hardware':['H','H','H','B'],'current_w':[1,2,3,4]})
        w=weights(d);self.assertAlmostEqual(w[:3].sum(),w[3]);self.assertAlmostEqual(w.mean(),1)
        _,x,m,_=encode(d,pd.DataFrame({'hardware':['NEW'],'current_w':[4]}),['hardware','current_w'])
        self.assertEqual(x.hardware.iloc[0],-1);self.assertNotIn('NEW',m['hardware'])
    def test_target_restoration_persistence_and_p99(self):
        ev=self.raw().head(60);ev['actual_w']=ev.current_w+5
        z=prediction_frame(ev,ev.current_w.to_numpy(),'persistence','test',10,20)
        per=metrics(z);summary=summarize(per)
        self.assertTrue(per.p99_abs_error.isna().all());self.assertTrue((summary.mae==5).all());self.assertTrue((summary.signed_bias==-5).all())
        delta=ev.actual_w-ev.current_w;np.testing.assert_allclose(ev.current_w+delta,ev.actual_w)


if __name__=='__main__':unittest.main()
