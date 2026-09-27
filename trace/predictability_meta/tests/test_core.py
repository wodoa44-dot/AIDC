import numpy as np
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'predictability'))
from src.core import crps, rearrange, steps
def test_crps_and_order():
 q=np.array([.1,.5,.9]);p=np.array([[2.,1.,3.]])
 z,c=rearrange(p);assert c==1 and np.all(np.diff(z)>=0)
 assert crps(np.array([1.]),np.array([[1.,1.,1.]]),q)[0]==0
def test_exact_horizon_units():
 assert steps(1000,200)==5
