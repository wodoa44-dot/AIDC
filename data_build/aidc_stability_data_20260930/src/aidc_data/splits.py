from __future__ import annotations
from sklearn.model_selection import GroupKFold
import numpy as np

def make(index,outer=5,inner=3):
    groups=index.split_group_id.astype(str).to_numpy();unique=np.unique(groups)
    n=min(int(outer),len(unique)); out={'split_group_is_verified_independent':False,'unique_group_count':len(unique),'outer_folds':[]}
    if n<2:return out
    for fold,(train,test) in enumerate(GroupKFold(n_splits=n).split(index,groups=groups)):
        tg=np.unique(groups[train]); nested=[]
        if len(tg)>=2:
            for a,b in GroupKFold(n_splits=min(int(inner),len(tg))).split(train,groups=groups[train]):
                nested.append({'train_groups':np.unique(groups[train[a]]).tolist(),'validation_groups':np.unique(groups[train[b]]).tolist()})
        out['outer_folds'].append({'fold':fold,'train_groups':np.unique(groups[train]).tolist(),
            'test_groups':np.unique(groups[test]).tolist(),'train_rows':len(train),'test_rows':len(test),'inner_folds':nested})
    return out
