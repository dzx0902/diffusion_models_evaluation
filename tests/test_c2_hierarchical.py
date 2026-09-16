import sys
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'src'):
    sys.path.insert(0,str(path))
from ms_video_eval.c2_residual import fit_pca,encode,decode
from scripts.run_c2_sessions import features,ridge_fit
from scripts.run_c2_hierarchical import train_centers,predict_parts,fit,evaluate


def fixture():
    torch.manual_seed(9)
    target=torch.randn(24,3,8)
    pca=fit_pca(target[:12],4)
    z=encode(target,pca)
    return {'eeg':torch.randn(24,3,62,128),'z':z,'pca':pca,'categories':torch.arange(24)%6,
            'splits':{'train':list(range(12)),'validation':list(range(12,18)),'test':list(range(18,24))},
            'caption_ids':torch.arange(24),'ids':[str(i) for i in range(24)],
            'floor_mse':(target-decode(z,pca)).square().flatten(1).mean(1),
            'raw_mean_mse':(target.flatten(1)-pca['mean']).square().mean(1)}


def test_train_centers_remove_class_means():
    package=fixture()
    z,c=package['z'][:12],package['categories'][:12]
    centers=train_centers(z,c)
    residual=z-centers[c]
    for category in range(6):
        assert torch.allclose(residual[c==category].mean(0),torch.zeros(4),atol=1e-6)
    with pytest.raises(ValueError):
        train_centers(z[c!=5],c[c!=5])


def test_fit_ignores_test_targets_and_unseen_session(tmp_path):
    torch.set_num_threads(1)
    package=fixture()
    x=features(package['eeg'][:12,:2])
    y=torch.nn.functional.one_hot(package['categories'][:12],6)[:,None].expand(-1,2,-1).flatten(0,1).double()
    classifier=ridge_fit(x,y,1.)
    a,grid=fit(package,classifier,[0,1],'fixture','classifier')
    changed=dict(package,z=package['z'].clone(),eeg=package['eeg'].clone(),categories=package['categories'].clone())
    changed['z'][18:]+=100
    changed['categories'][18:]=0
    changed['eeg'][:,2]+=1000
    b,_=fit(changed,classifier,[0,1],'fixture','classifier')
    assert a['residual_weight']==b['residual_weight']
    assert a['temperature']==b['temperature']
    assert torch.equal(a['centers'],b['centers'])
    assert torch.equal(a['residual_fit']['weights'],b['residual_fit']['weights'])
    coarse,residual,probabilities=predict_parts(package['eeg'][18:,:2],a)
    assert coarse.shape==residual.shape==(6,4)
    assert torch.allclose(probabilities.sum(-1),torch.ones(6))
    assert torch.allclose(coarse,probabilities@a['centers'])
    result=evaluate(package,a,tmp_path,2,42,False)
    assert len(result['reports'])==18
    assert result['reports']['train/session3/centers_plus_residual']['heldout_session']
    assert len(result['reports']['test/session3/centers_plus_residual']['residual_shuffle_mrr'])==2
    assert len(grid['residual_grid'])==24
