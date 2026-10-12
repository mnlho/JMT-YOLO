"""Small CPU checks, synthetic data only; does not establish paper reproduction."""
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from revision_support import PairedDataset, seed_everything, safe_dynamic_k
from revision_quality import summarise


def main():
    base=Path(tempfile.mkdtemp(prefix='spl-revision-check-'))
    hyp=dict(perspective=0.,degrees=0.,scale=.2,shear=0.,translate=.1,
             hsv_h=0.,hsv_s=0.,hsv_v=0.,flipud=.5,fliplr=.5,mosaic=1.,mixup=0.)
    for tree in ['real1/images','real1/labels','real2/images','real2/labels',
                 'fake/images','seg/images','seg/labels','target/images','targetfake/images']:
        (base/tree).mkdir(parents=True)
    for i,key in enumerate(['a','b']):
        im=np.zeros((40,60,3),dtype=np.uint8);im[10:30,15:45]=255
        mask=np.zeros((40,60),dtype=np.uint8);mask[10:30,15:45]=1
        root='real%d'%(i+1)
        for tree in [root+'/images','fake/images','seg/images','target/images','targetfake/images']:
            Image.fromarray(im).save(base/tree/(key+'.png'))
        Image.fromarray(mask).save(base/'seg/labels'/(key+'.png'))
        (base/root/'labels'/(key+'.txt')).write_text('0 .5 .5 .5 .5\n')
    ds=PairedDataset([str(base/'real1/images'),str(base/'real2/images')],str(base/'fake/images'),64,hyp,str(base/'seg/images'),1,2)
    assert len(ds)==2
    seed_everything(42);a=ds[0]
    seed_everything(42);b=ds[0]
    assert torch.equal(a[0],b[0]) and torch.equal(a[6],b[6]) and torch.equal(a[2],b[2])
    for seed in range(10):
        seed_everything(seed);a=ds[0]
        assert a[0].shape==(3,64,64) and a[6].shape==(64,64)
        white=a[0][0].numpy()>200;mask=a[6].numpy()==1
        assert (white&mask).sum()/max(1,(white|mask).sum())>.85
    target=PairedDataset(str(base/'target/images'),str(base/'targetfake/images'),64,hyp)
    assert target[0][6] is None and target[0][2].shape==(0,6)
    class Assigner: topk=13
    quality,assigned=safe_dynamic_k(Assigner(),torch.tensor([[1.,1e8],[2.,1e8]]),torch.ones(2,2),2,None)
    assert not (assigned==2).any()
    p=np.array([[0,0,10,10,.9,0],[0,0,10,10,.8,0],[20,20,30,30,.7,1]],dtype=float)
    g=np.array([[0,0,10,10,0],[20,20,30,30,0]],dtype=float)
    result=summarise([(p,g,p)])
    assert result['IoU_0.5']['TP']==1 and abs(result['IoU_0.5']['F1']-.4)<1e-8
    assert result['class_error_given_class_agnostic_IoU_match']==.5
    empty=summarise([(np.empty((0,6)),np.empty((0,5)),np.empty((0,6)))])
    assert empty['IoU_0.5']['F1']==0 and empty['calibration_ECE'] is None
    print('PASS: all source roots, deterministic shared augmentation, unlabelled target, invalid-match exclusion, pseudo-label metrics/empty cases.')
    print('Synthetic files: '+str(base))


if __name__=='__main__': main()
