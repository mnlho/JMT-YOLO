"""Development-only synthetic end-to-end CPU smoke test; not a benchmark."""
import os,sys,runpy,tempfile
from pathlib import Path
import yaml,numpy as np,torch
from PIL import Image

def main():
    root=Path(__file__).resolve().parent
    base=Path(tempfile.mkdtemp(prefix='spl-trainer-check-'))
    for tree in ['source/images','source/labels','fake/images','seg/images','seg/labels',
                 'target/images','targetfake/images','val/images','val/labels']:
        (base/tree).mkdir(parents=True)
    for key in ['a','b']:
        im=np.zeros((64,64,3),np.uint8);im[16:48,16:48]=255
        mask=np.zeros((64,64),np.uint8);mask[16:48,16:48]=1
        for tree in ['source/images','fake/images','seg/images','target/images','targetfake/images']:
            Image.fromarray(im).save(base/tree/(key+'.png'))
        (base/'source/labels'/(key+'.txt')).write_text('0 .5 .5 .5 .5\n')
        Image.fromarray(mask).save(base/'seg/labels'/(key+'.png'))
    Image.fromarray(im).save(base/'val/images/c.png')
    (base/'val/labels/c.txt').write_text('0 .5 .5 .5 .5\n')
    data=dict(path=str(base),nc=1,n_segcls=2,names=['object'])
    for key,tree in dict(train_source_real='source/images',train_source_fake='fake/images',
                         train_source_seg_real='seg/images',train_source_seg_fake='seg/images',
                         train_target_real='target/images',train_target_fake='targetfake/images',
                         train_target_seg_real='target/images',train_target_seg_fake='targetfake/images',
                         test_target_real='val/images').items():data[key]=[str(base/tree)]
    (base/'data.yaml').write_text(yaml.safe_dump(data))
    cfg=yaml.safe_load((root/'models/yolov5l_city_seg.yaml').read_text())
    cfg.update(nc=1,n_segcls=2,depth_multiple=.1,width_multiple=.125)
    # Keep the original head shape contract, reduce only segmentation hidden channels.
    for layer in cfg['head']:
        if layer[2]=='SegMaskPSP':layer[3][2]=32
    (base/'model.yaml').write_text(yaml.safe_dump(cfg))
    os.environ['WANDB_MODE']='disabled'
    torch.set_num_threads(2)
    # Original repository uses old NumPy aliases; formal instructions pin NumPy 1.23.5.
    if not hasattr(np,'int'):np.int=int
    sys.argv=['ssda_yolov5_train_revision.py','--weights','','--cfg',str(base/'model.yaml'),
              '--data',str(base/'data.yaml'),'--hyp',str(root/'data/hyps/hyp.scratch.yaml'),
              '--epochs','1','--batch-size','2','--img-size','64','--device','cpu','--workers','0',
              '--variant','spl','--conf_thres','0.0001','--score_floor','-1',
              '--consistency_loss','--noautoanchor','--max_batches','1','--project',str(base/'runs'),'--name','smoke']
    print('Synthetic trainer directory:',base,flush=True)
    runpy.run_path(str(root/'ssda_yolov5_train_revision.py'),run_name='__main__')
    assert (base/'runs/smoke/revision_metrics.json').is_file()
    assert (base/'runs/smoke/weights/teacher_epoch_0001.pt').is_file()
    print('PASS: synthetic training, validation and diagnostic checkpoint export.')

if __name__=='__main__':main()
