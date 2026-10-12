"""Prepare configs, validate data, and launch sequential paired-seed training."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
TASKS = {'c2f':('cityscapes_csfoggy_VOC.yaml',200),
         'v2c':('pascalvoc0712_clipart1k_VOC.yaml',200),
         'k2c':('KITTI_CityScapes_VOC.yaml',50)}


def prepare(args):
    original = yaml.safe_load((ROOT/'data/yamls_sda'/TASKS[args.task][0]).read_text(encoding='utf-8'))
    old_root = original['path'].rstrip('/')
    new_root = Path(args.data_root).expanduser().resolve()
    original['path'] = str(new_root)
    for key,value in original.items():
        if key.startswith(('train_','test_')):
            values = value if isinstance(value,list) else [value]
            original[key] = [str(new_root / (x[len(old_root):].lstrip('/') if x.startswith(old_root+'/') else x)) for x in values]
    out = ROOT/'revision_configs'; out.mkdir(exist_ok=True)
    (out/(args.task+'.yaml')).write_text(yaml.safe_dump(original,allow_unicode=True,sort_keys=False),encoding='utf-8')
    model = yaml.safe_load((ROOT/'models/yolov5l_city_seg.yaml').read_text())
    model['nc'],model['n_segcls'] = original['nc'],original['n_segcls']
    (out/('model_'+args.task+'.yaml')).write_text(yaml.safe_dump(model,sort_keys=False),encoding='utf-8')
    print('Created',out/(args.task+'.yaml'))
    print('Edit ALL image paths to your real directories, including both VOC2007 and VOC2012 fake images.')
    print('Model classes preserved from the provided data YAML; verify detection IDs/background and mask IDs.')


def resolve_data(path):
    d = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    root = Path(d.get('path','')).expanduser()
    for k,v in d.items():
        if k.startswith(('train_','test_')):
            values = v if isinstance(v,list) else [v]
            d[k] = [str(Path(x) if Path(x).is_absolute() else root/x) for x in values]
    return d


def check(args):
    import torch
    from revision_support import PairedDataset, image_map, label_path
    weights = Path(args.weights)
    if not weights.is_file(): raise FileNotFoundError('Find the compatible YOLOv5-L pretrained checkpoint first: '+str(weights))
    d = resolve_data(args.data)
    hyp = yaml.safe_load((ROOT/'data/hyps/hyp.scratch.yaml').read_text())
    sr = PairedDataset(d['train_source_real'],d['train_source_fake'],320,hyp,
                       d['train_source_seg_real'],d['nc'],d['n_segcls'])
    tr = PairedDataset(d['train_target_real'],d['train_target_fake'],320,hyp)
    validation = image_map(d['test_target_real'])
    target_ids = set(tr.ids)
    overlap = target_ids.intersection(validation)
    if overlap: raise ValueError('Target train/val overlap by image ID: '+str(sorted(overlap)[:5]))
    source_ids = set(sr.ids)
    if source_ids.intersection(validation):
        raise ValueError('Source training and target validation share image IDs; inspect split leakage before training')
    for p in validation.values():
        if not label_path(p,'.txt').is_file(): raise FileNotFoundError(label_path(p,'.txt'))
    for ds in [sr,tr]:
        for i in range(len(ds)): ds.load(i)  # full shape/mask-ID validation, no target GT reads
        ds[0]  # smoke-test synchronized mosaic/mask/box transforms
    rows = {'source':sr.img_files,'source_fake':sr.img_files_fake,
            'target':tr.img_files,'target_fake':tr.img_files_fake,
            'validation':[str(p) for p in validation.values()]}
    digest = hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    output = {'counts':{k:len(v) for k,v in rows.items()},'file_list_sha256':digest,'files':rows,
              'note':'Fingerprint is of file lists, not image contents. Target labels were not opened.'}
    out = Path(args.data).with_suffix('.manifest.json')
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output['counts'],ensure_ascii=False))
    print('Data check PASSED; manifest:',out)
    print('CUDA:',torch.cuda.is_available(),'torch:',torch.__version__)
    if not torch.cuda.is_available(): print('Data checks passed, but formal training requires the configured CUDA environment.')


def train(args):
    if not (ROOT/'ssda_yolov5_train_revision.py').is_file():
        raise FileNotFoundError('Run python revision_make_entry.py first')
    tasks = TASKS.keys() if args.task=='all' else [args.task]
    for task in tasks:
        data = ROOT/'revision_configs'/(task+'.yaml')
        cfg = ROOT/'revision_configs'/('model_'+task+'.yaml')
        if not data.is_file() or not cfg.is_file(): raise FileNotFoundError('Run prepare for '+task)
        if args.execute and not data.with_suffix('.manifest.json').is_file():
            raise FileNotFoundError('Run check successfully before training '+task)
        variants = args.variants
        seeds = args.seeds
        if args.smoke: seeds,variants = [seeds[0]],[variants[0]]
        for variant in variants:
            for seed in seeds:
                project = ROOT/'runs'/('revision_smoke' if args.smoke else 'revision')/task
                name = '%s_seed%d'%(variant,seed)
                if args.execute and (project/name).exists():
                    raise FileExistsError('Output exists; preserve the old run and choose a different project/name: '+str(project/name))
                command = [sys.executable,str(ROOT/'ssda_yolov5_train_revision.py'),
                           '--weights',str(Path(args.weights).resolve()),'--cfg',str(cfg),'--data',str(data),
                           '--hyp',str(ROOT/'data/hyps/hyp.scratch.yaml'),
                           '--epochs',str(1 if args.smoke else TASKS[task][1]),
                           '--batch-size','2','--img-size',str(320 if args.smoke else 960),
                           '--device',args.device,'--workers',str(args.workers),'--seed',str(seed),
                           '--variant',variant,'--teacher_alpha','0.99','--conf_thres',str(args.threshold),
                           '--iou_thres','0.3','--max_gt_boxes','20','--lambda_weight','0.005',
                           '--iou_factor','2.0','--dist_factor','0.1','--topk','13',
                           '--daat_ema_momentum','0.99','--daat_warmup_batches','10',
                           '--score_floor',str(args.score_floor),'--consistency_loss','--noautoanchor',
                           '--project',str(project),'--name',name]
                if args.smoke: command += ['--max_batches','5']
                print(subprocess.list2cmdline(command),flush=True)
                if args.execute:
                    if not Path(args.weights).is_file(): raise FileNotFoundError(args.weights)
                    env = dict(os.environ,WANDB_MODE='disabled',PYTHONHASHSEED=str(seed),CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    subprocess.run(command,cwd=str(ROOT),env=env,check=True)
    if not args.execute: print('Preview only. Add --execute after reviewing paths and the common protocol.')


def stats(args):
    import numpy as np
    from scipy import stats as scipy_stats
    import csv
    rows = []
    for p in Path(args.runs).glob('*/*/revision_metrics.json'):
        d = json.loads(p.read_text(encoding='utf-8'))
        if not d['epochs'] or d['epochs'][-1]['epoch'] != d['options']['epochs']:
            print('Skipping unfinished run: '+str(p.parent))
            continue
        m = d['best']['metrics']
        profile = json.loads((p.parent/'training_profile.json').read_text())
        rows.append({'task':p.parent.parent.name,'variant':d['variant'],'seed':d['seed'],
                     'selected_epoch':d['best']['epoch'],'AP50':m[2]*100,'AP75':m[3]*100,'AP50_95':m[4]*100,
                     'training_hours':profile['training_iteration_seconds_total']/3600,
                     'peak_allocated_GiB':profile['peak_allocated_GiB'],'peak_reserved_GiB':profile['peak_reserved_GiB']})
    if not rows: raise FileNotFoundError('No completed revision_metrics.json runs in '+args.runs)
    out = Path(args.runs)/'per_seed_results.csv'
    with out.open('w',newline='',encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary = []
    for task in sorted({r['task'] for r in rows}):
        subset = [r for r in rows if r['task']==task]
        for variant in sorted({r['variant'] for r in subset}):
            a = np.asarray([r['AP50'] for r in subset if r['variant']==variant])
            summary.append({'task':task,'variant':variant,'n':len(a),'mean':float(a.mean()),
                            'sample_std':float(a.std(ddof=1)) if len(a)>1 else None})
        baseline = {r['seed']:r['AP50'] for r in subset if r['variant']=='baseline'}
        spl = {r['seed']:r['AP50'] for r in subset if r['variant']=='spl'}
        if baseline and spl:
            if baseline.keys()!=spl.keys(): raise ValueError('Unpaired/incomplete seed sets for '+task)
            ids = sorted(baseline)
            delta = np.asarray([spl[s]-baseline[s] for s in ids])
            if len(delta)<2: continue
            se = delta.std(ddof=1)/np.sqrt(len(delta))
            width = scipy_stats.t.ppf(.975,len(delta)-1)*se
            t = scipy_stats.ttest_rel([spl[s] for s in ids],[baseline[s] for s in ids])
            summary.append({'task':task,'comparison':'spl-baseline','n':len(ids),'delta_mean':float(delta.mean()),
                            'delta_CI95':[float(delta.mean()-width),float(delta.mean()+width)],
                            'paired_t_pvalue':float(t.pvalue) if np.isfinite(t.pvalue) else None,
                            'note':'Two-sided paired t-test; interpret assumptions and adjust multiple comparisons separately.'})
    target = Path(args.runs)/'summary_statistics.json'
    target.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(out); print(target);print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='action',required=True)
    a = sub.add_parser('prepare');a.add_argument('--task',choices=TASKS,required=True);a.add_argument('--data-root',required=True);a.set_defaults(func=prepare)
    a = sub.add_parser('check');a.add_argument('--data',required=True);a.add_argument('--weights',required=True);a.set_defaults(func=check)
    a = sub.add_parser('train');a.add_argument('--task',choices=list(TASKS)+['all'],required=True)
    a.add_argument('--weights',required=True);a.add_argument('--variants',nargs='+',choices=['baseline','daat','dpla','spl'],default=['baseline','spl'])
    a.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4]);a.add_argument('--device',default='0');a.add_argument('--workers',type=int,default=4)
    a.add_argument('--threshold',type=float,default=.5);a.add_argument('--score-floor',type=float,default=.5)
    a.add_argument('--smoke',action='store_true');a.add_argument('--execute',action='store_true');a.set_defaults(func=train)
    a = sub.add_parser('stats');a.add_argument('--runs',default=str(ROOT/'runs/revision'));a.set_defaults(func=stats)
    args = p.parse_args();args.func(args)


if __name__=='__main__': main()
