"""Generate an inspectable revision trainer; never edit the original trainer."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
src = (ROOT/'ssda_yolov5_train_ASA.py').read_text(encoding='utf-8')


def replace(old,new):
    global src
    count = src.count(old)
    if count != 1:
        raise RuntimeError('Expected one source match, got %s: %s' % (count,old[:100]))
    src = src.replace(old,new,1)


replace('from utils.daat import DAAT, create_daat',
        'from revision_support import (seed_everything, create_revision_loader, install_assignment_fixes, '
        'copy_teacher_buffers, RunAudit, save_teacher_snapshot)\nfrom utils.daat import DAAT, create_daat')
replace('    init_seeds(1 + RANK)',
        '    seed_everything(opt.seed)\n    install_assignment_fixes(opt.topk)')
replace('    # Dataset\n',
        '    model_teacher.load_state_dict(model_student.state_dict())\n\n    # Dataset\n')
start = src.index('    # Trainloader\n')
end = src.index('    mlc = ',start)
src = src[:start]+'''    # Paired source images, boxes and segmentation masks share all spatial transforms.
    dataloader_sr, dataset_sr = create_revision_loader(
        train_path_source_real, train_path_source_fake, imgsz, batch_size, hyp,
        opt.seed + 101, workers, masks=train_path_source_seg_real, nc=nc,
        nseg=model_student.yaml['n_segcls'])
    # Target labels and target segmentation masks are never read during training.
    dataloader_tr, dataset_tr = create_revision_loader(
        train_path_target_real, train_path_target_fake, imgsz, batch_size, hyp,
        opt.seed + 202, workers)

'''+src[end:]
replace("labels_to_class_weights(dataset_tr.labels, nc)","labels_to_class_weights(dataset_sr.labels, nc)")
replace('nb = (len(dataloader_sr) + len(\n        dataloader_tr)) // 2',
        'nb = max(1, (len(dataloader_sr) + len(dataloader_tr)) // 2)\n    if opt.max_batches > 0: nb = min(nb, opt.max_batches)')
replace('        warmup_batches=opt.daat_warmup_batches\n',
        '        warmup_batches=opt.daat_warmup_batches,\n        conf_filter_min=None if opt.score_floor < 0 else opt.score_floor\n')
replace('    for epoch in range(start_epoch, epochs):',
        '    audit = RunAudit(save_dir, opt, device)\n    for epoch in range(start_epoch, epochs):')
replace('        model_student.train()\n', '        audit.begin_epoch()\n        model_student.train()\n')
src = src.replace('            dataloader_tr_seg.sampler.set_epoch(epoch)\n','').replace('            dataloader_sr_seg.sampler.set_epoch(epoch)\n','')
src = src.replace('        data_iter_sr_seg = iter(dataloader_sr_seg)\n','').replace('        data_iter_tr_seg = iter(dataloader_tr_seg)\n','')
replace('            # start batch -------------------------------------------------------',
        '            audit.begin()\n            # start batch -------------------------------------------------------')
src = src.replace('imgs_sr, imgs_sf, targets_sr, paths_sr, paths_sf, _ = next(data_iter_sr)',
                  'imgs_sr, imgs_sf, targets_sr, paths_sr, paths_sf, _, targets_sr_seg = next(data_iter_sr)')
src = src.replace('imgs_tr, imgs_tf, targets_tr, paths_tr, paths_tf, _ = next(data_iter_tr)',
                  'imgs_tr, imgs_tf, targets_tr, paths_tr, paths_tf, _, _ = next(data_iter_tr)')
# Iterator exhaustion is the only recoverable dataloader condition.
src = src.replace('            except:\n                data_iter_sr', '            except StopIteration:\n                data_iter_sr')
src = src.replace('            except:\n                data_iter_tr', '            except StopIteration:\n                data_iter_tr')
start = src.index('            # for segmentation data, source and target\n')
end = src.index('            # Warmup\n',start)
src = src[:start]+src[end:]
src = src.replace('targets_sr.to(device), use_asa=opt.use_asa)', 'targets_sr.to(device))')
replace('                    if total_pseudo_labels > 0:\n                        training_logger.log_pseudo_label_count(total_pseudo_labels)',
        '                    training_logger.log_pseudo_label_count(total_pseudo_labels / len(imgs_tf))')
start = src.index('            try:\n                # Backward\n')
end = src.index('            # Print\n',start)
src = src[:start]+'''            if not torch.isfinite(loss).all():
                raise FloatingPointError('Non-finite loss at epoch=%s batch=%s' % (epoch, i))
            scaler.scale(loss).backward()
            if ni - last_opt_step >= accumulate:
                old_scale = scaler.get_scale()
                scaler.step(student_optimizer)
                scaler.update()
                student_optimizer.zero_grad()
                if scaler.get_scale() >= old_scale:  # AMP did not skip the update
                    if ema:
                        ema.update(model_student)
                    teacher_optimizer.step()
                    copy_teacher_buffers(model_teacher, model_student)
                    audit.steps += 1
                last_opt_step = ni
            audit.end()

'''+src[end:]
replace('            training_logger.end_epoch(epoch)',
        '            training_logger.end_epoch(epoch)\n            audit.record_epoch(epoch, results)\n'
        '            save_teacher_snapshot(save_dir, epoch, epochs, model_teacher, daat, opt)')
replace("    parser.add_argument('--epochs', type=int, default=200)",
        "    parser.add_argument('--seed', type=int, default=0)\n"
        "    parser.add_argument('--variant', choices=['baseline','daat','dpla','spl'], default='baseline')\n"
        "    parser.add_argument('--topk', type=int, default=13)\n"
        "    parser.add_argument('--max_batches', type=int, default=0, help='Smoke tests only; 0 keeps the full budget')\n"
        "    parser.add_argument('--score_floor', type=float, default=0.5, help='DAAT histogram floor; -1 uses all finite [0,1] scores')\n"
        "    parser.add_argument('--epochs', type=int, default=200)")
replace('def main(opt):\n', '''def main(opt):
    if RANK != -1 or WORLD_SIZE != 1:
        raise ValueError('This revision entry is validated for single-GPU training only')
    if opt.resume or opt.student_weight != 'None' or opt.teacher_weight != 'None':
        raise ValueError('Use fresh, uninterrupted runs; exact resume is not implemented in this revision entry')
    if opt.multi_scale or opt.quad or opt.image_weights or opt.rect:
        raise ValueError('Keep multi_scale/quad/image_weights/rect disabled in the controlled protocol')
    if not opt.noautoanchor:
        raise ValueError('Use --noautoanchor to keep fixed, identical anchors without target GT')
    if not opt.consistency_loss:
        raise ValueError('Use --consistency_loss to retain source segmentation and mask objectives')
    if opt.notest or opt.nosave:
        raise ValueError('Controlled runs require validation and checkpoints')
    if opt.topk < 1 or opt.seed < 0:
        raise ValueError('topk must be positive and seed nonnegative')
    opt.use_daat = opt.variant in ('daat','spl')
    opt.use_asa = opt.variant in ('dpla','spl')
''')
replace('        check_git_status()','        # Original git/network status check omitted in the revision entry.')
replace("        check_requirements(exclude=['thop'])","        # Install dependencies explicitly; no automatic pip installs during training.")
# Fresh runs must not inherit the source checkpoint optimiser or epoch number.
replace("        if 'optimizer' in ckpt and ckpt['optimizer'] is not None:",
        "        if resume and 'optimizer' in ckpt and ckpt['optimizer'] is not None:")
replace("        if ema and ckpt.get('ema'):","        if resume and ema and ckpt.get('ema'):")
replace("        start_epoch = ckpt['epoch'] + 1", "        start_epoch = ckpt.get('epoch', -1) + 1 if resume else 0")
ast.parse(src)
output = ROOT/'ssda_yolov5_train_revision.py'
output.write_text('# Generated by revision_make_entry.py; original trainer preserved.\n'+src,encoding='utf-8')
print(output)
