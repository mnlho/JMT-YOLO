"""Repeated single-GPU forward and host-image-to-NMS timing, without disk I/O."""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',nargs='+',required=True);p.add_argument('--images',required=True)
    p.add_argument('--output',default='runs/revision/inference_benchmark.json')
    p.add_argument('--device',default='0');p.add_argument('--half',action='store_true')
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--img-size',type=int,default=960)
    p.add_argument('--warmup',type=int,default=100);p.add_argument('--iterations',type=int,default=1000);p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args()
    import torch,cv2
    from models.experimental import attempt_load
    from revision_support import image_map
    from utils.datasets import letterbox
    from utils.general import non_max_suppression
    from utils.torch_utils import select_device
    device=select_device(a.device)
    if device.type!='cuda':raise ValueError('Formal benchmark requires CUDA on the common GPU')
    if min(a.batch_size,a.img_size,a.iterations,a.repeats)<1 or a.warmup<0:raise ValueError('Invalid benchmark counts')
    paths=list(image_map(a.images).values())
    # Keep the same finite set of decoded native images in host RAM for both models.
    paths=paths[:min(len(paths),100)]
    native=[cv2.imread(str(p)) for p in paths]
    if any(im is None for im in native):raise ValueError('Unreadable image')
    def preprocess(images):
        ar=np.stack([letterbox(im,a.img_size,auto=False)[0][:,:,::-1].transpose(2,0,1) for im in images])
        x=torch.from_numpy(np.ascontiguousarray(ar)).to(device)
        return (x.half() if a.half else x.float())/255
    inputs=preprocess([native[i%len(native)] for i in range(a.batch_size)])
    results=[]
    with torch.no_grad():
        for weight in a.weights:
            model=attempt_load(weight,map_location=device).eval()
            model=model.half() if a.half else model.float()
            for scope in ['forward','host_array_to_NMS']:
                measurements=[];repeat_means=[]
                for repeat in range(a.repeats):
                    for i in range(a.warmup+a.iterations):
                        torch.cuda.synchronize(device);start=time.perf_counter()
                        x=inputs if scope=='forward' else preprocess([native[(i*a.batch_size+j)%len(native)] for j in range(a.batch_size)])
                        out=model(x)
                        if scope!='forward':non_max_suppression(out[0][0],.25,.45,max_det=300)
                        torch.cuda.synchronize(device);elapsed=(time.perf_counter()-start)*1000
                        if i>=a.warmup:measurements.append(elapsed)
                    repeat_means.append(float(np.mean(measurements[-a.iterations:])))
                results.append({'weights':weight,'scope':scope,'batch_ms_mean':float(np.mean(measurements)),
                                'batch_ms_p50':float(np.percentile(measurements,50)),'batch_ms_p95':float(np.percentile(measurements,95)),
                                'repeat_mean_ms':repeat_means,'repeat_mean_ms_std':float(np.std(repeat_means,ddof=1)) if len(repeat_means)>1 else None,
                                'images_per_second':a.batch_size*1000/float(np.mean(measurements))})
            del model
            torch.cuda.empty_cache()
    output={'protocol':vars(a),'GPU':torch.cuda.get_device_name(device),'torch':torch.__version__,'cuda':torch.version.cuda,
            'note':'Forward input already on GPU. Host-array-to-NMS includes resize/letterbox/normalisation/transfer/forward/NMS; excludes disk read, decoding, plotting and saving. Both models use the same decoded native images. Warmup repeated for each timing block.',
            'results':results}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(output,indent=2),encoding='utf-8');print(path)


if __name__=='__main__':main()
