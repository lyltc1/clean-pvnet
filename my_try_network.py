''' python my_try_network.py --cfg_file configs/epnet_linemod.yaml cls_type cat '''
from lib.config import cfg
from lib.networks import make_network
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_network
import tqdm
import torch
import time

cfg.train.num_workers = 0
cfg.train.batch_size = 4

print('********************** cfg ***********************')
print(cfg)
print('**************************************************')

network = make_network(cfg).cuda()
load_network(network, cfg.model_dir, resume=True)
network.eval()

data_loader = make_data_loader(cfg, is_train=False)
total_time = 0
for batch in tqdm.tqdm(data_loader):
    for k in batch:
        if k != 'meta':
            batch[k] = batch[k].cuda()
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        network(batch['inp'], batch)
        torch.cuda.synchronize()
        total_time += time.time() - start
print(total_time / len(data_loader))
