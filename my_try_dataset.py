''' python my_try_dataset.py --cfg_file configs/epnet_linemod.yaml cls_type cat '''
from lib.config import cfg
from lib.datasets import make_data_loader
import tqdm
from lib.visualizers import make_visualizer

cfg.train.num_workers = 0
cfg.train.batch_size = 4

print('********************** cfg ***********************')
print(cfg)
print('**************************************************')

data_loader = make_data_loader(cfg, is_train=True)
visualizer = make_visualizer(cfg)
for batch in tqdm.tqdm(data_loader):
    for k in batch:
        if k != 'meta':
            batch[k] = batch[k].cuda()
    output = []
    # use ‘visualize_utils.visualize_linemod_ann(torch.tensor(inp), kpt_2d, mask, amodal_mask, True)’
    # in last line of function 'def __getitem__(self, index_tuple):' in lib/datasets/linemod/epnet.py

