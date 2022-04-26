import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image
from lib.utils.pvnet import pvnet_data_utils, pvnet_linemod_utils, visualize_utils
from lib.utils.linemod import linemod_config
from lib.datasets.augmentation import crop_or_padding_to_fixed_size, rotate_instance, crop_resize_instance_v1
import random
import torch
from lib.config import cfg


class Dataset(data.Dataset):

    def __init__(self, ann_file, data_root, split, transforms=None):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self._transforms = transforms
        self.cfg = cfg

        # init variables used for occlude_augment
        self.other_object_names = []
        self.init_occlude_augment()

    def init_occlude_augment(self):
        for object_name in linemod_config.linemod_cls_names:
            if object_name in self.cfg['cls_type']:
                continue
            self.other_object_names.append(object_name)

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        inp = Image.open(path)
        kpt_2d = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)

        cls_idx = linemod_config.linemod_cls_names.index(anno['cls']) + 1
        mask = pvnet_data_utils.read_linemod_mask(anno['mask_path'], anno['type'], cls_idx)
        amodal_mask = pvnet_data_utils.read_linemod_amodal_mask(anno['amodal_mask_path'], anno['type'])
        return inp, kpt_2d, mask, amodal_mask

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        img_id = self.img_ids[index]

        img, kpt_2d, mask, amodal_mask = self.read_data(img_id)
        if self.split == 'train':
            img = np.asarray(img).astype(np.uint8)
            inp, kpt_2d, mask, amodal_mask = self.augment(img, mask, amodal_mask, kpt_2d, height, width)

        else:
            inp = img

        if self._transforms is not None:
            inp, kpt_2d, mask, amodal_mask = self._transforms(inp, kpt_2d, mask, amodal_mask)

        vertex = pvnet_data_utils.compute_vertex(mask, kpt_2d).transpose(2, 0, 1)
        ret = {'inp': inp, 'mask': mask.astype(np.uint8), 'amodal_mask': amodal_mask.astype(np.uint8),
               'vertex': vertex, 'img_id': img_id, 'meta': {}}
        visualize_utils.visualize_linemod_ann(torch.tensor(inp), kpt_2d, mask, amodal_mask, False)

        return ret

    def __len__(self):
        return len(self.img_ids)

    def augment(self, img, mask, amodal_mask, kpt_2d, height, width):
        # add one column to kpt_2d for convenience to calculate
        hcoords = np.concatenate((kpt_2d, np.ones((kpt_2d.shape[0], 1))), axis=-1)
        img = np.asarray(img).astype(np.uint8)
        foreground = np.sum(mask)

        if foreground > 0:
            img, mask = self.occlude_with_another_object(img, mask)
            img, mask, amodal_mask, hcoords = rotate_instance(img, mask, amodal_mask, hcoords,
                                                              self.cfg.train.rotate_min, self.cfg.train.rotate_max)
            img, mask, amodal_mask, hcoords = crop_resize_instance_v1(img, mask, amodal_mask, hcoords, height, width,
                                                                      self.cfg.train.overlap_ratio,
                                                                      self.cfg.train.resize_ratio_min,
                                                                      self.cfg.train.resize_ratio_max)
        else:
            img, mask, amodal_mask = crop_or_padding_to_fixed_size(img, mask, amodal_mask, height, width)
        kpt_2d = hcoords[:, :2]

        return img, kpt_2d, mask, amodal_mask

    def occlude_with_another_object(self, image, mask):
        orig_image, orig_mask = image.copy(), mask.copy()
        try:
            while True:
                image = orig_image.copy()
                mask = orig_mask.copy()
                other_object_name = random.choice(self.other_object_names)
                other_data_root = os.path.abspath(os.path.join(self.data_root, "../..", other_object_name))
                other_image_dir = os.path.join(other_data_root, "JPEGImages")
                other_length = len(list(filter(lambda x: x.endswith('jpg'), os.listdir(other_image_dir))))
                other_idx = random.randrange(other_length)
                other_image_path = os.path.join(other_image_dir, '{:06}.jpg'.format(other_idx))
                other_mask_path = os.path.join(other_data_root, 'mask', '{:04d}.png'.format(other_idx))
                other_image = Image.open(other_image_path)
                other_image = np.asarray(other_image).astype(np.uint8)
                other_mask = np.array(Image.open(other_mask_path))
                if len(other_mask.shape) == 3:
                    other_mask = (other_mask[..., 0] != 0).astype(np.uint8)
                else:
                    other_mask = (other_mask != 0).astype(np.uint8)
                other_mask = (other_mask != 0).astype(np.uint8)

                ys, xs = np.nonzero(mask)
                ymin, ymax = np.min(ys), np.max(ys)
                xmin, xmax = np.min(xs), np.max(xs)
                other_ys, other_xs = np.nonzero(other_mask)
                other_ymin, other_ymax = np.min(other_ys), np.max(other_ys)
                other_xmin, other_xmax = np.min(other_xs), np.max(other_xs)
                other_mask = other_mask[other_ymin:other_ymax, other_xmin:other_xmax]
                other_image = other_image[other_ymin:other_ymax, other_xmin:other_xmax]

                start_y = np.random.randint(ymin - other_mask.shape[0], ymax + 1)
                end_y = start_y + other_mask.shape[0]
                start_x = np.random.randint(xmin - other_mask.shape[1], xmax + 1)
                end_x = start_x + other_mask.shape[1]
                if start_y < 0:
                    other_mask = other_mask[-start_y:, :]
                    other_image = other_image[-start_y:, :, :]
                    start_y = 0
                if end_y > image.shape[0]:
                    end_y = image.shape[0]
                    other_mask = other_mask[:image.shape[0] - start_y, :]
                    other_image = other_image[:image.shape[0] - start_y, :, :]
                if start_x < 0:
                    other_mask = other_mask[:, -start_x:]
                    other_image = other_image[:, -start_x:, :]
                    start_x = 0
                if end_x > image.shape[1]:
                    end_x = image.shape[1]
                    other_mask = other_mask[:, :image.shape[1] - start_x]
                    other_image = other_image[:, :image.shape[1] - start_x, :]
                other_outline = (other_mask == 0)[:, :, None]
                image[start_y:end_y, start_x:end_x] *= other_outline
                other_image[other_mask == 0] = 0
                image[start_y:end_y, start_x:end_x] += other_image
                mask[start_y:end_y, start_x:end_x] *= (other_mask == 0)
                if mask.sum() >= 150:
                    break
        except:
            return orig_image, orig_mask
        return image, mask
