import matplotlib.pyplot as plt
from lib.utils import img_utils
from lib.utils.pvnet import pvnet_config

mean = pvnet_config.mean
std = pvnet_config.std


def visualize_ann(img, kpt_2d, mask, savefig=False):
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax1.plot(kpt_2d[:, 0], kpt_2d[:, 1], '.')
    ax2.imshow(mask)
    if savefig:
        plt.savefig('test.jpg')
    else:
        plt.show()


def visualize_linemod_ann(img, kpt_2d, mask, amodal_mask, savefig=False):
    img = img_utils.unnormalize_img(img, mean, std, False).permute(1, 2, 0)
    plt.figure('visualize_linemod_ann')
    plt.subplot(131)
    plt.imshow(img)
    plt.plot(kpt_2d[:, 0], kpt_2d[:, 1], '.')
    plt.title('image & keypoints')
    plt.subplot(132)
    plt.imshow(mask)
    plt.title('mask')
    plt.subplot(133)
    plt.imshow(amodal_mask)
    plt.title('amodal_mask')
    if savefig:
        plt.savefig('test.jpg')
    else:
        plt.show()
    plt.close('visualize_linemod_ann')
