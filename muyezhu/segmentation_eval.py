import os
import re
import numpy as np
import cv2

"""
metrics based on:
https://arxiv.org/pdf/1411.4038.pdf
modified from:
https://github.com/martinkersner/py_img_seg_eval
"""


def pixel_accuracy(eval_segm, gt_segm):
    """
    sum_i(n_ii) / sum_i(t_i)
    """
    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if sum_t_i == 0:
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm):
    """
    (1/n_cl) sum_i(n_ii/t_i)
    """
    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if t_i != 0:
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_


def mean_IU(eval_segm, gt_segm):
    """
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    """

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_


def frequency_weighted_IU(eval_segm, gt_segm):
    """
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    """

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_

"""
Auxiliary functions used during evaluation.
"""


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if h_e != h_g or w_e != w_g:
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


"""
high level image segmentation evaluation API
"""


def write_basiced_segm_results(group='training'):
    if group != 'training' and group != 'holdout':
        raise ValueError('group should be either training or holdout')
    eval_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data', 'eval', group)
    project_dir = '/media/muyezhu/Dima/project_files/deep_learning' \
                   '/csci599_project'
    seg_out_dir = os.path.join(project_dir, 'segmentation')
    xyz_pattern = re.compile('Z([0-9]+)_x([0-9]+)_y([0-9]+)')
    img = np.zeros((2048, 2048), dtype=np.uint8)
    for trad_name in os.listdir(os.path.join(eval_dir, 'traditional')):
        m = re.match(xyz_pattern, trad_name)
        if m is None:
            raise ValueError('can not find xyz pattern in {}'.format(trad_name))
        z = m.group(1)
        seg_dir = os.path.join(seg_out_dir, 'Z{}'.format(z))
        print(seg_out_dir)
        assert os.path.isdir(seg_out_dir)
        x = int(m.group(2))
        y = int(m.group(3))
        for i in range(0, 8):
            for j in range(0, 8):
                seg_name = 'Z{}_x{}_y{}_seg.tif'.format(z, x + 256 * j,
                                                        y + 256 * i)
                seg_img = cv2.imread(os.path.join(seg_dir, seg_name), 0)
                img[256 * i: 256 * (i + 1), 256 * j: 256 * (j + 1)] = seg_img
        seg_name = 'Z{}_x{}_y{}_seg.tif'.format(z, x, y)
        cv2.imwrite(os.path.join(eval_dir, 'basic_ed', seg_name), img)


def eval_segm(group='training'):
    if group != 'training' and group != 'holdout':
        raise ValueError('group should be either training or holdout')
    write_basiced_segm_results(group=group)
    eval_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data', 'eval', group)
    xyz_pattern = re.compile('Z[0-9]+_x[0-9]+_y[0-9]+')
    gt_dir = os.path.join(eval_dir, 'ground_truth')
    trad_dir = os.path.join(eval_dir, 'traditional')
    basiced_dir = os.path.join(eval_dir, 'basic_ed')
    trad_eval = np.zeros((len(os.listdir(trad_dir)), 4))
    seg_eval = np.zeros((len(os.listdir(trad_dir)), 4))
    if os.path.isfile(os.path.join(eval_dir, 'eval_{}'.format(group))):
        os.remove(os.path.join(eval_dir, 'eval_{}'.format(group)))
    for i, trad_name in enumerate(os.listdir(trad_dir)):
        m = re.match(xyz_pattern, trad_name)
        if m is None:
            continue
        xyz = m.group()
        gt = cv2.imread(os.path.join(gt_dir, '{}_manual.tif'.format(xyz)), 0)
        trad = cv2.imread(os.path.join(trad_dir, '{}_anno.tif'.format(xyz)), 0)
        seg = cv2.imread(os.path.join(basiced_dir, '{}_seg.tif'.format(xyz)), 0)
        trad_eval[i, 0] = pixel_accuracy(trad, gt)
        trad_eval[i, 1] = mean_accuracy(trad, gt)
        trad_eval[i, 2] = mean_IU(trad, gt)
        trad_eval[i, 3] = frequency_weighted_IU(trad, gt)
        seg_eval[i, 0] = pixel_accuracy(seg, gt)
        seg_eval[i, 1] = mean_accuracy(seg, gt)
        seg_eval[i, 2] = mean_IU(seg, gt)
        seg_eval[i, 3] = frequency_weighted_IU(seg, gt)
        with open(os.path.join(eval_dir, 'eval_{}'.format(group)), 'a+') as f:
            f.write('{}.tif:\n'.format(xyz))
            f.write('traditional method:\n')
            f.write('pixel accuracy = {}\n'.format(trad_eval[i, 0]))
            f.write('mean accuracy = {}\n'.format(trad_eval[i, 1]))
            f.write('mean IU = {}\n'.format(trad_eval[i, 2]))
            f.write('frequency weighted IU = {}\n'.format(trad_eval[i, 3]))
            f.write('encoder decoder:\n')
            f.write('pixel accuracy = {}\n'.format(seg_eval[i, 0]))
            f.write('mean accuracy = {}\n'.format(seg_eval[i, 1]))
            f.write('mean IU = {}\n'.format(seg_eval[i, 2]))
            f.write('frequency weighted IU = {}\n'.format(seg_eval[i, 3]))
    with open(os.path.join(eval_dir, 'eval_{}'.format(group)), 'a+') as f:
        f.write('overall:\n')
        f.write('traditional method:\n')
        f.write('pixel accuracy = {}\n'.format(np.mean(trad_eval[:, 0])))
        f.write('mean accuracy = {}\n'.format(np.mean(trad_eval[:, 1])))
        f.write('mean IU = {}\n'.format(np.mean(trad_eval[:, 2])))
        f.write('frequency weighted IU = {}\n'.format(np.mean(trad_eval[:, 3])))
        f.write('encoder decoder:\n')
        f.write('pixel accuracy = {}\n'.format(np.mean(seg_eval[:, 0])))
        f.write('mean accuracy = {}\n'.format(np.mean(seg_eval[:, 1])))
        f.write('mean IU = {}\n'.format(np.mean(seg_eval[:, 2])))
        f.write('frequency weighted IU = {}\n'.format(np.mean(seg_eval[:, 3])))


if __name__ == '__main__':
    eval_segm()
