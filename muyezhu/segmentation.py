import os
import numpy as np
import tensorflow as tf
import cv2
from dataset import PatchExtractor
from encoder_decoder import BasicED


def basiced_seg():
    project_dir = '/media/muyezhu/Dima/project_files/deep_learning' \
                   '/csci599_project'
    models_dir = os.path.join(project_dir, 'model')
    model_path = os.path.join(models_dir, 'basiced_171121_0750-best/basiced.ckpt')
    seg_out_dir = os.path.join(project_dir, 'segmentation')
    if not os.path.isdir(seg_out_dir):
        os.makedirs(seg_out_dir)
    N, H, W = 8, 256, 256
    ex = PatchExtractor(N, H, W, z_start=50, z_end=51)
    config = tf.ConfigProto(device_count={'GPU': 0})
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.Session(config=config) as sess:
        ed = BasicED(8, 256, 256, n_labels=3)
        ed.restore_model(sess, model_path)
        n = N
        while n > 0:
            n, patches = ex.next_batch()
            if n > 0:
                seg_imgs = ed.segment_imgs(sess, patches)
                for i in range(0, n):
                    fullimg_name = ex.img_names[ex.patches_xyz[i][2]] \
                                     .replace('.tif', '')
                    xyz_str = '{}_x{}_y{}' \
                              .format(fullimg_name,
                                      ex.patches_xyz[i][0],
                                      ex.patches_xyz[i][1])
                    ori_img = np.squeeze(patches[i, ...])
                    ori_name = '{}.tif'.format(xyz_str)
                    seg_img = np.squeeze(seg_imgs[i, ...])
                    seg_name = '{}_seg.tif'.format(xyz_str)
                    print(seg_name)
                    out_dir = os.path.join(seg_out_dir, fullimg_name)
                    if not os.path.isdir(out_dir):
                        os.makedirs(out_dir)
                    cv2.imwrite(os.path.join(out_dir, ori_name), ori_img)
                    cv2.imwrite(os.path.join(out_dir, seg_name), seg_img)


if __name__ == '__main__':
    basiced_seg()