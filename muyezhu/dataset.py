import os
import re
import numpy as np
import scipy.misc
try:
    import cv2
except ImportError:
    pass

TILE_L = 2048
RAW_IMG_DIR = '/media/muyezhu/Dima/project_files/AKO_TIFF/' \
              'AKO_04_03_P42_X30_01.vsi.Collection/Layer0'
BOUNDARY_WIDTH = 2


class DataLoader:
    """
    load [H, W] sized crops from 2048 * 2048 images
    """
    def __init__(self, mode='train', seg_method='auto', n_class=3):
        self.data_root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data')
        self.mode = None
        self.n_class = n_class
        self.seg_method = seg_method
        self.data_paths = []
        self.label_paths = []
        self.n_img, self.H, self.W = 0, TILE_L, TILE_L
        self.indices = []
        self.valid_zvals = set(range(50, 80))
        self.set_paths(mode=mode, seg_method=self.seg_method)
        self.ind_pos, self.h_pos, self.w_pos, self.rot_pos = 0, 0, 0, 0
        self.data, self.label = None, None

    def set_paths(self, mode='train', seg_method='auto'):
        if not mode == 'train' and not mode == 'test':
            raise ValueError('mode must be either train or test')
        if not seg_method == 'auto' and not seg_method == 'manual':
            raise ValueError('seg method must be either auto or manual')
        if self.mode == mode and self.seg_method == seg_method:
            return
        self.mode = mode
        self.seg_method = seg_method
        self.data_paths.clear()
        self.label_paths.clear()
        if self.seg_method == 'auto' and self.mode == 'train':
            self.set_auto_train_paths()
        if self.seg_method == 'manual' and self.mode == 'train':
            self.set_manual_train_paths()
        if self.mode == 'test':
            self.set_test_paths()
        self.n_img = len(self.data_paths)
        self.indices = np.arange(self.n_img)
        np.random.shuffle(self.indices)

    def set_auto_train_paths(self):
        data_dir = os.path.join(self.data_root, self.mode)
        label_dir = os.path.join(self.data_root, self.mode)
        for crop_folder in os.listdir(data_dir):
            data_crop_dir = os.path.join(data_dir, crop_folder)
            label_crop_dir = os.path.join(label_dir, crop_folder,
                                          'annotate_{}'.format(self.n_class))
            assert os.path.isdir(data_crop_dir)
            assert os.path.isdir(label_crop_dir)
            for img_name in os.listdir(data_crop_dir):
                if img_name.find('.tif') < 0:
                    continue
                zval = self.extract_zval(img_name)
                if int(zval) not in self.valid_zvals:
                    continue
                self.data_paths.append(os.path.join(data_crop_dir, img_name))
            self.data_paths.sort()
            label_paths = [os.path.join(label_crop_dir, img_name)
                           for img_name in os.listdir(label_crop_dir)]
            self.label_paths.extend(label_paths)
            self.label_paths.sort()

    def set_manual_train_paths(self):
        src_dir = os.path.join(self.data_root, self.seg_method)
        data_dir, label_dir = '', ''
        for d in os.listdir(src_dir):
            if d.find('_manual') > 0:
                label_dir = os.path.join(src_dir, d)
            else:
                data_dir = os.path.join(src_dir, d)
        label_dir = os.path.join(label_dir, 'annotate_{}'.format(self.n_class))
        for img_name in os.listdir(label_dir):
            self.data_paths.append(os.path.join(data_dir, img_name.replace('_manual', '')))
            self.label_paths.append(os.path.join(label_dir, img_name))
        self.data_paths.sort()
        self.label_paths.sort()

    def set_test_paths(self):
        data_dir = os.path.join(self.data_root, self.mode)
        for crop_folder in os.listdir(data_dir):
            data_crop_dir = os.path.join(data_dir, crop_folder)
            assert os.path.isdir(data_crop_dir)
            for img_name in os.listdir(data_crop_dir):
                if img_name.find('.tif') < 0:
                    continue
                self.data_paths.append(os.path.join(data_crop_dir, img_name))
            self.data_paths.sort()
            self.n_img = len(self.data_paths)

    def validate_data_storage(self):
        if not len(self.data_paths) == len(self.label_paths):
            raise ValueError('number of data image and label image is not equal')
        for i in range(0, self.n_img):
            data_path = os.path.basename(self.data_paths[i])
            label_path = os.path.basename(self.label_paths[i])
            postfix = '_anno' if self.seg_method == 'auto' else '_manual'
            if not label_path.replace(postfix, '') == data_path:
                raise ValueError('unpaired data and label image name: \n{}\n{}'
                                 .format(data_path, label_path))

    def num_samples(self, w, h):
        if self.seg_method == 'auto':
            return self.n_img * (self.H // h) * (self.W // w)
        else:
            return 4 * self.n_img * (self.H // h) * (self.W // w)

    def extract_zval(self, img_path):
        img_name = os.path.basename(img_path)
        zpattern = 'Z([0-9]+)_'
        m = re.match(zpattern, img_name)
        if m is None:
            raise ValueError('can not find z level pattern in {}'.format(img_name))
        return m.group(1)

    def draw_numbers(self, hn, wn):
        h_start = np.random.randint(0, hn)
        w_start = np.random.randint(0, wn)
        img_ind = np.random.randint(0, self.n_img)
        return tuple([img_ind, h_start, w_start])

    def load_data(self, n, h, w, mode='train', seg_method='auto'):
        self.set_paths(mode=mode, seg_method=seg_method)
        if self.mode == 'train':
            self.validate_data_storage()
        if mode == 'train':
            return self._load_train_data(n, h, w)
        else:
            return self._load_test_data(n, h, w)

    def _load_train_data(self, n, h, w):
        data_batch = np.zeros((n, h, w, 1), dtype=np.uint8)
        label_batch = np.zeros((n, h, w, 1), dtype=np.uint8)
        fill_pos = 0
        while fill_pos < n:
            if self.h_pos == 0 and self.w_pos == 0 and self.rot_pos == 0:
                self.data = scipy.misc.imread(self.data_paths[self.indices[self.ind_pos]])
                self.label = scipy.misc.imread(self.label_paths[self.indices[self.ind_pos]])
            data_sub = self.data[self.h_pos * h: (self.h_pos + 1) * h,
                                 self.w_pos * w: (self.w_pos + 1) * w]
            data_sub = np.expand_dims(data_sub, axis=2)
            label_sub = self.label[self.h_pos * h: (self.h_pos + 1) * h,
                                   self.w_pos * w: (self.w_pos + 1) * w]
            label_sub = np.expand_dims(label_sub, axis=2)
            data_batch[fill_pos, ...] = np.rot90(data_sub, self.rot_pos)
            label_batch[fill_pos, ...] = np.rot90(label_sub, self.rot_pos)
            fill_pos += 1
            self.rot_pos += 1
            if self.rot_pos == 4:
                self.rot_pos = 0
                self.w_pos += 1
                if self.w_pos == self.W // w:
                    self.w_pos = 0
                    self.h_pos += 1
                    if self.h_pos == self.H // h:
                        self.h_pos = 0
                        self.ind_pos += 1
                        if self.ind_pos == self.n_img:
                            self.ind_pos = 0
        return data_batch, label_batch

    def _load_test_data(self, n, h, w):
        data_batch = np.zeros((n, h, w, 1), dtype=np.uint8)
        fill_pos = 0
        while fill_pos < n:
            if self.h_pos == 0 and self.w_pos == 0 and self.rot_pos == 0:
                self.data = scipy.misc.imread(self.data_paths[self.indices[self.ind_pos]])
            data_sub = self.data[self.h_pos * h: (self.h_pos + 1) * h,
                                 self.w_pos * w: (self.w_pos + 1) * w]
            data_sub = np.expand_dims(data_sub, axis=2)
            data_batch[fill_pos, ...] = np.rot90(data_sub, self.rot_pos)
            fill_pos += 1
            self.rot_pos += 1
            if self.rot_pos == 4:
                self.rot_pos = 0
                self.w_pos += 1
                if self.w_pos == self.W // w:
                    self.w_pos = 0
                    self.h_pos += 1
                    if self.h_pos == self.H // h:
                        self.h_pos = 0
                        self.ind_pos += 1
                        if self.ind_pos == self.n_img:
                            self.ind_pos = 0
        return data_batch


class PatchExtractor:
    def __init__(self, N, H, W, z_start=0, z_end=-1):
        """
        extract patches from img_paths[z_start: z_end)
        shape should be (N, H, W, 1)
        Args:
            z_start:
            z_end:

        Returns:

        """
        self.fullsize_dir = '/media/muyezhu/Dima/project_files/AKO_TIFF' \
                            '/AKO_04_03_P42_X30_01.vsi.Collection/Layer0'
        self.N, self.H, self.W = N, H, W
        self.zcur, self.xcur, self.ycur = 0, 0, 0
        self.z_start, self.z_end = z_start, z_end
        self.img_height, self.img_width = 0, 0
        self.img = None
        self.patches = np.zeros((self.N, self.H, self.W, 1), dtype=np.uint8)
        self.patches_xyz = []
        self.img_names = []
        self._build_img_names()

    def _build_img_names(self):
        for name in os.listdir(self.fullsize_dir):
            if not os.path.isfile(os.path.join(self.fullsize_dir, name)) or \
               name.find('.tif') < 0:
                continue
            self.img_names.append(name)
        self.img_names.sort()
        if self.z_end < 0:
            self.z_end = len(self.img_names)
        if self.z_start < 0:
            self.z_start = 0
        if self.z_start >= self.z_end:
            raise ValueError('empty img names list from provided z values')
        self.zcur = self.z_start

    def next_batch(self):
        if self.zcur == self.z_end:
            return 0, None
        n_extracted = 0
        patch = np.zeros((self.H, self.W), dtype=np.uint8)
        self.patches_xyz = []
        while n_extracted < self.N:
            if self.xcur == 0 and self.ycur == 0:
                print(os.path.join(self.fullsize_dir, self.img_names[self.zcur]))
                self.img = scipy.misc.imread(
                    os.path.join(self.fullsize_dir, self.img_names[self.zcur]),
                    mode='L')
                self.img = np.invert(self.img)
                self.img_height = self.img.shape[0]
                self.img_width = self.img.shape[1]
            wmax = np.minimum(self.img_width, (self.xcur + 1) * self.W)
            hmax = np.minimum(self.img_height, (self.ycur + 1) * self.H)
            patch[0: hmax - self.ycur * self.H,
                  0: wmax - self.xcur * self.W] = \
                self.img[self.ycur * self.H: hmax, self.xcur * self.W: wmax]
            patch = patch.reshape((self.H, self.W, 1))
            self.patches[n_extracted, ...] = patch
            patch = patch.reshape((self.H, self.W))
            self.patches_xyz.append((self.xcur * self.W,
                                     self.ycur * self.H,
                                     self.zcur))
            n_extracted += 1
            self.xcur += 1
            if self.xcur > self.img_width // self.W:
                self.xcur = 0
                self.ycur += 1
                if self.ycur > self.img_height // self.H:
                    self.ycur = 0
                    self.zcur += 1
                    if self.zcur == self.z_end:
                        break
        return n_extracted, self.patches


def gen_image_tiles(img_path):
    manual_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data', 'manual')
    if not os.path.isfile(img_path):
        raise ValueError("{} does not exist".format(img_path))
    slide_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
    slide_name = slide_name.replace('AKO_', '')\
                           .replace('_P42_X30_01.vsi.Collection', '')
    slide_dir = os.path.join(manual_dir, slide_name)
    print(slide_dir)
    if not os.path.isdir(slide_dir):
        os.makedirs(slide_dir)
    I = cv2.imread(img_path, 0)
    for i in range(I.shape[0] // TILE_L):
        for j in range(I.shape[1] // TILE_L):
            if (i + 1) * TILE_L <= I.shape[0] and \
                    (j + 1) * TILE_L <= I.shape[1]:
                t = I[i * TILE_L: (i + 1) * TILE_L,
                      j * TILE_L: (j + 1) * TILE_L]
                t = 255 - t
                t_name = os.path.basename(img_path) \
                           .replace('.tif', '_x{}_y{}.tif'
                                    .format(j * TILE_L, i * TILE_L))
                cv2.imwrite(os.path.join(slide_dir, t_name), t)


def _gen_4class_label(annotate3_dir, annotate4_dir):
    if not os.path.isdir(annotate3_dir):
        raise ValueError('{} does not exist'.format(annotate3_dir))
    if not os.path.isdir(annotate4_dir):
        os.makedirs(annotate4_dir)
    for img_name in os.listdir(annotate3_dir):
        img_anno = cv2.imread(os.path.join(annotate3_dir, img_name), 0)
        anno_background = (img_anno == 0).astype(np.uint8)
        distance = cv2.distanceTransform(anno_background, cv2.DIST_L1, 3)
        img_anno[np.logical_and(distance <= BOUNDARY_WIDTH, anno_background > 0)] = 3
        cv2.imwrite(os.path.join(annotate4_dir, img_name), img_anno)


def gen_4class_label(src='train'):
    if not src == 'train' and not src == 'test' and not src == 'manual':
        raise ValueError('src label should come from train, test or manual set')
    data_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data')
    src_dir = os.path.join(data_dir, src)
    if src == 'train' or src == 'test':
        for crop_folder in os.listdir(src_dir):
            crop_dir = os.path.join(src_dir, crop_folder)
            annotate_dir = os.path.join(crop_dir, 'annotate')
            annotate3_dir = os.path.join(crop_dir, 'annotate_3')
            annotate4_dir = os.path.join(crop_dir, 'annotate_4')
            if not os.path.isdir(annotate3_dir) and os.path.isdir(annotate_dir):
                os.rename(annotate_dir, annotate3_dir)
            if not os.path.isdir(annotate4_dir):
                os.makedirs(annotate4_dir)
            _gen_4class_label(annotate3_dir, annotate4_dir)
    if src == 'manual':
        for d in os.listdir(src_dir):
            if d.find('_manual') > 0:
                src_dir = os.path.join(src_dir, d)
                break
        annotate3_dir = os.path.join(src_dir, 'annotate_3')
        if not os.path.isdir(annotate3_dir):
            raise ValueError('{} does not exist'.format(annotate3_dir))
        annotate4_dir = os.path.join(src_dir, 'annotate_4')
        if not os.path.isdir(annotate4_dir):
            os.makedirs(annotate4_dir)
        _gen_4class_label(annotate3_dir, annotate4_dir)


def gen_manual_label(group=None):
    data_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data')
    src_dir = os.path.join(data_dir, 'manual')
    for d in os.listdir(src_dir):
        if d.find('_manual') > 0:
            src_dir = os.path.join(src_dir, d)
            break
    if group == 'holdout':
        src_dir = os.path.join(src_dir, group)
    annotate3_dir = os.path.join(src_dir, 'annotate_3')
    if not os.path.isdir(annotate3_dir):
        os.makedirs(annotate3_dir)
    annotate4_dir = os.path.join(src_dir, 'annotate_4')
    if not os.path.isdir(annotate4_dir):
        os.makedirs(annotate4_dir)
    for img_name in os.listdir(src_dir):
        if img_name.find('.tif') < 0:
            continue
        img = cv2.imread(os.path.join(src_dir, img_name), -1)
        label3 = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
        # BGR
        label3[np.logical_and.reduce([img[:, :, 1] > 230, img[:, :, 1] > img[:, :, 0], img[:, :, 1] > img[:, :, 2]])] = 1
        label3[np.logical_and.reduce([img[:, :, 2] > 230, img[:, :, 2] > img[:, :, 0], img[:, :, 2] > img[:, :, 1]])] = 2
        cv2.imwrite(os.path.join(annotate3_dir, img_name), label3)
    gen_4class_label(src='manual')


if __name__ == '__main__':
    # gen_image_tiles(os.path.join(RAW_IMG_DIR, 'Z20.tif'))
    # gen_4class_label()
    gen_manual_label(group='holdout')
