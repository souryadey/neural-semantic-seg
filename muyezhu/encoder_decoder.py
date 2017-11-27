from dataset import DataLoader

import numpy as np
import os
import scipy.misc
import tensorflow as tf

src_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = '/media/muyezhu/Dima/project_files/deep_learning/csci599_project/model'
copy_files = {os.path.join(src_dir, 'dataset.py'),
              os.path.join(src_dir, 'encoder_decoder.py')}


class BasicED:
    """
    only tuned for (256, 256) dimensions images
    """
    def __init__(self, N, H, W, n_labels=3, seg_method='auto',
                 lr=0.001, lr_decay=0.9, n_epoch=10):
        self.N, self.H, self.W, = N, H, W
        self.lr = lr
        self.lr_decay = lr_decay
        self.n_epoch = n_epoch
        self.n_labels = n_labels
        self.seg_method = seg_method
        self.steps = 0
        self.save_period = 100
        self.conv_layer_weights = {'relu1': 0.5,
                                   'relu2': 1,
                                   'relu3': 1,
                                   'relu4': 1,
                                   'relu5': 1.5,
                                   'relu6': 1.5}
        self.class_weights = {'background': 1,
                              'neurite': 5,
                              'soma': 15,
                              'border': 15}
        self.input = tf.placeholder(tf.float32, shape=(N, H, W, 1))
        self.label = tf.placeholder(tf.uint8, shape=(N, H, W, 1))
        self.is_train = tf.placeholder(tf.bool)
        self.save_dir = None
        self._model()
        self._init_ops()
        self.trained, self.restored = False, False

    def _init_ops(self):
        self.segment_op = self._segment()
        self.loss_op = self._loss()
        optimizer = tf.train.RMSPropOptimizer(self.lr, decay=self.lr_decay)
        self.train_op = optimizer.minimize(self.loss_op)

    def _model(self):
        n_filters = 16
        # in: N * 256 * 256 * 1
        with tf.variable_scope('conv1'):
            self.conv1 = tf.layers.conv2d(self.input, n_filters, (7, 7),
                                          padding='same', name='conv1',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(self.conv1.shape)
            self.bn1 = tf.layers.batch_normalization(self.conv1, name='bn1',
                                                     training=self.is_train)
            self.relu1 = tf.nn.relu(self.bn1)
            self.pool1 = tf.layers.max_pooling2d(self.relu1, (2, 2), (2, 2),
                                                 padding='same')
        # in: N * 128 * 128 * 8
        n_filters *= 2
        with tf.variable_scope('conv2'):
            self.conv2 = tf.layers.conv2d(self.pool1, n_filters, (5, 5),
                                          padding='same', name='conv2',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(self.conv2.shape)
            self.bn2 = tf.layers.batch_normalization(self.conv2, name='bn2',
                                                     training=self.is_train)
            self.relu2 = tf.nn.relu(self.bn2)
            self.pool2 = tf.layers.max_pooling2d(self.relu2, (2, 2), (2, 2), padding='same')
        # in: N * 64 * 64 * 16
        n_filters *= 2
        with tf.variable_scope('conv3'):
            self.conv3 = tf.layers.conv2d(self.pool2, n_filters, (5, 5),
                                          padding='same', name='conv3',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(self.conv3.shape)
            self.bn3 = tf.layers.batch_normalization(self.conv3, name='bn3',
                                                     training=self.is_train)
            self.relu3 = tf.nn.relu(self.bn3)
            self.pool3 = tf.layers.max_pooling2d(self.relu3, (2, 2), (2, 2), padding='same')
        # in: N * 32 * 32 * 32
        n_filters *= 2
        with tf.variable_scope('conv4'):
            self.conv4 = tf.layers.conv2d(self.pool3, n_filters, (3, 3),
                                          padding='same', name='conv4',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(self.conv4.shape)
            self.bn4 = tf.layers.batch_normalization(self.conv4, name='bn4',
                                                     training=self.is_train)
            self.relu4 = tf.nn.relu(self.bn4)
            self.pool4 = tf.layers.max_pooling2d(self.relu4, (2, 2), (2, 2), padding='same')
        # in: N * 16 * 16 * 64
        n_filters *= 2
        with tf.variable_scope('conv5'):
            self.conv5 = tf.layers.conv2d(self.pool4, n_filters, (3, 3),
                                          padding='same', name='conv5',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(self.conv5.shape)
            self.bn5 = tf.layers.batch_normalization(self.conv5, name='bn5',
                                                     training=self.is_train)
            self.relu5 = tf.nn.relu(self.bn5)
            self.pool5 = tf.layers.max_pooling2d(self.relu5, (2, 2), (2, 2), padding='same')
        # in: N * 8 * 8 * 128
        n_filters *= 2
        with tf.variable_scope('conv6'):
            self.conv6 = tf.layers.conv2d(self.pool5, n_filters, (3, 3),
                                          padding='same', name='conv6',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(self.conv6.shape)
            self.bn6 = tf.layers.batch_normalization(self.conv6, name='bn6',
                                                     training=self.is_train)
            self.relu6 = tf.nn.relu(self.bn6)
        # out: N * 16 * 16 * 64
        n_filters //= 2
        with tf.variable_scope('dconv1'):
            self.dconv1 = tf.layers.conv2d_transpose(self.relu6, n_filters,
                                                     (3, 3), strides=(2, 2),
                                                     padding='same',
                                                     name='dconv1',
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(self.dconv1.shape)
            self.dbn1 = tf.layers.batch_normalization(self.dconv1, name='dbn1',
                                                      training=self.is_train)
            self.drelu1 = tf.nn.relu(self.dbn1) + \
                          self.relu5 * self.conv_layer_weights['relu5']
        # out: N * 32 * 32 * 32
        n_filters //= 2
        with tf.variable_scope('dconv2'):
            self.dconv2 = tf.layers.conv2d_transpose(self.drelu1, n_filters,
                                                     (3, 3), strides=(2, 2),
                                                     padding='same',
                                                     name='dconv2',
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(self.dconv2.shape)
            self.dbn2 = tf.layers.batch_normalization(self.dconv2, name='dbn2',
                                                      training=self.is_train)
            self.drelu2 = tf.nn.relu(self.dbn2) + \
                          self.relu4 * self.conv_layer_weights['relu4']
        # out: N * 64 * 64 * 16
        n_filters //= 2
        with tf.variable_scope('dconv3'):
            self.dconv3 = tf.layers.conv2d_transpose(self.drelu2, n_filters,
                                                     (5, 5), strides=(2, 2),
                                                     padding='same',
                                                     name='dconv3',
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(self.dconv3.shape)
            self.dbn3 = tf.layers.batch_normalization(self.dconv3, name='dbn3',
                                                      training=self.is_train)
            self.drelu3 = tf.nn.relu(self.dbn3) + \
                          self.relu3 * self.conv_layer_weights['relu3']

        # out: N * 128 * 128 * 8
        n_filters //= 2
        with tf.variable_scope('dconv4'):
            self.dconv4 = tf.layers.conv2d_transpose(self.drelu3, n_filters,
                                                     (5, 5), strides=(2, 2),
                                                     padding='same',
                                                     name='dconv4',
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(self.dconv4.shape)
            self.dbn4 = tf.layers.batch_normalization(self.dconv4, name='dbn4',
                                                      training=self.is_train)
            self.drelu4 = tf.nn.relu(self.dbn4) + \
                          self.relu2 * self.conv_layer_weights['relu2']

        # out: N * 256 * 256 * 1
        n_filters //= 2
        with tf.variable_scope('dconv5'):
            self.dconv5 = tf.layers.conv2d_transpose(self.drelu4, n_filters,
                                                     (7, 7), strides=(2, 2),
                                                     padding='same',
                                                     name='dconv5',
                                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(self.dconv5.shape)
            self.dbn5 = tf.layers.batch_normalization(self.dconv5, name='dbn5',
                                                      training=self.is_train)
            self.drelu5 = tf.nn.relu(self.dbn5) + \
                          self.relu1 * self.conv_layer_weights['relu1']
            self.fc = tf.layers.conv2d_transpose(self.drelu5, self.n_labels, (1, 1),
                                                 padding='same', name='fc',
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(self.fc.shape)

    def _loss(self):
        logits = tf.reshape(self.fc, (-1, self.n_labels))
        labels = tf.reshape(self.label, (-1, 1))
        labels = tf.one_hot(labels, self.n_labels)
        labels = tf.reshape(labels, (-1, self.n_labels))
        weights = [self.class_weights['background'],
                   self.class_weights['neurite'],
                   self.class_weights['soma']]
        if self.n_labels == 4:
            weights.append(self.class_weights['border'])
        weights = np.array(weights)
        ce_loss = tf.nn.weighted_cross_entropy_with_logits(labels, logits, weights)
        return tf.reduce_mean(ce_loss)

    def _segment(self):
        logits = tf.reshape(self.fc, (-1, self.n_labels))
        activation = tf.nn.softmax(logits)
        classes = tf.argmax(activation, axis=1)
        return tf.reshape(classes, (self.N, self.H, self.W, -1))

    def _preprocess(self, data):
        if not data.shape == (self.N, self.H, self.W, 1):
            raise ValueError('data should be a numpy array of dimensions '
                             '({}, {}, {}, 1)'
                             .format(self.N, self.H, self.W, 1))
        data = data.reshape((self.N, -1))
        data = data - np.mean(data, axis=0)
        data /= 255
        data = data.reshape((self.N, self.H, self.W, 1))
        return data

    def train(self, session):
        loader = DataLoader(mode='train',
                            seg_method=self.seg_method,
                            n_class=self.n_labels)
        n_samples = loader.num_samples(self.H, self.W)
        for epoch in range(self.n_epoch):
            for i in range(n_samples // self.N):
                self.steps += 1
                data, label = loader.load_data(self.N, self.H, self.W,
                                               mode='train',
                                               seg_method=self.seg_method)
                data = self._preprocess(data)
                feed_dict = {self.input: data,
                             self.label: label,
                             self.is_train: True}
                _, loss, segmented = session.run([self.train_op,
                                                  self.loss_op,
                                                  self.segment_op],
                                                 feed_dict=feed_dict)
                print(np.sum(segmented == 1))
                print(np.sum(segmented == 2))
                print('Iteration {0}: loss = {1}'.format(self.steps, loss))
                if self.steps % self.save_period == 0:
                    print('Iteration {}: output segment result'.format(self.steps))
                    self.save_train_imgs(data, label, segmented)
                    self.save_model()
        self.trained = True
        self.restored = False

    def test(self, session):
        if not self.trained and not self.restored:
            raise ValueError('model is neither trained nor restored')
        loader = DataLoader(mode='test')
        n_samples = loader.num_samples(self.H, self.W)
        for i in range(n_samples // self.N):
            data = loader.load_data(self.N, self.H, self.W, mode='test')
            data = self._preprocess(data)
            # self.is_train here is not typo. setting it to False produces
            # all black output, possibly because each batch has only 2
            # distinct images and 6 rotated versions
            feed_dict = {self.input: data,
                         self.is_train: True}
            segmented = session.run([self.segment_op], feed_dict=feed_dict)
            print('test batch {0}'.format(i))
            self.save_test_imgs(data, segmented[0], i)

    def segment_imgs(self, session, imgs):
        if not self.trained and not self.restored:
            raise ValueError('model is neither trained nor restored')
        if not imgs.shape == (self.N, self.H, self.W, 1):
            raise ValueError('imgs should be a numpy array of dimensions '
                             '({}, {}, {}, 1)'
                             .format(self.N, self.H, self.W, 1))
        imgs = self._preprocess(imgs)
        feed_dict = {self.input: imgs, self.is_train: True}
        segmented = session.run([self.segment_op], feed_dict=feed_dict)
        return segmented[0]

    def save_train_imgs(self, data, labels, segmented):
        out_dir = os.path.join(self.get_save_dir(), 'train')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        for j in range(self.N):
            orig = np.squeeze(data[j, ...])
            label = np.squeeze(labels[j, ...])
            label_max = np.max(label)
            if label_max > 0:
                label *= (255 // label_max)
            seg = np.squeeze(segmented[j, ...])
            scipy.misc.imsave(os.path.join(out_dir,
                              '{}_{}orig.tif'.format(self.steps, j)), orig)
            scipy.misc.imsave(os.path.join(out_dir,
                              '{}_{}label.tif'.format(self.steps, j)), label)
            scipy.misc.imsave(os.path.join(out_dir,
                              '{}_{}seg.tif'.format(self.steps, j)), seg)

    def save_test_imgs(self, data, segmented, i):
        out_dir = os.path.join(self.get_save_dir(), 'test')
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        for j in range(self.N):
            orig = np.squeeze(data[j, ...])
            seg = np.squeeze(segmented[j, ...])
            scipy.misc.imsave(os.path.join(out_dir,
                              '{}_{}orig.tif'.format(i, j)), orig)
            scipy.misc.imsave(os.path.join(out_dir,
                              '{}_{}seg.tif'.format(i, j)), seg)

    def get_save_dir(self):
        from time import gmtime, strftime
        if self.save_dir is None:
            stamp = strftime("%y%m%d_%H%M", gmtime())
            self.save_dir = os.path.join(model_dir, 'basiced_{}'.format(stamp))
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        return self.save_dir

    def save_model(self):
        from shutil import copy
        with tf.device('/cpu:0'):
            saver = tf.train.Saver()
            save_path = saver.save(sess, os.path.join(self.get_save_dir(), "basiced.ckpt"))
            print("Model saved in file: %s" % save_path)
            if self.steps == self.save_period:
                for f in copy_files:
                    copy(f, os.path.join(self.get_save_dir(), os.path.basename(f)))
                    print('save source file: {}'.format(f))

    def restore_model(self, session, model_path):
        saver = tf.train.Saver()
        saver.restore(session, model_path)
        self.trained = False
        self.restored = True


if __name__ == '__main__':
    with tf.Session() as sess:
        ed = BasicED(8, 256, 256, n_labels=3, seg_method='manual', n_epoch=40)
        sess.run(tf.global_variables_initializer())
        with tf.device('/gpu:0'):
            ed.train(sess)
            ed.test(sess)
        ed.save_model()
