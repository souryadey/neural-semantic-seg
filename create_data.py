import numpy as np
np.set_printoptions(threshold=np.inf)
from PIL import Image

def create_data(dataset='thresholded'):
    '''
    Crop 8192x2048 images into four 2048x2048 images
    dataset: Either 'filtered' or 'thresholded'
    '''
    if dataset=='filtered':
        appendname = ''
    elif dataset=='thresholded':
        appendname  = '_seg'
    image_data = np.zeros((396,2048,2048),dtype='float32')
    for i in xrange(99):
        s = str(i)
        if len(s)==1:
            s = '0'+s
        im = Image.open('./{0}/Z{1}_crop_invert_filtered{2}.tif'.format(dataset,s,appendname))
        imnp = np.asarray(im)
        for j in xrange(4):
            image_data[i*4+j] = imnp[:,j*2048:(j+1)*2048]/255.
        print '{0}th image done'.format(i)
    if dataset=='filtered':
        np.savez_compressed('./filtered_image_data_1.npz', data=image_data[:80,:,:])
        np.savez_compressed('./filtered_image_data_2.npz', data=image_data[80:160,:,:])
        np.savez_compressed('./filtered_image_data_3.npz', data=image_data[160:240,:,:])
        np.savez_compressed('./filtered_image_data_4.npz', data=image_data[240:320,:,:])
        np.savez_compressed('./filtered_image_data_5.npz', data=image_data[320:,:,:])
    elif dataset=='thresholded':
        np.savez_compressed('./thresholded_image_data.npz', data=image_data)




#==============================================================================
# im2 = Image.open(os.path.dirname(os.path.dirname(os.path.realpath('__file__')))+'/cropped_labeled_images/filtered/threshold/Z00_crop_invert_filtered_seg.tif')
# imnp2 = np.asarray(im2)
# imnp2 = imnp2.astype('int')
# imnp2 = 255-imnp2
#==============================================================================

