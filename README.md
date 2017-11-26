# neural-semantic-seg
Semantic Segmentation of Ultra-Resolution 3D Microscopic Neural Imagery

Data:

filtered_image_data_{x}.npz, where x=1,2,3,4,5 -- Input data, each has shape (num,2048,2048), where num=80 for x=1,2,3,4 and num=76 for x=5. All values are float32 [0-1]
thresolded_image_data.npz -- Labeled data with shape (396,2048,2048). All values are float32 {0,1}

filtered_image_data_keras_{x}.npz, where x=1,2,3,4,5 -- Input data, each has shape (num,2048,2048,1), where num=80 for x=1,2,3,4 and num=76 for x=5. All values are float16 [0-1]
thresholded_image_data_keras.npz -- Keras-friendly labeled data with shape (396,4194304,2), where 4194304=2048*2048. Each [x,y] index is a 2-tuple which is either [1,0] if the original value was 0, otherwise [0,1] if the original value was 1. Float64
thresholded_image_data_keras.npz -- Same as above, except this is int8