# Image-inpainting

First, dataset is processed in ‘data_process.py’. By using img directory, categories of img data go to list called ‘imgs_dir’. By using function ‘__getitem__’, the index can be achieved which is required to get an image. In this stage, data augmentation is implemented(random rotation of ±60˚random affine of 15˚).
Mask generator is also processed in ‘data_process.py’. For the irregular random masks, same masks which are used in gated convolution [1] were used. For the rectangular masks, random size, random number, and random location rectangular mask is used using ‘numpy’ module. In this case, max mask number is set to 10 and if the mask size is too big, the generator reduces the number of masks.
