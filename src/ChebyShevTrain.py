import os
CUDA_VISIBLE_DEVICES="0"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import neurite as ne
import voxelmorph as vxm
import tensorflow as tf
import nibabel as nib
from scipy import ndimage
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI
import csv
import functools
import scipy
from skimage import measure
import pystrum.pynd.ndutils as nd
tf.compat.v1.disable_eager_execution()
import matplotlib
import spektral
from sklearn.model_selection import GridSearchCV
import spektral.layers as sp
from tensorflow.keras import regularizers
import medpy.metric.binary

from plot_keras_history import show_history, plot_history

tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)
def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    #scan = image_read(filepath) 
    #scan = ants.image_read(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

#def normalize(volume):
#    """Normalize the volume"""
#    min = -1000
#    max = 400
#    volume[volume < min] = min
#    volume[volume > max] = max
#    volume = (volume - min) / (max - min)
#    volume = volume.astype("float32")
#    return volume

#def resize_volume(img):
#    """Resize across z-axis"""
#    # Set the desired depth
#    desired_depth = 224
#    desired_width = 160
#    desired_height = 192
#    # Get current depth
#    current_depth = img.shape[-1]
#    current_width = img.shape[0]
#    current_height = img.shape[1]
#    # Compute depth factor
#    depth = current_depth / desired_depth
#    width = current_width / desired_width
#    height = current_height / desired_height
#    depth_factor = 1 / depth
#    width_factor = 1 / width
#    height_factor = 1 / height
#    # Rotate
#    img = ndimage.rotate(img, 90, reshape=False)
#    # Resize across z-axis
#    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
#    return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    #volume = normalize(volume)
    # Resize width, height and depth
    #volume = resize_volume(volume)
    return volume
  train_data, validate_data, test_data = np.array_split(images, 3)
  def voxelmorph_generator(data):
    image_shape = data.shape[1:]
    ndims = len(image_shape)
    # the gradient loss is computed entirely on the predicted deformation,
    # so the reference data for the second model output can be ignored and
    # we can just provide an image of zeros
    zeros = np.zeros([1, *image_shape, ndims], dtype='float32')
    while True:
        # for each training iteration, grab a random pair of images to align
        moving_image = data[np.random.randint(len(data))][np.newaxis]
        fixed_image = data[np.random.randint(len(data))][np.newaxis]
        inputs = [moving_image, fixed_image]
        outputs = [fixed_image, zeros]
        yield (inputs, outputs)
   #adjacency_tensor = np.array([[1, 1],
#                             [1, 1]])
adjacency_tensor = images[0].squeeze()
adjacency_tensor.shape
#adjacency_tensor = np.array([[1, 1],
#                             [1, 1]])
adjacency_tensor = images[0].squeeze()
adjacency_tensor.shape
lambda_weight = hp_input
image_loss = lambda yt, yp: vxm.losses.MSE(0.05).loss(yt, yp) * (1 - lambda_weight)
gradient_loss = lambda yt, yp: vxm.losses.Grad('l2').loss(yt, yp) * lambda_weight
losses = [image_loss, gradient_loss]
#Model Design For Registration
hp_input = tf.keras.Input(shape=[1])
x = sp.ChebConv(32,
                K=1,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.01))([hp_input, adjacency_tensor])
x = sp.ChebConv(64,
                K=1,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.01))([x, adjacency_tensor])
x = sp.ChebConv(128,
                K=1,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.01))([x, adjacency_tensor])
x = sp.ChebConv(128,
                K=1,
                activation='relu',
                dropout_rate=0.5,
                kernel_regularizer=regularizers.l2(0.01))([x, adjacency_tensor])
x = sp.ChebConv(256,
                K=1,
                activation='relu',
                dropout_rate=0.5,
                kernel_regularizer=regularizers.l2(0.01))([x, adjacency_tensor])
x = sp.ChebConv(1024,
                K=1,
                activation='relu',
                dropout_rate=0.5,
                kernel_regularizer=regularizers.l2(0.01))([x, adjacency_tensor])
x = sp.ChebConv(2048,
                K=1,
                activation='relu',
                dropout_rate=0.5,
                kernel_regularizer=regularizers.l2(0.01))([x, adjacency_tensor])
x = sp.ChebConv(4096,
                K=1,
                activation='relu',
                dropout_rate=0.5,
                kernel_regularizer=regularizers.l2(0.01))([x, adjacency_tensor])
x = sp.ChebConv(8192,
                K=1,
                activation='relu',
                dropout_rate=0.5,
                kernel_regularizer=regularizers.l2(0.01))([x, adjacency_tensor])
x = sp.ChebConv(16384,
                K=1,
                activation='relu',
                dropout_rate=0.5,
                kernel_regularizer=regularizers.l2(0.01))([x, adjacency_tensor])
chebyshev_hypernetwork = tf.keras.Model(hp_input, x, name='chebyshev_hypernetwork')
model = vxm.networks.VxmDense(image_shape, int_steps=0, hyp_model=chebyshev_hypernetwork)
lambda_weight = hp_input
image_loss = lambda yt, yp: vxm.losses.MSE(0.05).loss(yt, yp) * (1 - lambda_weight)
gradient_loss = lambda yt, yp: vxm.losses.Grad('l2').loss(yt, yp) * lambda_weight
losses = [image_loss, gradient_loss]
def hypermorph_generator(data):
    # initialize the base image pair generator
    gen = voxelmorph_generator(data)
    while True:
        inputs, outputs = next(gen)
        # append a random λ value to the input list
        inputs = (*inputs, np.random.rand(1, 1))
        yield (inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses)
gen = hypermorph_generator(train_data)
history = model.fit_generator(gen, epochs=1500, steps_per_epoch=100, verbose=True)
lambdas = np.linspace(0, 2, 8).reshape((-1, 1))
slices = [model.predict([moving_image, fixed_image, hyp])[0] for hyp in lambdas]
titles = ['λ = %.2f' % hyp for hyp in lambdas]
ne.plot.slices(slices, titles=titles)
print('dice score')
print(f'image[0] {medpy.metric.binary.dc(slices[0],fixed_image)}')
print(f'image[1] {medpy.metric.binary.dc(slices[1],fixed_image)}')
print(f'image[2] {medpy.metric.binary.dc(slices[2],fixed_image)}')
print(f'image[3] {medpy.metric.binary.dc(slices[3],fixed_image)}')
print(f'image[4] {medpy.metric.binary.dc(slices[4],fixed_image)}')
print(f'image[5] {medpy.metric.binary.dc(slices[5],fixed_image)}')
print(f'image[6] {medpy.metric.binary.dc(slices[6],fixed_image)}')
print(f'image[7] {medpy.metric.binary.dc(slices[7],fixed_image)}')
print('Target Registration Error')
print(f'image[0] {medpy.metric.binary.hd95(slices[0],fixed_image)}')
print(f'image[1] {medpy.metric.binary.hd95(slices[1],fixed_image)}')
print(f'image[2] {medpy.metric.binary.hd95(slices[2],fixed_image)}')
print(f'image[3] {medpy.metric.binary.hd95(slices[3],fixed_image)}')
print(f'image[4] {medpy.metric.binary.hd95(slices[4],fixed_image)}')
print(f'image[5] {medpy.metric.binary.hd95(slices[5],fixed_image)}')
print(f'image[6] {medpy.metric.binary.hd95(slices[6],fixed_image)}')
print(f'image[7] {medpy.metric.binary.hd95(slices[7],fixed_image)}')
