"""
tensorflow/keras networks for voxelmorph
If you use this code, please cite one of the voxelmorph papers:
https://github.com/voxelmorph/voxelmorph/blob/master/citations.bib
Copyright 2020 Adrian V. Dalca
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""


# internal python imports
import warnings
from collections.abc import Iterable

# third party imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI

# local imports
import neurite as ne
from .. import default_unet_features
from . import layers
from . import utils

# make directly available from vxm
###############################################################################
# HyperMorph
###############################################################################

class HyperVxmDense(ne.modelio.LoadableModel):
    """
    Dense HyperMorph network for amortized hyperparameter learning.
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_hyp_params=1,
                 nb_hyp_layers=6,
                 nb_hyp_units=128,
                 name='hyper_vxm_dense',
                 **kwargs):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_hyp_params: Number of input hyperparameters.
            nb_hyp_layers: Number of dense layers in the hypernetwork.
            nb_hyp_units: Number of units in each dense layer of the hypernetwork.
            name: Model name - also used as layer name prefix. Default is 'vxm_dense'.
            kwargs: Forwarded to the internal VxmDense model.
        """

        # build hypernetwork
        hyp_input = tf.keras.Input(shape=[nb_hyp_params], name='%s_hyp_input' % name)
        hyp_last = hyp_input
        for n in range(nb_hyp_layers):
            hyp_last = KL.Dense(nb_hyp_units, activation='relu',
                                name='%s_hyp_dense_%d' % (name, n + 1))(hyp_last)

        # attach hypernetwork to vxm dense network
        hyp_model = tf.keras.Model(inputs=hyp_input, outputs=hyp_last, name='%s_hypernet' % name)
        vxm_model = VxmDense(inshape, hyp_model=hyp_model, name=name, **kwargs)

        # rebuild model
        super().__init__(name=name, inputs=vxm_model.inputs, outputs=vxm_model.outputs)

        # cache pointers to layers and tensors for future reference
        self.references = vxm_model.references
        self.references.hyper_val = hyp_input


###############################################################################
# Private functions
###############################################################################

def _conv_block(x, nfeat, strides=1, name=None, do_res=False, hyp_tensor=None,
                include_activation=True, kernel_initializer='he_normal'):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims

    extra_conv_params = {}
    if hyp_tensor is not None:
        Conv = getattr(ne.layers, 'HyperConv%dDFromDense' % ndims)
        conv_inputs = [x, hyp_tensor]
    else:
        Conv = getattr(KL, 'Conv%dD' % ndims)
        extra_conv_params['kernel_initializer'] = kernel_initializer
        conv_inputs = x

    convolved = Conv(nfeat, kernel_size=3, padding='same',
                     strides=strides, name=name, **extra_conv_params)(conv_inputs)

    if do_res:
        # assert nfeat == x.get_shape()[-1], 'for residual number of features should be constant'
        add_layer = x
        print('note: this is a weird thing to do, since its not really residual training anymore')
        if nfeat != x.get_shape().as_list()[-1]:
            add_layer = Conv(nfeat, kernel_size=3, padding='same',
                             name='resfix_' + name, **extra_conv_params)(conv_inputs)
        convolved = KL.Lambda(lambda x: x[0] + x[1])([add_layer, convolved])

    if include_activation:
        name = name + '_activation' if name else None
        convolved = KL.LeakyReLU(0.2, name=name)(convolved)

    return convolved


def _upsample_block(x, connection, factor=2, name=None):
    """
    Specific upsampling and concatenation layer for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    UpSampling = getattr(KL, 'UpSampling%dD' % ndims)

    size = (factor,) * ndims if ndims > 1 else factor
    upsampled = UpSampling(size=size, name=name)(x)
    name = name + '_concat' if name else None
    return KL.concatenate([upsampled, connection], name=name)
