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
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel


class VxmDense(ne.modelio.LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 svf_resolution=1,
                 int_resolution=2,
                 int_downsize=None,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 input_model=None,
                 hyp_model=None,
                 fill_value=None,
                 reg_field='preintegrated',
                 name='vxm_dense'):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an 
                integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            svf_resolution: Resolution (relative voxel size) of the predicted SVF.
                Default is 1.
            int_resolution: Resolution (relative voxel size) of the flow field during
                vector integration. Default is 2.
            int_downsize: Deprecated - use int_resolution instead.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Deprecated - use svf_resolution instead.
            input_model: Model to replace default input layer before concatenation. Default is None.
            hyp_model: HyperMorph hypernetwork model. Default is None.
            reg_field: Field to regularize in the loss. Options are 'svf' to return the
                SVF predicted by the Unet, 'preintegrated' to return the SVF that's been
                rescaled for vector-integration (default), 'postintegrated' to return the
                rescaled vector-integrated field, and 'warp' to return the final, full-res warp.
            name: Model name - also used as layer name prefix. Default is 'vxm_dense'.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), name='%s_source_input' % name)
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='%s_target_input' % name)
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        # configure inputs
        inputs = input_model.inputs
        if hyp_model is not None:
            hyp_input = hyp_model.input
            hyp_tensor = hyp_model.output
            if not any([hyp_input is inp for inp in inputs]):
                inputs = (*inputs, hyp_input)
        else:
            hyp_input = None
            hyp_tensor = None

        if int_downsize is not None:
            warnings.warn('int_downsize is deprecated, use the int_resolution parameter.')
            int_resolution = int_downsize

        # compute number of upsampling skips in the decoder (to downsize the predicted field)
        if unet_half_res:
            warnings.warn('unet_half_res is deprecated, use the svf_resolution parameter.')
            svf_resolution = 2

        nb_upsample_skips = int(np.floor(np.log(svf_resolution) / np.log(2)))

        # build core unet model and grab inputs
        unet_model = Unet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            nb_upsample_skips=nb_upsample_skips,
            hyp_input=hyp_input,
            hyp_tensor=hyp_tensor,
            name='%s_unet' % name
        )

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        flow_mean = Conv(ndims, kernel_size=3, padding='same',
                         kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                         name='%s_flow' % name)(unet_model.output)

        # optionally include probabilities
        if use_probs:
            # initialize the velocity variance very low, to start stable
            flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                                 kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
                                 bias_initializer=KI.Constant(value=-10),
                                 name='%s_log_sigma' % name)(unet_model.output)
            flow_params = KL.concatenate([flow_mean, flow_logsigma], name='%s_prob_concat' % name)
            flow_inputs = [flow_mean, flow_logsigma]
            flow = ne.layers.SampleNormalLogVar(name='%s_z_sample' % name)(flow_inputs)
        else:
            flow = flow_mean

        # rescale field to target svf resolution
        pre_svf_size = np.array(flow.shape[1:-1])
        svf_size = np.array([np.round(dim / svf_resolution) for dim in inshape])
        if not np.array_equal(pre_svf_size, svf_size):
            rescale_factor = svf_size[0] / pre_svf_size[0]
            flow = layers.RescaleTransform(rescale_factor, name=f'{name}_svf_resize')(flow)

        # cache svf
        svf = flow

        # rescale field to target integration resolution
        if int_steps > 0 and int_resolution > 1:
            int_size = np.array([np.round(dim / int_resolution) for dim in inshape])
            if not np.array_equal(svf_size, int_size):
                rescale_factor = int_size[0] / svf_size[0]
                flow = layers.RescaleTransform(rescale_factor, name=f'{name}_flow_resize')(flow)

        # cache pre-integrated flow field
        preint_flow = flow

        # optionally negate flow for bidirectional model
        pos_flow = flow
        if bidir:
            neg_flow = ne.layers.Negate(name='%s_neg_flow' % name)(flow)

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow = layers.VecInt(method='ss',
                                     name='%s_flow_int' % name,
                                     int_steps=int_steps)(pos_flow)
            if bidir:
                neg_flow = layers.VecInt(method='ss',
                                         name='%s_neg_flow_int' % name,
                                         int_steps=int_steps)(neg_flow)

        # cache the intgrated flow field
        postint_flow = pos_flow

        # resize to final resolution
        if int_steps > 0 and int_resolution > 1:
            rescale_factor = inshape[0] / int_size[0]
            pos_flow = layers.RescaleTransform(rescale_factor, name='%s_diffflow' % name)(pos_flow)
            if bidir:
                neg_flow = layers.RescaleTransform(rescale_factor,
                                                   name='%s_neg_diffflow' % name)(neg_flow)

        # warp image with flow field
        y_source = layers.SpatialTransformer(
            interp_method='linear',
            indexing='ij',
            fill_value=fill_value,
            name='%s_transformer' % name)([source, pos_flow])

        if bidir:
            st_inputs = [target, neg_flow]
            y_target = layers.SpatialTransformer(interp_method='linear',
                                                 indexing='ij',
                                                 fill_value=fill_value,
                                                 name='%s_neg_transformer' % name)(st_inputs)

        # initialize the keras model
        outputs = [y_source, y_target] if bidir else [y_source]

        # determine regularization output
        reg_field = reg_field.lower()
        if use_probs:
            # compute loss on flow probabilities
            outputs.append(flow_params)
        elif reg_field == 'svf':
            # regularize the immediate, predicted SVF
            outputs.append(svf)
        elif reg_field == 'preintegrated':
            # regularize the rescaled, pre-integrated SVF
            outputs.append(preint_flow)
        elif reg_field == 'postintegrated':
            # regularize the rescaled, integrated field
            outputs.append(postint_flow)
        elif reg_field == 'warp':
            # regularize the final, full-resolution deformation field
            outputs.append(pos_flow)
        else:
            raise ValueError(f'Unknown option "{reg_field}" for reg_field.')

        super().__init__(name=name, inputs=inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.source = source
        self.references.target = target
        self.references.svf = svf
        self.references.preint_flow = preint_flow
        self.references.postint_flow = postint_flow
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow if bidir else None
        self.references.y_source = y_source
        self.references.y_target = y_target if bidir else None
        self.references.hyp_input = hyp_input

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        st_input = [img_input, warp_model.output]
        y_img = layers.SpatialTransformer(interp_method=interp_method)(st_input)
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])
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
