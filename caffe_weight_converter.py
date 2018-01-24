'''
A tool to convert `.caffemodel` weights to Keras-compatible HDF5 files or to export them to a simpler Python dictionary structure for further processing.

Copyright (C) 2018 Pierluigi Ferrari

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import os
os.environ['GLOG_minloglevel'] = '2' # Prevents Caffe from printing sooo much stuff to the console.
import caffe
os.environ['GLOG_minloglevel'] = '0'
import numpy as np
import warnings
import argparse
try:
    import pickle
except ImportError:
    warnings.warn("'pickle' module is missing. You can export the weights to an HDF5 file only.")
try:
    import h5py
except ImportError:
    warning.warn("'h5py' module is missing. You can export the weights to a pickle file only.")

def convert_caffemodel_to_keras(output_filename,
                                prototxt_filename,
                                caffemodel_filename,
                                include_layers_without_weights=False,
                                include_unknown_layer_types=True,
                                keras_backend='tf',
                                verbose=True):
    '''
    Converts Caffe weights from the `.caffemodel` format to an HDF5 format that is
    compatible with Keras 2.x with TensorFlow backend. The Theano and CNTK backends
    are currently not supported.

    Note that this converter converts the weights only, not the model definition.

    The most painfree way to use this weight converter is to leave the
    `include_layers_without_weights` option deactivated and load the weights into
    an appropriate Keras model by setting `by_name = True` in the `Model.load_weights()`
    method.

    This converter can handle all layer types, but it is not guaranteed to perform
    the conversion correctly for unknown layer types. What this means concretely
    is that the converter can always extract the weights from a Caffe model and
    put them into the Keras-compatible HDF5 format regardless of the layer type,
    but some layer types may need processing on top of that that the converter
    cannot perform for layers it doesn't know. Two potential issues come to mind
    for unsupported layer types:
    1) For layers that have multiple weight tensors, the converter will save
       the weight tensors in the order they have in the Caffe model. If this happens
       not to be the same order in which Keras saves the weights for that same
       layer type, then we obviously have a problem. For supported layer types
       it is ensured that the order is correct, but for a given unknown layer
       type this may or may not be the case.
    2) If the weights of a layer type need to be processed in a certain way,
       the converter is not going to know about this for unknown layer types.
       For example, the axes of the kernels of convolutional layers need to be
       transposed between Caffe and Keras with TensorFlow backend. Similar processing
       might be necessary for the weights of other (unknown) layer types to work
       correctly, so be aware of that.

    Of course any layer types that do not have trainable weights (such as Reshape,
    ReLU, Split, Concat, Permute, Flatten, Pooling etc.) won't cause any trouble
    because the converter does not care about them. The possible issues described
    above might occur only with unknown layer types that do have trainable weights.

    The currently supported (i.e. known) Caffe layer types that do have trainable
    weights are:
    - BatchNorm (i.e. BatchNorm layer followed by subsequent Scale layer)
    - Convolution
    - Deconvolution
    - InnerProduct

    If your model contains batch normalization layers, make sure that the names of
    the batch normalization layers in the Keras model are the same as the names of the
    corresponding 'BatchNorm' layers in the Caffe model, not the 'Scale' layers.

    Arguments:
        output_filename (str): The filename (full path, but excluding the file extension)
            under which to save the HDF5 file with the converted weights.
        prototxt_filename (str): The filename (full path including file extension)
            of the `.prototxt` file that defines the Caffe model.
        caffemodel_filename (str): The filename (full path including file extension)
            of the `.caffemodel` file that contains the weights for the Caffe model.
        include_layers_without_weights (bool, optional): If `False`, layers without
            weights (e.g. Input, Reshape, or ReLU layers) will be skipped by the
            converter. This means that the HDF5 output file will only contain those
            layers of a model that have any weights. This is the recommended usage
            of this converter, but if you really must include all layers
            in the output file, then set this option to `True`.
            Note: If `False`, then you should load the weights into the Keras model
            `by_name = True`, since not all layers are present in the HDF5 file.
        include_unknown_layer_types (bool, optional): If `True`, weights from unknown layer
            types will be included, even though it is not guaranteed that they will be
            converted correctly. It is recommended that you keep this option
            activated, see if the converted weights work correctly, and only deactivate
            this option in case they don't.
        keras_backend (str, optional): For which Keras backend to convert the weights.
            Currently only the TensorFlow backend is supported, but you can simply
            follow the procedure [here](https://github.com/keras-team/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa)
            to convert the resulting TensorFlow backend weights to Theano backend
            weights.
        verbose (bool, optional): If `True`, prints out the conversion status for
            every layer as well as some stats when the conversion is complete.

    Returns:
        None.
    '''
    if keras_backend != 'tf':
        raise ValueError("Only the TensorFlow backend is supported at the moment.")

    # Create a list of the Caffe model weights as Numpy arrays stored in dictionaries.
    # The reason why we use dictionaries is that we don't only store the weights themselves,
    # but also other information like the layer name, layer type, tops, and bottoms (tops = outputs,
    # bottoms = inputs for the non-Caffe people) for each layer.
    caffe_weights_list = convert_caffemodel_to_dict(prototxt_filename,
                                                    caffemodel_filename,
                                                    out_path=None,
                                                    verbose=False)

    # Create the HDF5 file in which to save the extracted weights.
    out_name = '{}.h5'.format(output_filename)
    out = h5py.File(out_name, 'w')

    # Save the layer names in this list.
    layer_names = []

    # These counters are just to be able to show some statistics upon completion of the conversion.
    counter_unknown = 0
    counter_no_weights = 0

    iterator = iter(range(len(caffe_weights_list)))

    for i in iterator:
        layer = caffe_weights_list[i]
        layer_name = layer['name']
        layer_type = layer['type']
        if (len(layer['weights']) > 0) or include_layers_without_weights: # Check whether this is a layer that contains weights.
            if layer_type in {'Convolution', 'Deconvolution', 'InnerProduct'}: # If this is a convolution layer or fully connected layer...
                # Get the kernel and transpose it.
                kernel = layer['weights'][0]
                if layer_type in {'Convolution', 'Deconvolution'}:
                    # Caffe kernel order for Convolution: `(out_channels, in_channels, filter_height, filter_width)`
                    # TensorFlow kernel order for Convolution: `(filter_height, filter_width, in_channels, out_channels)`
                    # Caffe kernel order for Deconvolution: `(in_channels, out_channels, filter_height, filter_width)`
                    # TensorFlow kernel order for Convolution Transpose: `(filter_height, filter_width, out_channels, in_channels)`
                    # That is, the transposition order is the same for both layer types.
                    kernel = np.transpose(kernel, (2, 3, 1, 0))
                if layer_type == 'InnerProduct':
                    # Transpose the kernel from Caffe's `(out_channels, in_channels)` format
                    # to TensorFlow's `(in_channels, out_channels)` format.
                    kernel = np.transpose(kernel, (1, 0))
                # Set the name for the kernel.
                weight_names = ['kernel']
                # If this layer has a bias (which does not necessarily have to be the case), add it, too.
                if (len(layer['weights']) > 1):
                    bias = layer['weights'][1]
                    weight_names.append('bias')
                # Compose the extended weight names with layer name prefix.
                extended_weight_names = np.array(['{}/{}:0'.format(layer_name, weight_names[k]).encode() for k in range(len(weight_names))])
                # Create a group (i.e. folder) named after this layer.
                group = out.create_group(layer_name)
                # Create a weight names attribute for this group, which is just a list of the names of the weights
                # that this layer is expected to have in the Keras model.
                group.attrs.create(name='weight_names', data=extended_weight_names)
                # Create a subgroup (i.e. subfolder) in which to save the weights of this layer.
                subgroup = group.create_group(layer_name)
                # Create the actual weights datasets.
                subgroup.create_dataset(name='{}:0'.format(weight_names[0]), data=kernel)
                if (len(layer['weights']) > 1):
                    subgroup.create_dataset(name='{}:0'.format(weight_names[1]), data=bias)
                # One last thing left to do: Append this layer's name to the global list of layer names.
                layer_names.append(layer_name.encode())
                if verbose:
                    print("Converted weights for layer '{}' of type '{}'".format(layer_name, layer_type))
            elif layer['type'] == 'BatchNorm': # If this is a batch normalization layer...
                # Caffe has a batch normalization layer, but it doesn't apply a scaling factor or bias
                # after normalizing. Instead, the 'BatchNorm' layer must be followed by a 'Scale' layer
                # in order to implement batch normalization the way you are used to. This means we
                # need to grab the weights from both this 'BatchNorm' layer and also from the subsequent
                # 'Scale' layer and put them together.
                # Gather all weights (expected: mean, variance, gamma, and beta) in this list.
                weights = []
                weight_names = []
                # Get the weights of this layer (the 'BatchNorm' layer).
                mean = layer['weights'][0]
                variance = layer['weights'][1]
                # If the subsequent layer is a 'Scale' layer, grab its weights, too.
                next_layer = caffe_weights_list[i + 1]
                if next_layer['type'] == 'Scale':
                    gamma = next_layer['weights'][0]
                    weights.append(gamma)
                    weight_names.append('gamma')
                    if (len(next_layer['weights']) == 1):
                        warnings.warn("This 'Scale' layer follows a 'BatchNorm' layer and is expected to have a bias, but it doesn't. Make sure to set `center = False` in the respective Keras batch normalization layer.")
                    else:
                        beta = next_layer['weights'][1]
                        weights.append(beta)
                        weight_names.append('beta')
                    # Increment the iterator by one since we need to skip the subsequent 'Scale' layer after we're done here.
                    next(iterator)
                else:
                    warnings.warn("No 'Scale' layer after 'BatchNorm' layer. Make sure to set `scale = False` and `center = False` in the respective Keras batch normalization layer.")
                weights.append(mean)
                weights.append(variance)
                weight_names.append('moving_mean') # It doesn't have to be a moving mean, but that's what Keras calls this parameter.
                weight_names.append('moving_variance')  # It doesn't have to be a moving variance, but that's what Keras calls this parameter.
                # Compose the extended weight names with layer name prefix.
                extended_weight_names = np.array(['{}/{}:0'.format(layer_name, weight_names[k]).encode() for k in range(len(weight_names))])
                # Create a group (i.e. folder) named after this layer.
                group = out.create_group(layer_name)
                # Create a weight names attribute for this group, which is just a list of the names of the weights
                # that this layer is expected to have in the Keras model.
                group.attrs.create(name='weight_names', data=extended_weight_names)
                # Create a subgroup (i.e. subfolder) in which to save the weights of this layer.
                subgroup = group.create_group(layer_name)
                # Create the actual weights datasets.
                for j in range(len(weights)):
                    subgroup.create_dataset(name='{}:0'.format(weight_names[j]), data=weights[j])
                # One last thing left to do: Append this layer's name to the global list of layer names.
                layer_names.append(layer_name.encode())
                if verbose:
                    print("Converted weights for layer '{}' of type '{}'".format(layer_name, layer_type))
            elif (len(layer['weights']) > 0) and include_unknown_layer_types: # For all other (unsupported) layer types...
                # Set the weight names for this layer type.
                weight_names = ['weights_{}'.format(i) for i in range(len(layer['weights']))]
                # Compose the extended weight names with layer name prefix.
                extended_weight_names = np.array(['{}/{}:0'.format(layer_name, weight_names[k]).encode() for k in range(len(weight_names))])
                # Create a group (i.e. folder) named after this layer.
                group = out.create_group(layer_name)
                # Create a weight names attribute for this group, which is just a list of the names of the weights
                # that this layer is expected to have in the Keras model.
                group.attrs.create(name='weight_names', data=extended_weight_names)
                # Create a subgroup (i.e. subfolder) in which to save the weights of this layer.
                subgroup = group.create_group(layer_name)
                # Create the actual weights datasets.
                for j in range(len(layer['weights'])):
                    subgroup.create_dataset(name='{}:0'.format(weight_names[j]), data=layer['weights'][j])
                # One last thing left to do: Append this layer's name to the global list of layer names.
                layer_names.append(layer_name.encode())
                if verbose:
                    print("Converted weights for layer '{}' of unknown type '{}'".format(layer_name, layer_type))
                counter_unknown += 1
            elif (len(layer['weights']) == 0):
                # Create a group (i.e. folder) named after this layer.
                group = out.create_group(layer_name)
                # Create a weight names attribute for this group, which is just a list of the names of the weights
                # that this layer is expected to have in the Keras model.
                group.attrs.create(name='weight_names', data=np.array([]))
                # Create a subgroup (i.e. subfolder) in which to save the weights of this layer.
                subgroup = group.create_group(layer_name)
                # One last thing left to do: Append this layer's name to the global list of layer names.
                layer_names.append(layer_name.encode())
                if verbose:
                    print("Processed layer '{}' of type '{}' which doesn't have any weights".format(layer_name, layer_type))
                counter_no_weights += 1
            elif verbose:
                print("Skipped layer '{}' of unknown type '{}'".format(layer_name, layer_type))
        elif verbose:
            print("Skipped layer '{}' of type '{}' because it doesn't have any weights".format(layer_name, layer_type))
    # Create the global attributes of this HDF5 file.
    out.attrs.create(name='layer_names', data=np.array(layer_names))
    out.attrs.create(name='backend', data=b'tensorflow')
    # Setting the Keras version is actually important since Keras uses this number to determine
    # whether and how it will convert the loaded weights. Since we're preparing the weights
    # in a way that is compatible with Keras version 2, we'll inform Keras about this by
    # setting the version accordingly.
    out.attrs.create(name='keras_version', data=b'2.0.8')
    # We're done, close the output file.
    out.close()
    print("Weight conversion complete.")
    if verbose:
        print("{} \t layers were processed, out of which:".format(len(layer_names)))
        print("{} \t were of an unknown layer type".format(counter_unknown))
        print("{} \t did not have any weights".format(counter_no_weights))
    print('File saved as {}'.format(out_name))

def convert_caffemodel_to_dict(prototxt_filename,
                               caffemodel_filename,
                               out_path=None,
                               verbose=False):
    '''
    Extracts the weights from a Caffe model into a simple structure of
    Python lists, dictionaries and Numpy arrays.

    Arguments:
        prototxt_filename (str): The full path to the `.prototxt` file that defines
            the Caffe model.
        caffemodel_filename (str): The full path to the `.caffemodel` file that
            contains the weights for this Caffe model.
        out_path (str, optional): The filename (full path, but excluding the file extension)
            under which to save a pickled file with the extracted weights. If `None`,
            then the extracted weights will not be saved to disk.
        verbose (bool, optional): If `True`, prints out the processing status for
            every layer.

    Returns:
        A list of dictionaries. Each dictionary contains the data for one layer of the
        model. The data contained in each dictionary can be accessed by the following keys:

            'name':    The name of the layer.
            'type':    The type of the layer, e.g. 'Convolution'.
            'weights': The weights of the layer as a list of Numpy arrays.
            'bottoms': The names and shapes of all inputs into the layer.
            'tops':    The names and shapes of all outputs from the layer.

        In case a layer has no weights, that layer's weights list will be empty.
    '''
    # Load the Caffe net and weights.
    net = caffe.Net(prototxt_filename, 1, weights=caffemodel_filename)
    # Store the weights and other information for each layer in this list.
    layer_list = []
    for li in range(len(net.layers)): # For each layer in the net...
        # ...store the weights and other relevant information in this dictionary.
        layer = {}
        # Store the layer name.
        layer['name'] = net._layer_names[li]
        # Store the layer type.
        layer['type'] = net.layers[li].type
        # Store the layer weights. In case the layer has no weights, this list will be empty.
        layer['weights'] = [net.layers[li].blobs[bi].data[...]
                            for bi in range(len(net.layers[li].blobs))]
        # Store the names and shapes of each input to this layer (aka "bottom").
        layer['bottoms'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape)
                             for bi in list(net._bottom_ids(li))]
        # Store the names and shapes of each output of this layer (aka "top").
        layer['tops'] = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape)
                          for bi in list(net._top_ids(li))]
        layer_list.append(layer)
        if verbose:
            print("Processed layer '{}' of type '{}'".format(layer['name'], layer['type']))

    # Free the occupied resources.
    del net

    if verbose:
        print("Weight extraction complete. Processed {} layers.".format(len(layer_list)))

    if not (out_path is None):
        out_name = '{}.pkl'.format(out_path)
        with open(out_name, 'wb') as f:
            pickle.dump(layer_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('File saved as {}.'.format(out_name))

    return layer_list

def main(argv):
    if argv.format == 'hdf5':
        convert_caffemodel_to_keras(output_filename=argv.out_file,
                                    prototxt_filename=argv.prototxt,
                                    caffemodel_filename=argv.caffemodel,
                                    include_layers_without_weights=argv.include_non_weight,
                                    include_unknown_layer_types=not(argv.skip_unknown),
                                    keras_backend=argv.backend,
                                    verbose=argv.verbose)
    elif argv.format == 'pickle':
        _ = convert_caffemodel_to_dict(prototxt_filename=argv.prototxt,
                                       caffemodel_filename=argv.caffemodel,
                                       out_path=argv.out_file,
                                       verbose=argv.verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Converts `.caffemodel` weights to either of '
                                                  '(1) Keras-compatible HDF5 format or '
                                                  '(2) a more general Python list of dictionaries suitable for further processing.'))
    parser.add_argument('out_file', action='store', type=str, help='The output filename as the full path, but excluding the file extension.')
    parser.add_argument('prototxt', action='store', type=str, help='The filename (full path including file extension) of the `.prototxt` file that defines the Caffe model. ')
    parser.add_argument('caffemodel', action='store', type=str, help='The filename (full path including file extension) of the `.caffemodel` file that contains the weights for the Caffe model.')
    parser.add_argument('-f', '--format', action='store', type=str, default='hdf5', choices={'hdf5', 'pickle'}, help="To which format to export the weights. Choices are {%(choices)s}, and the default is %(default)s. "
                                                                                                           "If the HDF5 format is selected, the converted weights will be compatible with Keras 2.x. "
                                                                                                           "If the Pickle format is selected, the weights will be exported to a more general Python list of "
                                                                                                           "dictionaries that contain the weights as Numpy arrays, along with other information such as "
                                                                                                           "layer names and types. This format may be useful if you want to process the weights further "
                                                                                                           "after exporting them.")
    parser.add_argument('-n', '--include_non_weight', action='store_true', default=False, help="This option is only relevant if the output format is HDF5. Include layers that have no weights "
                                                                                               "(e.g. Input, Reshape, or ReLU layers) in the converted weights file. "
                                                                                               "The recommended usage for HDF5 conversion is not to use this option and to load the weights into "
                                                                                               "the Keras model using `Model.load_weights()` with `by_name = True`.")
    parser.add_argument('-u', '--skip_unknown', action='store_true', default=False, help="This option is only relevant if the output format is HDF5. Skip layer types that are unknown to the "
                                                                                         "converter. It is recommended to try using the converter without this option first, then check whether the "
                                                                                         "converted weights work correctly or not, and only use this option in case they don't.")
    parser.add_argument('-b', '--backend', action='store', type=str, default='tf', choices={'tf'}, help="This option is only relevant if the output format is HDF5. For which Keras backend to convert the weights. "
                                                                                              "At the moment the only choice is 'tf' for the TensorFlow backend. %(default)s is also the default value.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help="Prints out the conversion status for every layer.")

    args = parser.parse_args()

    main(args)
