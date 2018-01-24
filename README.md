## Caffe-to-Keras Weight Converter
---
### Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [How to use it](#how-to-use-it)
4. [Important notes](#important-notes)
5. [ToDo](#todo)
6. [Converted weights](#converted-weights)
7. [Why it is better to convert weights only, not model definitions](#why-it-is-better-to-convert-weights-only,-not-model-definitions)

### Overview

This is a Python tool to extract weights from a `.caffemodel` file and do either of two things:
1. Export the Caffe weights to an HDF5 file that is compatible with Keras 2.
Or
2. Export the Caffe weights to a pickled file that contains the weights as plain Numpy arrays along with some other information (such as layer types and names for all layers). This format may be useful if you want to load the weights into a deep learning framework other than Keras.

That is, this is mainly a Caffe-to-Keras weight converter, but you can also have it export the weights into a simpler, possibly more familiar Python format (list of dictionaries) instead.

Further below you can also find a list of links to weights for various models that I ported to Keras using this very converter.

There are tools out there that attempt to convert both the model definition and the weights of a Caffe model to a given other deep learning framework (like the great [`caffe-tensorflow`](https://github.com/ethereon/caffe-tensorflow)), but I don't believe that's the right approach. If you'd like to know why, read below. This program converts the weights only, not the model definition.

### Dependencies

* Python 2.7 or 3.x (this program is compatible with both)
* Numpy
* Caffe 1.x with Pycaffe
* h5py if you want to be able to generate Keras-compatible HDF5 files
* pickle if you want to be able to generate pickled files

The bad news is that you need to have Caffe with Pycaffe installed to use this converter. The good news is that you don't need to have any clue about how to use Caffe, it just needs to be installed.

### How to use it

#### 1. Command line interface

To convert a `.caffemodel` file to a Keras-compatible HDF5 file with verbose console output:
```c
python caffe_weight_converter.py 'desired/name/of/your/output/file/without/file/extension' \
                                 'path/to/the/caffe/model/definition.prototxt' \
                                 'path/to/the/caffe/weights.caffemodel' \
                                 --verbose
```
To extract the weights as Numpy arrays and save them in a pickled file along with layer types, names, inputs and outputs:
```c
python caffe_weight_converter.py 'desired/name/of/your/output/file/without/file/extension' \
                                 'path/to/the/caffe/model/definition.prototxt' \
                                 'path/to/the/caffe/weights.caffemodel' \
                                 --format=pickle
```
The command line interface takes three positional arguments in this order:
* `out_file`: The desired file name (including path) for the output file without the file extension. The file extension will be added by the converter and is `.h5` for HDF5 files and `.pkl` for pickle files.
* `prototxt`: The `.prototxt` file that contains the Caffe model definition
* `caffemodel`: The `.caffemodel` file that contains the weights for the Caffe model

For more details about the available options, execute
```c
python caffe_weight_converter.py --help
```

#### 2. Use within another Python program or Jupyter notebook

```python
from caffe_weight_converter import convert_caffemodel_to_keras, convert_caffemodel_to_dict
```
Read the documentation in [`caffe_weight_converter.py`](caffe_weight_converter.py) for details on how to use these two functions.

### Important notes

* The Keras converter supports the TensorFlow backend only at the moment, but Keras provides functions to transform weights from one backend to another. It would be nice to support the Theano format directly, but I don't know enough about Theano to do that. If you're a Theano user and interested in Theano support, let me know.
* Even though the Keras converter can generally convert the weights of any Caffe layer type, it is not guaranteed to do so correctly for layer types it doesn't know. For example, for layers that have multiple weight tensors, it might save the weight tensors in the wrong order, or certain weight tensors may need to be transposed or otherwise processed in a certain way, etc. The Keras converter provides the option to skip layer types that it doesn't know. It is recommended that you just try whether the converted weights of an unknown layer type work correctly, there is a chance that they will. Of course any layer types that do not have weights (such as Input, ReLU, Pooling, Reshape, etc.) will not cause any issues because the converter doesn't care about them. The currently supported Caffe layer types that have trainable weights are:
  * BatchNorm (i.e. BatchNorm layer followed by subsequent Scale layer)
  * Convolution
  * Deconvolution
  * InnerProduct

### ToDo

* Expand support for the Keras converter for other layer types. If you need a specific layer type to be supported, let me know.
* Support the Theano and CNTK backends for the Keras converter.

### Converted weights

I'll post any weights that I ported to Keras here. The filenames of the weight files are always the same as the names of the original `.caffemodel` files from which they were ported.

* [ResNet50](https://drive.google.com/open?id=1mXP3juk-fBFljindLdU0HXikGGZaQ8zP)
* [ResNet101](https://drive.google.com/open?id=1aGIduyjHqKIE4ZksRAd3OJ_w3-pEmWlA)
* [ResNet152](https://drive.google.com/open?id=19MYQYMJDuWyIf6hUukwwBteLZLijEb_L)
* [FCN-8s at-once Pascal](https://drive.google.com/open?id=1eesyNbscB_3ex_P4StW_PWtTwwFUO_nb)
* [SSD (all original models)](https://github.com/pierluigiferrari/ssd_keras)

### Why it is better to convert weights only, not model definitions

There are a few reasons why I think it makes more sense to have a converter that does't try to translate the model definition, but instead converts the model's weights only:

* Much easier to maintain: Every time that a converter that translates model definitions encounters a layer type it doesn't know, it breaks down. The authors of the converter then have to incorporate translation instructions for the new layer type so that the converter will know what to do with it. Since new network architectures lead to new Caffe layer types all the time, and countless models use their own custom layer types, this makes such a converter incredibly tedious to maintain. A converter that converts weights only doesn't have this problem to the same extent. This is because
  * (1) the vast majority of all layer types in Caffe doesn't have any weights. Think of Input, Split, ReLU, Pooling, or Reshape layers. This means that a converter that converts weights only doesn't have to care about those layers that don't have weights. If tomorrow a new activation function becomes popular (that has no trainable weights), then this converter will still work fine without needing any updates. A converter that tries to translate the model definition will need to be updated in order to know how to translate the new activation function. And
  * (2) even if tomorrow there is a new layer type in Caffe that does have trainable weights, then there is still at least chance that this converter can convert the weights of the new layer type correctly without any update. Weights are just numbers, and the format in which those numbers are arranged might be the same between Caffe and Keras if we are lucky. But there is a zero-percent chance that a converter that tries to translate the new layer type from Caffe to Keras can accidentally do that without any prior updates.
* Converting the model definition is unnecessary: I don't really care about the model definition being translated automatically for me. I can do that myself relatively easily. I can just look at the `.prototxt` file that defines the Caffe model and maybe at a few `.cpp` source files that define non-standard layer types and write my model in TensorFlow or Keras manually. It might take a while, but it's not that big a deal. What I **cannot** do, however, is manually transcribe the millions of weights from some non-human-readable binary protocol buffer file whose format I don't understand in the slightest. What I really care about is to have a program that gets those weights extracted and converted to the right format for me. Getting the TensorFlow (or whatever framework) graph definition served on a platter is nice to have but secondary.
