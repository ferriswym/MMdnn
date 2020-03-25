from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from conversion_imagenet import TestModels
from mmdnn.conversion.examples.imagenet_test import TestKit
import numpy as np
import imp
import caffe

def get_test_table():
    return {
        'pytorch': {
            'yolov2': [TestModels.caffe_emit],
            'yolov3': [TestModels.caffe_emit],
        },

        'tensorflow': {

        },

        'darknet': {
            # 'yolov2': [TestModels.caffe_emit],
            'yolov3': [TestModels.caffe_emit],
        }
    }

image_path = "mmdnn/conversion/examples/data/dog.jpg"

def load_caffe_model(original_framework, architecture_name):
    converted_file = '/media/yuming/Pro/Projects/MMdnn/tests/tmp/darknet_caffe_yolov3_converted'
    # import converted model
    imported = imp.load_source('CaffeModel', converted_file + '.py')

    imported.make_net(converted_file + '.prototxt')
    imported.gen_weight(converted_file + '.npy', converted_file + '.caffemodel', converted_file + '.prototxt')
    model_converted = caffe.Net(converted_file + '.prototxt', converted_file + '.caffemodel', caffe.TEST)

    func = TestKit.preprocess_func[original_framework][architecture_name]
    img = func(image_path)
    img = np.transpose(img, [2, 0, 1])
    input_data = np.expand_dims(img, 0)

    model_converted.blobs[model_converted.inputs[0]].data[...] = input_data
    predict = model_converted.forward()[model_converted.outputs[-1]]
    converted_predict = np.squeeze(predict)

    return converted_predict

def test_framework():
    test_table = get_test_table()
    tester = TestModels(test_table)
    # tester._test_function('pytorch', tester.pytorch_parse)
    # tester._test_function('tensorflow', tester.tensorflow_parse)
    tester._test_function('darknet', tester.darknet_parse)
    # original_predict = tester.darknet_parse('yolov3', image_path)
    converted_predict = load_caffe_model('darknet', 'yolov3')
    # print("original predict: ", type(original_predict))
    print("converted_predict: ", converted_predict.shape)


if __name__ == '__main__':
    test_framework()
