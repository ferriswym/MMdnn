import imp
import os

import caffe
import numpy as np

from mmdnn.conversion.caffe.caffe_emitter import CaffeEmitter
from mmdnn.conversion.darknet.darknet_parser import DarknetParser
from mmdnn.conversion.examples.darknet.extractor import darknet_extractor
from mmdnn.conversion.examples.imagenet_test import TestKit
from utils import ensure_dir

tmpdir = './tmp/'

def caffe_emit(dstdir, IR_file, architecture_name, test_input_path=None):

    # IR to code
    converted_file = os.path.join(dstdir, architecture_name)
    emitter = CaffeEmitter((IR_file + ".pb", IR_file + ".npy"))
    emitter.run(converted_file + '.py', converted_file + '.npy', 'test')

    # import converted model
    imported = imp.load_source('CaffeModel', converted_file + '.py')

    imported.make_net(converted_file + '.prototxt')
    imported.gen_weight(converted_file + '.npy', converted_file + '.caffemodel', converted_file + '.prototxt')
    model_converted = caffe.Net(converted_file + '.prototxt', converted_file + '.caffemodel', caffe.TEST)

    if test_input_path:
        func = TestKit.preprocess_func[original_framework][architecture_name]
        img = func(test_input_path)
        img = np.transpose(img, [2, 0, 1])
        input_data = np.expand_dims(img, 0)

        model_converted.blobs[model_converted.inputs[0]].data[...] = input_data
        predict = model_converted.forward()[model_converted.outputs[-1]]
        converted_predict = np.squeeze(predict)
        print(converted_predict.shape)

    os.remove(converted_file + '.py')
    os.remove(converted_file + '.npy')
    # os.remove(converted_file + '.prototxt')
    # os.remove(converted_file + '.caffemodel')

def darknet_parse(architecture_name, cfg_path, weight_path):
    # original to IR
    IR_file = tmpdir + 'tmp'

    if architecture_name == "yolov3":
        start = "1"
    else:
        start = "0"

    parser = DarknetParser(cfg_path, weight_path, start)
    parser.run(IR_file)

    return IR_file

def darknet2caffe(dstdir, cfg_path, weight_path, architecture_name='yolov3'):
    IR_file = darknet_parse(architecture_name, cfg_path, weight_path)
    caffe_emit(dstdir, IR_file, architecture_name)
    
def main():
    cfg_path = "tests/cache/yolov3.cfg"
    weight_path = "tests/cache/yolov3.weights"
    ensure_dir(tmpdir)
    darknet2caffe(tmpdir, cfg_path, weight_path)

if __name__ == '__main__':
    main()
