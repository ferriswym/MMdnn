import os
from easydict import EasyDict as edict

__C = edict()

# import the module by 
# from config import cfg
cfg = __C

# framework
__C.srcFramework        = "tensorflow" # choose from ['tensorflow', 'pytorch']
__C.dstFramework        = "caffe"

# input weight and network file path
# if only one file, put it in weight path
__C.inputWeight         = "/media/yuming/Pro/Projects/edgeAI/model/rice/denserfb-fpn-rice-weed.pb"
__C.inputNetwork        = ""

# node and shape
__C.inNodeName          = "data"
__C.dstNodeName         = "sigmoid"
__C.inputShape          = (512, 512, 3)

# output name
__C.output_dir          = "tmp/"
__C.outputModel         = "convert_test"

# directory of quantization reference images
__C.image_ref_dir       = "/media/yuming/Pro/Projects/edgeAI/data/rice/train/images" # better be absolute path

# NNIE mapper params
__C.NNIE                = edict()

__C.NNIE.prototxt_file      = __C.outputModel + ".prototxt"
__C.NNIE.caffemodel_file    = __C.outputModel + ".caffemodel"
__C.NNIE.batch_num          = 1
__C.NNIE.net_type           = 0
__C.NNIE.sparse_rate        = 0
__C.NNIE.compile_mode       = 1
__C.NNIE.is_simulation      = 0
__C.NNIE.log_level          = 1
__C.NNIE.instruction_name   = __C.outputModel
__C.NNIE.RGB_order          = "BGR"
__C.NNIE.data_scale         = 0.0039216
__C.NNIE.internal_stride    = 16
__C.NNIE.image_list         = "image_ref_list.txt"
__C.NNIE.image_type         = 1
__C.NNIE.mean_file          = None
__C.NNIE.norm_type          = 3
__C.NNIE.is_check_prototxt  = 0