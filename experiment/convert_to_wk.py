import os
from config import cfg

def to_caffe():
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    command = 'mmconvert'
    srcFramework = ' -sf %s'%cfg.srcFramework
    inputWeight = ' -iw %s'%cfg.inputWeight
    inNodeName = ' --inNodeName %s'%cfg.inNodeName
    inputShape = ' --inputShape %d,%d,%d'%(cfg.inputShape[0], cfg.inputShape[1], cfg.inputShape[2])
    dstNodeName = ' --dstNodeName %s'%cfg.dstNodeName
    dstFramework = ' -df caffe'
    outputModel = ' -om %s'%os.path.join(cfg.output_dir, cfg.outputModel)
    os.system(command + srcFramework + inputWeight + inNodeName + inputShape + dstNodeName + dstFramework + outputModel)

def gen_ref_list():
    ref_list = os.listdir(cfg.image_ref_dir)
    with open(os.path.join(cfg.output_dir, cfg.NNIE.image_list), 'w') as f:
        for file_name in ref_list:
            f.write(os.path.join(cfg.image_ref_dir, file_name) + '\n')

def gen_wk_cfg():
    cfg_path = cfg.outputModel + '.cfg'
    with open(cfg_path, 'w') as f:
        for ele in cfg.NNIE:
            if cfg.NNIE[ele] != None:
                f.write("[{}] {}\n".format(ele, cfg.NNIE[ele]))
            else:
                f.write("[{}] {}\n".format(ele, "null"))
    return cfg_path

def caffe_to_wk():
    gen_ref_list()
    os.chdir(cfg.output_dir)
    cfg_path = gen_wk_cfg()
    command = 'nnie_mapper_11'
    os.system(command + ' ' + cfg_path)

def main():
    to_caffe()
    caffe_to_wk()

if __name__ == '__main__':
    main()