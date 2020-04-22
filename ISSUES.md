Private:

## 1. BN层不过

原因: BN层有不同的实现方式，可能有些实现方式有些算子不支持，需换种实现方式

fixed: 目前用的是tf.contrib里面的实现，可以过

## 2. 反卷积没有实现

fixed: 参考卷积的实现实现了反卷积Deconvoulution

## 3. AttributeError: data_filler

原因: caffe的constant类型只支持数值，不支持list，另外即使caffe支持DummyData，NNIE也不支持

fixed: 限制计算图设计不出现常数算子，即不出现DummyData层

## 4. tensorflow的pad机制可以支持两边不对称pad，一般在设成SAME用到，但是caffe只支持对称的pad

fixed: 原mmdnn中加入了crop机制来检查shape是否相等，并crop成相等的，但是因为这个机制需要加入DummyData，所以不适用，只能**在设计网络时限制输入图片尺寸为16的倍数以在层间shape计算时不出现问题**
