import mxnet as mx

def _split_channels(total_filters, num_groups):
  split = [total_filters // num_groups for _ in range(num_groups)]
  split[0] += total_filters - sum(split)
  return split

def _split_data(data,num_groups,slice_axis=1):
    in_channels=data.shape[slice_axis]
    split = _split_channels(in_channels, num_groups)
    data_split = []
    for i in range(len(split)):
        if i == 0:
            data_split.append(mx.nd.slice_axis(data, axis=slice_axis, begin=0, end=split[i]))
        else:
            slice_index = sum(split[:i - 1])
            data_split.append(mx.nd.slice_axis(data, axis=slice_axis, begin=slice_index, end=slice_index + split[i]))
    return data_split

def _make_same_pad(ks):
    return ks//2

class MixBlock(mx.gluon.nn.HybridBlock):

    def __init__(self,in_channels=1,channels=1,t=1,dw_kernels=[1],pw_kernels=[1],stride=1,
                 se_ratio=0,use_sw=False,batch_norm_momentum=0.99,batch_norm_epsilon=1e-3,**kwargs):

        """Initializes the block.
            Parameters:
              ---------
              in_channels: int
                    Specify the number of input channels
              channels: int
                    Specify the number of output channls
              t: int
                    Specify the number of pw expand ratio
              dw_kernels:list
                    Specify kernel size list in depthwise convolution,split input according to length of kernels list
              pw_kernels:list
                    Specify kernel size list in pointwise convolution,split input according to length of kernels list
              stride:list
                    Specift stride size
              se_ratio: float
                    Specify whether use senet and ratio
              use_sw: bool
                    if True ,use swish activation to replace relu
        """
        super(MixBlock,self).__init__(**kwargs)
        with self.name_scope():
            self._expand=t
            self._pw_expand=mx.gluon.nn.HybridSequential(prefix='pw_expand_')
            self._dw=mx.gluon.nn.HybridSequential(prefix='dw_')
            self._pw_project = mx.gluon.nn.HybridSequential(prefix='pw_project_')
            self._pw_num_group=len(pw_kernels)
            self._dw_num_group=len(dw_kernels)
            self._activation= mx.gluon.nn.Swish() if use_sw else mx.gluon.nn.PReLU()
            with self._pw_expand.name_scope():
                expand_filter=in_channels*t
                split=_split_channels(expand_filter,self._pw_num_group)
                for i in range(self._pw_num_group):
                    self._pw_expand.add(mx.gluon.nn.Conv2D(split[i],pw_kernels[i],strides=1,padding=_make_same_pad(pw_kernels[i]),prefix='conv{}_{}*{}_'.format(i,pw_kernels[i],pw_kernels[i])))
                self._pw_expand.add(mx.gluon.nn.BatchNorm(momentum=batch_norm_momentum,epsilon=batch_norm_epsilon))
            with self._dw.name_scope():
                split=_split_channels(expand_filter,self._dw_num_group)
                for i in range(self._dw_num_group):
                    self._dw.add(mx.gluon.nn.Conv2D(split[i],dw_kernels[i],strides=stride,padding=_make_same_pad(dw_kernels[i]),groups=split[i],prefix='conv{}_{}*{}_'.format(i,dw_kernels[i],dw_kernels[i])))
                self._dw.add(mx.gluon.nn.BatchNorm(momentum=batch_norm_momentum,epsilon=batch_norm_epsilon))
            with self._pw_project.name_scope():
                split=_split_channels(channels,self._pw_num_group)
                for i in range(self._pw_num_group):
                    self._pw_project.add(mx.gluon.nn.Conv2D(split[i],pw_kernels[i],strides=1,padding=_make_same_pad(pw_kernels[i]),prefix='conv{}_{}*{}_'.format(i,pw_kernels[i],pw_kernels[i])))
                self._pw_project.add(mx.gluon.nn.BatchNorm(momentum=batch_norm_momentum,epsilon=batch_norm_epsilon))

    def hybrid_forward(self, F, x, *args, **kwargs):
        #pw expand
        x_splits = _split_data(x, self._pw_num_group)
        x_outputs = [op(data) for data, op in zip(x_splits, self._pw_expand)]
        pw_expand_out = mx.nd.concat(*x_outputs, dim=1)
        pw_expand_out = self._activation(self._pw_expand[-1](pw_expand_out))
        # if self._expand !=1:
        #     x_splits=_split_data(x,self._pw_num_group)
        #     x_outputs=[op(data) for data,op in zip(x_splits,self._pw_expand)]
        #     pw_expand_out=mx.nd.concat(*x_outputs,dim=1)
        #     pw_expand_out=self._activation(self._pw_expand[-1](pw_expand_out))
        # else:
        #     pw_expand_out=x

        #dw
        pw_expand_out_split=_split_data(pw_expand_out,self._dw_num_group)
        dw_outputs = [op(data) for data, op in zip(pw_expand_out_split, self._dw)]
        dw_out = mx.nd.concat(*dw_outputs, dim=1)
        dw_out = self._activation(self._dw[-1](dw_out))

        #pw project

        dw_out_splits = _split_data(dw_out, self._pw_num_group)
        pw_project_outputs = [op(data) for data, op in zip(dw_out_splits, self._pw_project)]
        pw_project_out = mx.nd.concat(*pw_project_outputs, dim=1)
        pw_project_out = self._pw_project[-1](pw_project_out)

        return pw_project_out



class MixNetsM(mx.gluon.nn.HybridBlock):

    def __init__(self,feature_only=False,num_classes=1000,batch_norm_momentum=0.99,batch_norm_epsilon=1e-3,head_feature_size=1536,dropout_rate=0.25,**kwargs):
        super(MixNetsM,self).__init__(**kwargs)
        self._feature_only=feature_only
        self._num_classes=num_classes
        with self.name_scope():
            self._feature=mx.gluon.nn.HybridSequential(prefix='features_')
            with self._feature.name_scope():
                self._feature.add(mx.gluon.nn.Conv2D(channels=24,in_channels=3,kernel_size=3,strides=2,padding=_make_same_pad(3),use_bias=False,prefix='stem_conv_'))
                self._feature.add(mx.gluon.nn.BatchNorm(momentum=batch_norm_momentum,epsilon=batch_norm_epsilon,prefix='stem_batchnorm_'))
                self._feature.add(mx.gluon.nn.PReLU(prefix='stem_prelu_'))
                self._feature.add(MixBlock(in_channels=24,channels=24,t=1,dw_kernels=[3],pw_kernels=[1],stride=1,prefix='blocks0_0_'))
                self._feature.add(MixBlock(in_channels=24,channels=32,t=6,dw_kernels=[3,5,7],pw_kernels=[1,1],stride=2,prefix='blocks1_0_'))
                self._feature.add(MixBlock(in_channels=32,channels=32,t=3,dw_kernels=[3],pw_kernels=[1,1],stride=1,prefix='blocks2_0_'))
                self._feature.add(MixBlock(in_channels=32,channels=40,t=6,dw_kernels=[3,5,7,9],pw_kernels=[1],stride=2,use_sw=True,prefix='blocks3_0_'))
                self._feature.add(MixBlock(in_channels=40,channels=40,t=6,dw_kernels=[3,5],pw_kernels=[1,1],stride=1,use_sw=True,prefix='blocks4_0_'))
                self._feature.add(MixBlock(in_channels=40,channels=40,t=6,dw_kernels=[3,5],pw_kernels=[1,1],stride=1,use_sw=True,prefix='blocks4_1_'))
                self._feature.add(MixBlock(in_channels=40,channels=40,t=6,dw_kernels=[3,5],pw_kernels=[1,1],stride=1,use_sw=True,prefix='blocks4_2_'))
                self._feature.add(MixBlock(in_channels=40,channels=80,t=6,dw_kernels=[3,5,7],pw_kernels=[1],stride=2,use_sw=True,prefix='blocks5_0_'))
                self._feature.add(MixBlock(in_channels=80,channels=80,t=6,dw_kernels=[3,5,7,9],pw_kernels=[1,1],stride=1,use_sw=True,prefix='blocks6_0_'))
                self._feature.add(MixBlock(in_channels=80,channels=80,t=6,dw_kernels=[3,5,7,9],pw_kernels=[1,1],stride=1,use_sw=True,prefix='blocks6_1_'))
                self._feature.add(MixBlock(in_channels=80,channels=80,t=6,dw_kernels=[3,5,7,9],pw_kernels=[1,1],stride=1,use_sw=True,prefix='blocks6_2_'))
                self._feature.add(MixBlock(in_channels=80,channels=120,t=6,dw_kernels=[3],pw_kernels=[1],stride=1,use_sw=True,prefix='blocks7_0_'))
                self._feature.add(MixBlock(in_channels=120,channels=120,t=3,dw_kernels=[3,5,7,9],pw_kernels=[1,1],stride=1,use_sw=True,prefix='blocks8_0_'))
                self._feature.add(MixBlock(in_channels=120,channels=120,t=3,dw_kernels=[3,5,7,9],pw_kernels=[1,1],stride=1,use_sw=True,prefix='blocks8_1_'))
                self._feature.add(MixBlock(in_channels=120,channels=120,t=3,dw_kernels=[3,5,7,9],pw_kernels=[1,1],stride=1,use_sw=True,prefix='blocks8_2_'))
                self._feature.add(MixBlock(in_channels=120,channels=200,t=6,dw_kernels=[3,5,7,9],pw_kernels=[1],stride=2,use_sw=True,prefix='blocks9_0_'))
                self._feature.add(MixBlock(in_channels=200,channels=200,t=6,dw_kernels=[3,5,7,9],pw_kernels=[1],stride=1,use_sw=True,prefix='blocks10_0_'))
                self._feature.add(MixBlock(in_channels=200,channels=200,t=6,dw_kernels=[3,5,7,9],pw_kernels=[1],stride=2,use_sw=True,prefix='blocks10_1_'))
                self._feature.add(MixBlock(in_channels=200,channels=200,t=6,dw_kernels=[3,5,7,9],pw_kernels=[1],stride=2,use_sw=True,prefix='blocks10_2_'))
            self._head=mx.gluon.nn.HybridSequential(prefix='head_')
            with self._head.name_scope():
                self._head.add(mx.gluon.nn.Conv2D(channels=head_feature_size,kernel_size=1,strides=1,padding=_make_same_pad(1)))
                self._head.add(mx.gluon.nn.BatchNorm(momentum=batch_norm_momentum,epsilon=batch_norm_momentum))
            self._output=mx.gluon.nn.HybridSequential(prefix='output_')
            with self._output.name_scope():
                self._output.add(mx.gluon.nn.GlobalAvgPool2D())
                if not feature_only:
                    self._output.add(mx.gluon.nn.Dropout(dropout_rate))
                    self._output.add(mx.gluon.nn.Dense(num_classes))


    def hybrid_forward(self, F, x, *args, **kwargs):
        feature=self._feature(x)
        if self._feature_only:
            return feature
        else:
            head = self._head(feature)
            head = mx.nd.relu(head)
            output = self._output(head)
            return output