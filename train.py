import argparse
import logging
import math
import os
import time

import mxnet as mx
from gluoncv.utils import makedirs, LRSequential, LRScheduler
from mxnet import autograd
from mxnet import gluon

from models import MixNetsM


def parse_args():
    parser = argparse.ArgumentParser(description='Train a mixnets model for image classification.')
    parser.add_argument('--rec-train', type=str, default='~/.mxnet/datasets/imagenet/rec/train.rec',
                        help='the training data')
    parser.add_argument('--rec-train-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/train.idx',
                        help='the index of training data')
    parser.add_argument('--rec-val', type=str, default='~/.mxnet/datasets/imagenet/rec/val.rec',
                        help='the validation data')
    parser.add_argument('--rec-val-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/val.idx',
                        help='the index of validation data')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--use_se', action='store_true',
                        help='use SE layers or not in resnext. default is false.')
    parser.add_argument('--save-frequency', type=int, default=10,
                        help='frequency of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--logging-file', type=str, default='train_mixnets.log',
                        help='name of training log file')
    opt = parser.parse_args()
    return opt


def get_data_rec(opt,batch_size,num_workers):
    rec_train = os.path.expanduser(opt.rec_train)
    rec_train_idx = os.path.expanduser(opt.rec_train_idx)
    rec_val = os.path.expanduser(opt.rec_val)
    rec_val_idx = os.path.expanduser(opt.rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    input_size = opt.input_size
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_train,
        path_imgidx         = rec_train_idx,
        preprocess_threads  = num_workers,
        shuffle             = True,
        batch_size          = batch_size,

        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = 4. / 3.,
        min_aspect_ratio    = 3. / 4.,
        max_random_area     = 1,
        min_random_area     = 0.08,
        brightness          = jitter_param,
        saturation          = jitter_param,
        contrast            = jitter_param,
        pca_noise           = lighting_param,
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,

        resize              = resize,
        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return train_data, val_data, batch_fn

def test(ctx, acc_top1,acc_top5,val_data,batch_fn,net):
    val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1-top1, 1-top5)

def main():
    opt = parse_args()

    filehandler=logging.FileHandler(opt.logging_file,mode='w')
    streamhandler=logging.StreamHandler()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info(opt)

    batch_size = opt.batch_size
    classes = 1000
    num_training_samples = 1281167

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers

    save_frequency = opt.save_frequency
    if opt.save_dir and save_frequency:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_frequency = 0

    lr_decay = opt.lr_decay
    lr_decay_period = opt.lr_decay_period
    if opt.lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
    lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]
    num_batches = num_training_samples // batch_size

    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=opt.lr,
                    nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                    nepochs=opt.num_epochs - opt.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2)
    ])

    optimizer = 'nag'
    optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': lr_scheduler}

    net=MixNetsM(num_classes=classes)
    net.initialize(init=mx.init.MSRAPrelu(),ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    train_data, val_data, batch_fn = get_data_rec(opt, batch_size, num_workers)

    SML = gluon.loss.SoftmaxCrossEntropyLoss()

    train_loss = mx.metric.Loss()
    train_metric = mx.metric.Accuracy()

    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)


    for epoch in range(opt.num_epochs):
        tic=time.time()
        btic=time.time()

        train_loss.reset()
        train_metric.reset()
        train_data.reset()

        for i,batch in enumerate(train_data):
            data, label = batch_fn(batch, ctx)
            with autograd.record():
                outputs = [net(X) for X in data]
                loss = [SML(yhat, y) for yhat, y in zip(outputs, label)]

            for l in loss:
                l.backward()
            trainer.step(batch_size)

            train_loss.update(0, loss)
            train_metric.update(label, outputs)

            if opt.log_interval and not (i+1)%opt.log_interval:
                train_loss_name, train_loss_score = train_loss.get()
                train_metric_name, train_metric_score = train_metric.get()
                logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\t%s=%f\tlr=%f'%(
                                epoch, i, batch_size*opt.log_interval/(time.time()-btic),train_loss_name,train_loss_score,
                                train_metric_name, train_metric_score, trainer.learning_rate))
                btic = time.time()

        train_loss_name, train_loss_score = train_loss.get()
        train_metric_name, train_metric_score = train_metric.get()

        throughput = int(batch_size*i/(time.time()-tic))

        err_top1_val, err_top5_val = test(ctx, acc_top1,acc_top5,val_data,batch_fn,net)

        logger.info('[Epoch %d] training: %s=%f %s=%f'%(epoch, train_loss_name,train_loss_score,train_metric_name, train_metric_score))
        logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f'%(epoch, throughput, time.time()-tic))
        logger.info('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))

        if save_frequency and save_dir and (epoch+1) % save_frequency == 0:
            net.save_parameters('%s/imagenet-mixnets-%d.params' % (save_dir,epoch))

    if save_frequency and save_dir:
        net.save_parameters('%s/imagenet-mixnets-%d.params' % (save_dir,opt.num_epochs-1))

if __name__=='__main__':
    main()