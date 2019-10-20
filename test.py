from config import get_config
from dataset import Div2KDataSet as DataSet
from dataset import DataIterator
from util import split, merge, img_save

import tensorflow as tf
import numpy as np
import argparse
import cv2

import model


# Configuration
config, _ = get_config()
np.random.seed(config.seed)
tf.set_random_seed(config.seed)

parser = argparse.ArgumentParser()
parser.add_argument('--data_from', type=str, default='img', choices=['img', 'h5'])
args = parser.parse_args()
data_from = args.data_from


def get_img(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)[..., ::-1]  # BGR to RGB


def main():

    if data_from == 'img':
        ds = DataSet(ds_path=config.test_dir,
                     ds_name="X4",
                     use_save=True,
                     save_type="to_h5",
                     save_file_name=config.test_dir + "DIV2K",
                     use_img_scale=False,
                     n_patch=config.patch_size,
                     n_images=100,
                     is_train=False)
    else:  # .h5 files
        ds = DataSet(ds_hr_path=config.test_dir + "DIV2K-hr.h5",
                     ds_lr_path=config.test_dir + "DIV2K-lr.h5",
                     use_img_scale=False,
                     n_patch=config.patch_size,
                     n_images=100,
                     is_train=False)

    # [0, 1] scaled images
    if config.patch_size > 0:
        hr, lr = ds.patch_hr_images, ds.patch_lr_images
    else:
        hr, lr = ds.hr_images, ds.lr_images

    lr_shape = lr.shape[1:]
    hr_shape = hr.shape[1:]
    print("[+] Loaded LR patch image ", lr.shape)
    print("[+] Loaded HR patch image ", hr.shape)

    di = DataIterator(lr, hr, config.batch_size)
    rcan_model = model.RCAN(lr_img_size=lr_shape[:-1],
                            hr_img_size=hr_shape[:-1],
                            batch_size=config.batch_size,
                            img_scaling_factor=config.image_scaling_factor,
                            n_res_blocks=config.n_res_blocks,
                            n_res_groups=config.n_res_groups,
                            res_scale=config.res_scale,
                            n_filters=config.filter_size,
                            kernel_size=config.kernel_size,
                            activation=config.activation,
                            use_bn=config.use_bn,
                            reduction=config.reduction,
                            optimizer=config.optimizer,
                            lr=config.lr,
                            lr_decay=config.lr_decay,
                            lr_decay_step=config.lr_decay_step,
                            momentum=config.momentum,
                            beta1=config.beta1,
                            beta2=config.beta2,
                            opt_eps=config.opt_epsilon,
                            tf_log=config.summary,
                            n_gpu=config.n_gpu,
                            )
    # gpu config
    gpu_config = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_config)

    with tf.Session(config=tf_config) as sess:
        # Initializing
        writer = tf.summary.FileWriter(config.test_log, sess.graph)
        sess.run(tf.global_variables_initializer())

        # Load model & Graph & Weights
        ckpt = tf.train.get_checkpoint_state(config.summary)
        if ckpt and ckpt.model_checkpoint_path:
            rcan_model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise OSError("[-] No checkpoint file found")

        # get result
        total_psnr = []
        total_ssim = []
        i = 0
        for x_lr, x_hr in di.iterate():
            x_lr = np.true_divide(x_lr, 255., casting='unsafe')

            psnr,ssim,summary,output = sess.run([rcan_model.psnr,rcan_model.ssim,rcan_model.merged,rcan_model.output],
                                                feed_dict={
                                                   rcan_model.x_lr: x_lr,
                                                   rcan_model.x_hr: x_hr,
                                                   rcan_model.lr: config.lr,
                                               })
            # output = np.reshape(output, rcan_model.hr_img_size)  # (384, 384, 3)
            writer.add_summary(summary)
            total_psnr.append(psnr)
            total_ssim.append(ssim)

            # save result
            patch = int(np.sqrt(config.patch_size))
            img_save(merge(output, (patch, patch)), './output/test'+'/%d.png'%i, use_inverse=False)
            print("%d images tested, "%i,"PSNR : {:.4f} SSIM : {:.4f}".format(psnr, ssim))
            i+=1
        print("total PSNR is {:.4f}, SSIM is {:.4f}".format(sum(total_psnr)/len(total_psnr),sum(total_ssim)/len(total_ssim)))


if __name__ == "__main__":
    main()
