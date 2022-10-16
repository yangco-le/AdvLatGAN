import os
from torch.autograd import Variable
import torch
import torch.optim as optim
from absl import flags, app
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
# from tensorboardX import SummaryWriter
from tqdm import trange
from pytorch_gan_metrics import get_inception_score_and_fid
# 
from source.utils import generate_imgs, infiniteloop, set_seed
import source.losses as losses

import numpy as np
from prdc import compute_prdc
from pytorch_gan_metrics.core import get_inception_feature

from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image
from torchvision.transforms.functional import to_tensor

from argparse import ArgumentParser

FLAGS = flags.FLAGS
# advlatgan parameters
flags.DEFINE_integer('num_iter', 20, "number of adversarial iterations")
flags.DEFINE_float('eps', 0.01, "epsilon of I-FGSM")
# model and training
flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'stl10', 'lsn', 'celeba', 'imagenet'], "dataset")
flags.DEFINE_enum('model', 'dcgan', ['dcgan', 'wgan', 'wgangp', 'sngan'], "model")
flags.DEFINE_integer('batch_size', 128, "batch size")
flags.DEFINE_integer('img_size', 64, "image size")
flags.DEFINE_enum('arch', 'res32', ['res32', 'res48', 'cnn32', 'cnn48', 'cnn64', 'cnn128'], "architecture")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_enum('loss', 'hinge', ['bce', 'hinge', 'was', 'softplus'], "loss function")
flags.DEFINE_integer('seed', 0, "random seed")
# logging
flags.DEFINE_string('fid_cache', './stats/cifar10_stats.npz', 'FID cacuhe')
# generate
flags.DEFINE_string('pretrain', None, 'path to test model')
flags.DEFINE_integer('num_images', 50000, 'the number of generated images')

device = torch.device('cuda:0')

def config_model(FLAGS):
    if FLAGS.model == 'dcgan':
        import source.models.dcgan as models
    elif FLAGS.model == 'wgan' or FLAGS.model == 'wgangp':
        import source.models.wgangp as models
    elif FLAGS.model == 'sngan':
        import source.models.sngan as models

    net_G_models = {
        'res32': models.ResGenerator32,
        'res48': models.ResGenerator48,
        'cnn32': models.Generator32,
        'cnn48': models.Generator48,
        'cnn64': models.Generator64,
        'cnn128': models.Generator128
    }

    net_D_models = {
        'res32': models.ResDiscriminator32,
        'res48': models.ResDiscriminator48,
        'cnn32': models.Discriminator32,
        'cnn48': models.Discriminator48,
        'cnn64': models.Discriminator64,
        'cnn128': models.Discriminator128
    }

    loss_fns = {
        'bce': losses.BCEWithLogits,
        'hinge': losses.Hinge,
        'was': losses.Wasserstein,
        'softplus': losses.Softplus
    }
    
    return net_G_models, net_D_models, loss_fns

def generate_sampling_shift(net_G, net_D, device, z_dim=128, size=5000, batch_size=128):
    net_G_models, net_D_models, loss_fns = config_model(FLAGS)
    net_G.eval()
    net_D.eval()
    imgs = []
    loss_fn = loss_fns[FLAGS.loss]()
    for start in trange(0, size, batch_size, desc='Evaluating', ncols=0, leave=False):
        end = min(start + batch_size, size)
        z = torch.randn(end - start, z_dim).to(device)
        z_adv = Variable(z.data, requires_grad=True).to(device)
        for j in range(FLAGS.num_iter):
            net_G.zero_grad()
            net_D.zero_grad()
            if z_adv.grad is not None:
                z_adv.grad.data.fill_(0)
            loss = loss_fn(net_D(net_G(z_adv)))
            loss.backward()
            z_adv.grad.sign_()
            z_adv = z_adv - z_adv.grad * FLAGS.eps
            z_adv = Variable(z_adv.data, requires_grad=True).to(device)
        imgs.append(net_G(z_adv).cpu().data)
    net_G.train()
    imgs = torch.cat(imgs, dim=0)
    imgs = (imgs + 1) / 2
    return imgs

def test():
    net_G_models, net_D_models, loss_fns = config_model(FLAGS)
    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_D = net_D_models[FLAGS.arch]().to(device)
    # nearest_k, fea_dim = 10, 2048

    # tar_dir = "stats/lsun64"

    # # print(ImageDataset(tar_dir).__len__())

    # real_features, = get_inception_feature(DataLoader(ImageDataset(tar_dir)), dims=[fea_dim])
    # # real_features = np.load("stats/fea_lsun64.npz")['real_features']
    # real_features = real_features[0:10000]

    model = torch.load(FLAGS.pretrain)
    net_G.load_state_dict(model["net_G"])
    net_D.load_state_dict(model["net_D"])

    fake_imgs = generate_sampling_shift(net_G, net_D,device, FLAGS.z_dim, 50000, FLAGS.batch_size)

    # fake_features, = get_inception_feature(fake_imgs, dims=[fea_dim])
    # metrics = compute_prdc(real_features=real_features, fake_features=fake_features, nearest_k=nearest_k)
    # print(metrics)

    IS, FID = get_inception_score_and_fid(fake_imgs, FLAGS.fid_cache, verbose=True)
    print("inception_score: ",IS,"  FID:  ",FID)




def main(argv):
    set_seed(FLAGS.seed)
    test()


if __name__ == '__main__':
    app.run(main)
