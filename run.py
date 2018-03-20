import os
import sys
import logging
import argparse
import configs
from wae import WAE
from datahandler import DataHandler
import utils
import json

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default='mnist_small',
                    help='dataset [mnist/celebA/dsprites]')
parser.add_argument("--zdim",
                    help='dimensionality of the latent space',
                    type=int)
parser.add_argument("--lr",
                    help='ae learning rate',
                    type=float)
parser.add_argument("--z_test",
                    help='method of choice for verifying Pz=Qz [mmd/gan]')
parser.add_argument("--wae_lambda", help='WAE regularizer', type=int)
parser.add_argument("--work_dir")
parser.add_argument("--lambda_schedule",
                    help='constant or adaptive')
parser.add_argument("--enc_noise",
                    help="type of encoder noise:"\
                         " 'deterministic': no noise whatsoever,"\
                         " 'gaussian': gaussian encoder,"\
                         " 'implicit': implicit encoder,"\
                         " 'add_noise': add noise before feeding "\
                         "to deterministic encoder")
parser.add_argument("--smart_cost", type=bool,
                    help='Use smart costs')
parser.add_argument("--patch_var_w", type=float,
                    help='weight of patch_var cost term')
parser.add_argument("--l2sq_w", type=float,
                    help='weight of l2sq cost term')
parser.add_argument("--sylvain_adv_c_w", type=float,
                    help='weight of sylvains adv cost term')
parser.add_argument("--sylvain_emb_c_w", type=float,
                    help='weight of sylvains adv cost term')
parser.add_argument("--adv_c_num_units", type=int,
                    help='number of filters to use in adversarial cost')
parser.add_argument("--adv_c_patches_size", type=int,
                    help='size of filters to use in adversarial cost')

parser.add_argument("--e_num_filters", type=int,
                    help='Number of filters to use in encoder')
parser.add_argument("--g_num_filters", type=int,
                    help='Number of filters to use in generator')
parser.add_argument("--adv_use_sq", type=bool,
                    help='Whether to use square pixel values as inputs to adversarial cost')

parser.add_argument("--celebA_crop", type=str,
                    help='Method of cropping / data preprocessing to apply to celebA dataset. closecrop/closecrop_randomshift/resizecrop')


FLAGS = parser.parse_args()

def main():

    if FLAGS.exp == 'celebA':
        opts = configs.config_celebA
    elif FLAGS.exp == 'celebA_small':
        opts = configs.config_celebA_small
    elif FLAGS.exp == 'celebA_ae_patch_var':
        opts = configs.config_celebA_ae_patch_var
    elif FLAGS.exp == 'celebA_sylvain_adv':
        opts = configs.config_celebA_sylvain_adv
    elif FLAGS.exp == 'celebA_adv':
        opts = configs.config_celebA_adv
    elif FLAGS.exp == 'mnist':
        opts = configs.config_mnist
    elif FLAGS.exp == 'mnist_small':
        opts = configs.config_mnist_small
    elif FLAGS.exp == 'dsprites':
        opts = configs.config_dsprites
    elif FLAGS.exp == 'grassli':
        opts = configs.config_grassli
    elif FLAGS.exp == 'grassli_small':
        opts = configs.config_grassli_small
    elif FLAGS.exp == 'cifar':
        opts = configs.config_cifar
    else:
        assert False, 'Unknown experiment configuration'

    if FLAGS.zdim is not None:
        opts['zdim'] = FLAGS.zdim
    if FLAGS.lr is not None:
        opts['lr'] = FLAGS.lr
    if FLAGS.z_test is not None:
        opts['z_test'] = FLAGS.z_test
    if FLAGS.lambda_schedule is not None:
        opts['lambda_schedule'] = FLAGS.lambda_schedule
    if FLAGS.work_dir is not None:
        opts['work_dir'] = FLAGS.work_dir
    if FLAGS.wae_lambda is not None:
        opts['lambda'] = FLAGS.wae_lambda
    if FLAGS.celebA_crop is not None:
        opts['lambda'] = FLAGS.celebA_crop
    if FLAGS.enc_noise is not None:
        opts['e_noise'] = FLAGS.enc_noise
    if FLAGS.e_num_filters is not None:
        opts['e_num_filters'] = FLAGS.e_num_filters
    if FLAGS.g_num_filters is not None:
        opts['g_num_filters'] = FLAGS.g_num_filters
    if FLAGS.smart_cost is True:
        opts['cost'] = []
        if FLAGS.patch_var_w is not None:
            opts['cost'].append(('patch_variances', FLAGS.patch_var_w))
        if FLAGS.l2sq_w is not None:
            opts['cost'].append(('l2sq', FLAGS.l2sq_w))
        if FLAGS.sylvain_adv_c_w is not None and FLAGS.sylvain_emb_c_w is not None:
            adv_c_w = FLAGS.sylvain_adv_c_w
            emb_c_w = FLAGS.sylvain_emb_c_w
            opts['cost'].append(('_sylvain_recon_loss_using_disc_conv', [adv_c_w, emb_c_w]))
            opts['cross_p_w'] = 0
            opts['diag_p_w'] = 0
        if FLAGS.adv_c_num_units is not None:
            opts['adv_c_num_units'] = FLAGS.adv_c_num_units
        if FLAGS.adv_c_patches_size is not None:
            opts['adv_c_patches_size'] = FLAGS.adv_c_patches_size
        if FLAGS.adv_use_sq is not None:
            opts['adv_use_sq'] = FLAGS.adv_use_sq


    if opts['verbose']:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    utils.create_dir(opts['work_dir'])
    utils.create_dir(os.path.join(opts['work_dir'],
                     'checkpoints'))
    # Dumping all the configs to the text file
    with utils.o_gfile((opts['work_dir'], 'params.txt'), 'w') as text:
        text.write('Parameters:\n')
        for key in opts:
            text.write('%s : %s\n' % (key, opts[key]))

    # Loading the dataset

    data = DataHandler(opts)
    assert data.num_points >= opts['batch_size'], 'Training set too small'

    # Training WAE

    wae = WAE(opts)
    wae.train(data)

main()
