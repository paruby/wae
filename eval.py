import numpy as np
import os
import sys
import logging
import argparse
from eval_utils import model_details
from eval_utils import compute_metrics
from eval_utils import compute_inception_stats
from eval_utils import compute_blurriness
from datahandler import DataHandler

parser = argparse.ArgumentParser()
parser.add_argument("--path", default='..',
                    help='Root path for the experiment folders')
parser.add_argument("--recompute", default='no',
                    help='If yes then dont reuse any pre-existing computations')
parser.add_argument("--work_dir", default='.')
parser.add_argument("--exp_name", default='',
                    help='Substring specifying the name of experiments')
parser.add_argument("--exp_dir_prefix", default='')
parser.add_argument("--num_samples", default=1000,
                    help='Number of samples for evaluating FID',
                    type=int)
parser.add_argument("--batch_size", default=500,
                    type=int)

FLAGS = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    """
    # 1. Go and look for the checkpoint folders in the specified paths
        a. Specify the root directory
        b. Select all subdirectories containing specified substring in the name
    # 2. Check if the latest stored model was already FID-scored
    # 3. If not --- compute its FIDs and save them in the separate files
    # 4. Optionnaly aggregate all the FIDs together in a nice plot
    """
    assert FLAGS.num_samples % FLAGS.batch_size == 0, 'Number of samples should be factor of batch_size'
    path = FLAGS.path
    match = FLAGS.exp_name
    prefix = FLAGS.exp_dir_prefix
    to_process = []
    models_fid = {}
    # Get names of all entries in root path
    names = os.listdir(path)
    for name in names:
        if not (prefix in name and match in name):
            continue
        if not os.path.isdir(os.path.join(path, name)):
            continue
        chdir = os.path.join(path, name, 'checkpoints')
        if not os.path.isdir(chdir):
            continue
        model_names = os.listdir(chdir)
        model_names = [n for n in model_names if '.meta' in n and n[0] != '.']
        if len(model_names) == 0:
            continue
        key_f = lambda s: os.path.getctime(os.path.join(chdir, s))
        newest_file = max(model_names, key=key_f)
        newest_file = newest_file.split('.')[0]
        fid_val_file = os.path.join(chdir, newest_file + '.fid' + str(FLAGS.num_samples) + '.val')
        if os.path.exists(fid_val_file):
            # The model was already evaluated
            if FLAGS.recompute == 'no':
                with open(fid_val_file, 'r') as f:
                    lines = f.readlines()
                fid_gen = float(lines[0].split('FID=')[1].split(',')[0])
                fid_reconstr = float(lines[0].split('FID=')[2].split(',')[0])
                logging.error('Model %s already processed' % chdir)
                logging.error('FID=%f, computed using %d samples' % (fid, FLAGS.num_samples))
                models_fid[os.path.join(path, name)] = (fid_gen, fid_reconstr)
                continue
            else:
                os.remove(fid_val_file)
        fid_temp_file = os.path.join(chdir, newest_file + '.fid' + str(FLAGS.num_samples) + '.tmp')
        if os.path.exists(fid_temp_file):
            # The model is being evaluated right now
            continue
        to_process.append((os.path.join(path, name), chdir, newest_file))

    if len(to_process) > 0:
        logging.error('Going to process the following models:')
        for tup in to_process:
            logging.error(os.path.join(tup[1], tup[2]))
    else:
        logging.error('Everything already processed.')

    # Checking for the FID stats files for the required datasets
    datasets = {}
    for tup in to_process:
        opts = model_details(os.path.join(tup[0], 'params.txt'),
                             FLAGS.work_dir)
        dataset = opts['dataset']
        if dataset in datasets:
            continue
        datasets[dataset] = True
        data_path = opts['data_dir']
        stats_file = os.path.join(
            data_path,
            dataset + '.fidstats' + str(FLAGS.num_samples))
        if os.path.exists(stats_file + '.npz'):
            if FLAGS.recompute == 'yes':
                os.remove(stats_file + '.npz')
            else:
                continue
        logging.error('Computing FID stats for the %s dataset' % dataset)
        data = DataHandler(opts)
        random_ids = np.random.choice(data.num_points,
                               FLAGS.num_samples, replace=False)
        data_sample = data.data[random_ids]
        random_test_ids = np.random.choice(len(data.test_data),
                               FLAGS.num_samples, replace=False)
        data_test_sample = data.test_data[random_test_ids]
        if opts['input_normalize_sym']:
            data_sample = data_sample / 2. + 0.5
            data_test_sample = data_test_sample / 2. + 0.5
        data_sharpness = compute_blurriness(
            data_sample[:min(500, FLAGS.num_samples)])
        np.save(os.path.join(data_path, dataset + '.sharp' + \
                str(FLAGS.num_samples) + '.npy'), data_sharpness)
        logging.error('Data sharpness = %f' % np.mean(data_sharpness))

        mu, cov = compute_inception_stats(data_sample, FLAGS.batch_size)
        np.savez(stats_file, mu=mu, cov=cov)

    for tup in to_process:
        model = os.path.join(tup[1], tup[2])
        logging.error('Processing %s' % model)
        with open(os.path.join(tup[1], tup[2] + '.fid' + str(FLAGS.num_samples) + '.tmp'), 'w') as f:
            f.write('In progress...')
        # try:
        fid = compute_metrics(tup, FLAGS.num_samples, FLAGS.work_dir, FLAGS.batch_size, FLAGS.recompute)
        # except:
        #     logging.error("Unexpected error:", sys.exc_info()[0])
        #     logging.error('Failed to process %s' % model)
        #     os.remove(os.path.join(tup[1], tup[2] + '.fid' + str(FLAGS.num_samples) + '.tmp'))
        #     continue
        if fid == None:
            logging.error('Failed to process %s' % model)
            os.remove(os.path.join(tup[1], tup[2] + '.fid' + str(FLAGS.num_samples) + '.tmp'))
            continue
        fid_gen, fid_reconstr_train, fid_reconstr_test = fid
        with open(os.path.join(tup[1], tup[2] + '.fid' + str(FLAGS.num_samples) + '.txt'), 'a') as f:
            f.write('samples FID=%f, train reconstruction FID=%f, test reconstruction FID=%f, computed using %d samples' % \
                    (fid_gen, fid_reconstr_train, fid_reconstr_test, FLAGS.num_samples))
        with open(os.path.join(tup[1], tup[2] + '.fid' + str(FLAGS.num_samples) + '.val'), 'w') as f:
            f.write('samples FID=%f, train reconstruction FID=%f, test reconstruction FID=%f, computed using %d samples' % \
                    (fid_gen, fid_reconstr_train, fid_reconstr_test, FLAGS.num_samples))
            logging.error('samples FID=%f, train reconstruction FID=%f, test reconstruction FID=%f, computed using %d samples' % \
                    (fid_gen, fid_reconstr_train, fid_reconstr_test, FLAGS.num_samples))
        models_fid[tup[0]] = (fid_gen, fid_reconstr_train, fid_reconstr_test)
        os.remove(os.path.join(tup[1], tup[2] + '.fid' + str(FLAGS.num_samples) + '.tmp'))

    logging.error(models_fid)

main()
