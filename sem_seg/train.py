import argparse
import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
import indoor3d_util
from model import *

from pathlib import Path

import open3d as o3

import json

from config import *

idx = 0

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_path', default='model.ckpt', help='model checkpoint file path restore [default: model.ckpt]')
parser.add_argument('--model_save', default='model.ckpt', help='model checkpoint file path save [default: model.ckpt]')
parser.add_argument('--new_model', type=bool, default=False, help='Whether to start a new model or load existing from model_path')
parser.add_argument('--transfer_learning', type=bool, default=False, help='Whether to transfer up to the final classification layer from another model')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=str, default='', help='Which area to use for test, option: 1-6 [default: 6]')
FLAGS = parser.parse_args()

MODEL_PATH = FLAGS.model_path
MODEL_SAVE = FLAGS.model_save
NEW_MODEL = FLAGS.new_model
TRANSFER_MODEL = FLAGS.transfer_learning

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
NUM_POINT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096
NUM_CLASSES = 4

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,
                        batch * BATCH_SIZE,
                        DECAY_STEP,
                        DECAY_RATE,
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def capture_results(results, filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    for key, val in results.items():
        if key not in data:
            data[key] = []
        data[key].append(val)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def train():
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    with tf.Graph().as_default() as graph:
        with tf.Session(config=config) as sess:
            with tf.device('/gpu:'+str(GPU_INDEX)):
                pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
                is_training_pl = tf.placeholder(tf.bool, shape=())

                # Note the global_step=batch parameter to minimize.
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                batch = tf.Variable(0)
                bn_decay = get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

                # Get model and loss
                pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
                loss = get_loss(pred, labels_pl)
                tf.summary.scalar('loss', loss)

                correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
                accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
                tf.summary.scalar('accuracy', accuracy)

                # Get training operator
                learning_rate = get_learning_rate(batch)
                tf.summary.scalar('learning_rate', learning_rate)
                if OPTIMIZER == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
                elif OPTIMIZER == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=batch)

            saver = tf.train.Saver()

            # Add summary writers
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                      sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

            if TRANSFER_MODEL:
                init = tf.global_variables_initializer()
                sess.run(init, {is_training_pl: True})

                convs = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7']
                variables = {}
                for variable in tf.get_collection(tf.GraphKeys.VARIABLES):
                    if any([conv in variable.name for conv in convs]):
                        variables[variable.name.split(':')[0]] = graph.get_tensor_by_name(variable.name)
                saver = tf.train.Saver(var_list=variables)
                saver.restore(sess, tf.train.latest_checkpoint('log6'))
                print(f'Successly performed transfer learning from log6')
            elif NEW_MODEL:
                init = tf.global_variables_initializer()
                sess.run(init, {is_training_pl: True})
            else:
                saver.restore(sess, MODEL_PATH)
                log_string('Model restored.')
                print(f'Restored model from {MODEL_PATH}')

            ops = {'pointclouds_pl': pointclouds_pl,
                   'labels_pl': labels_pl,
                   'is_training_pl': is_training_pl,
                   'pred': pred,
                   'loss': loss,
                   'train_op': train_op,
                   'merged': merged,
                   'step': batch}

            test_data, test_label = load_test_data()

            for epoch in range(MAX_EPOCH):
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

#               for start, stop in zip(list(range(0, 29)), list(range(1, 30))):
                for start, stop in zip(list(range(0, 1)), list(range(1, 2))):
                    start *= 10
                    stop *= 10
                    train_data, train_label = load_train_data(start=start, stop=stop)
                    train_one_epoch(sess, ops, train_writer, train_data, train_label)

                if epoch % 1 == 0:
                    print(f'eval one epoch')
                    eval_one_epoch(sess, ops, test_writer, test_data, test_label)
                if epoch % 1 == 0:
                    # Need to redeclare this since var_list is set in
                    # case of TRANSFER_MODEL
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, os.path.join(LOG_DIR, MODEL_SAVE))
                    log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer, train_data, train_label):
    global idx

    is_training = True

    log_string('----')
    current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label)

#   write_to_point_cloud(current_data[0][:,0:3], current_data[0][:,3:6], idx)
#   colors = [indoor3d_util.g_label2color[label] for label in current_label[0]]
#   write_to_point_cloud(current_data[0][:,0:3], colors, idx)
#   idx += 1

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        if batch_idx % 100 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

#       points = []
#       colors = []
#       for data in current_data[start_idx:end_idx]:
#           print(f'data.shape = {data.shape}')
#           points.append(data[:,0:3])
#           colors.append(data[:,3:6])
#       points = np.concatenate(points)
#       colors = np.concatenate(colors)
#       print(f'points.shape = {points.shape}')
#       print(f'colors.shape = {colors.shape}')
#       write_to_point_cloud(points, colors, 100)
#       exit(0)

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val

    train_loss = loss_sum / float(num_batches)
    train_accuracy = total_correct / float(total_seen)
    train_results = {'train_loss': train_loss,
                     'train_accuracy': train_accuracy}
    capture_results(train_results, filename='results.json')

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

def eval_one_epoch(sess, ops, test_writer, test_data, test_label):
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string('----')
    current_data = test_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(test_label)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                                     feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE * NUM_POINT)
        loss_sum += (loss_val * BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

    eval_loss = loss_sum / float(total_seen/NUM_POINT)
    eval_accuracy = total_correct / float(total_seen)
    eval_avg_class_accuracy = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
    eval_results = {'eval_loss': eval_loss,
                    'eval_accuracy': eval_accuracy,
                    'eval_avg_class_accuracy': eval_avg_class_accuracy}
    capture_results(eval_results, filename='results.json')


def write_to_point_cloud(data, colors, idx):
    points = np.asarray(data, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    cloud = o3.PointCloud()
    cloud.points = o3.Vector3dVector(points)
    cloud.colors = o3.Vector3dVector(colors)
    print(f'writing to pointcloud{idx}.pcd')
    o3.write_point_cloud(f'pointcloud{idx}.pcd', cloud)
    o3.write_point_cloud(f'pointcloud{idx}.xyzrgb', cloud)


def load_train_data(start, stop):
    cwd = Path(os.path.abspath(os.path.dirname(__file__)))
    data = cwd / 'data'
    hdf5_dir = (data / 'hdf5_data').as_posix()
    train_hdf5_dir = os.path.join(hdf5_dir, 'train')
    files = provider.getDataFiles(os.path.join(train_hdf5_dir, 'all_files.txt'))

    data_batch_list = []
    label_batch_list = []

    count = 0
    for h5_filename in files[start:stop]:
        data_batch, label_batch = provider.load_h5(h5_filename)
#       data_batch = provider.jitter_point_cloud(data_batch)
        print(f'h5_filename = {h5_filename}')
        print(f'data_batch.shape = {data_batch.shape}')
        count += data_batch.shape[0]
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)

    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)
    print(f'data_batches.shape = {data_batches.shape}')
    print(f'label_batches.shape = {label_batches.shape}')

    print(f'count = {count}')
    train_idxs = list(range(0, count))

    train_data = data_batches[train_idxs,...]
    train_label = label_batches[train_idxs]
    print(f'train_data.shape, train_label.shape = {train_data.shape}, {train_label.shape}')
    return train_data, train_label


def load_test_data():
    """Load test data."""
    cwd = Path(os.path.abspath(os.path.dirname(__file__)))
    data = cwd / 'data'
    hdf5_dir = (data / 'hdf5_data').as_posix()
    test_hdf5_dir = os.path.join(hdf5_dir, 'test')
    test_files = provider.getDataFiles(os.path.join(test_hdf5_dir, 'all_files.txt'))

    data_batch_list = []
    label_batch_list = []
    count = 0
    for h5_filename in test_files:
        data_batch, label_batch = provider.load_h5(h5_filename)
#       data_batch = provider.jitter_point_cloud(data_batch)
        print(f'h5_filename = {h5_filename}')
        print(f'data_batch.shape = {data_batch.shape}')
        count += data_batch.shape[0]
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)

    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)
    print(f'data_batches.shape = {data_batches.shape}')
    print(f'label_batches.shape = {label_batches.shape}')

    test_idxs = list(range(0, count))
    print(f'len(test_idxs) = {len(test_idxs)}')

    test_data = data_batches[test_idxs,...]
    test_label = label_batches[test_idxs]
    print(f'test_data.shape, test_label.shape = {test_data.shape}, {test_label.shape}')
    return test_data, test_label


def main():
    train()
    LOG_FOUT.close()


if __name__ == "__main__":
    main()
