import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
from ptrNet_oneStep import PointerNet
import argparse
import os
from datetime import datetime
from tqdm import trange
import json
from numpy import argsort


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int, default=128)
    parser.add_argument("--dropout", type = float, default=0.0)
    parser.add_argument("--num_units", type = int, default=512)
    parser.add_argument("--learning_rate", type = float, default=0.001)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--num_epoch", type=int, default=10000)
    parser.add_argument("--time_major", type = bool, default=False)
    parser.add_argument("--use_previous_ckpt", type=bool, default=False)
    parser.add_argument("--input_dim", type = int, default=1)
    parser.add_argument("--model_dir", type = str, default='model')
    parser.add_argument("--log_step", type = int, default=50)
    parser.add_argument("--max_to_keep", type = int, default=5)
    parser.add_argument("--num_gpus", type=int, default=1)
    return parser

def preprocess(x):
    src_trgt = tf.string_split([x], delimiter="output").values
    src, trgt = src_trgt[0], src_trgt[1]
    src = tf.string_to_number(tf.string_split([src]).values, tf.float32)
    trgt = tf.string_to_number(tf.string_split([trgt]).values, tf.int32)
    src = tf.reshape(src, shape=(-1,params.input_dim))
    # src = tf.concat([tf.fill([1,params.input_dim],2.0), src], axis = 0)
    trgt_in = tf.gather_nd(src, tf.reshape(trgt-1, shape=[-1, 1]))
    # trgt_out = tf.concat([trgt, tf.fill([1],0)], axis = 0)
    trgt_in = tf.concat([tf.fill([1, params.input_dim], 0.0), trgt_in], axis=0)
    return src, tf.shape(src)[0], trgt_in, trgt, tf.shape(trgt)[0]



def get_iterator(filename, params):
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.map(preprocess, num_parallel_calls=8)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000))
    dataset = dataset.padded_batch(params.batch_size, padded_shapes=([None,params.input_dim], [],[None,params.input_dim], [None], []))
    return dataset.make_one_shot_iterator()

def get_infer_iterator(filename, params):
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.map(preprocess, num_parallel_calls=1)
    dataset = dataset.padded_batch(params.batch_size,
                                   padded_shapes=([None, params.input_dim], [], [None, params.input_dim], [None], []))
    return dataset.make_one_shot_iterator()

#
# parser = get_parser()
# params, unparsed = parser.parse_known_args()
# iter =get_infer_iterator("data/skyline1000-10-2d.txt", params)
#
# with tf.Session() as sess:
#     print(sess.run(iter.get_next()))
#


if __name__ =='__main__':
    parser = get_parser()
    params, unparsed = parser.parse_known_args()
    print(params.use_previous_ckpt)

    if params.mode == 'train':

        train_graph = tf.Graph()
        eval_graph = tf.Graph()

        with train_graph.as_default(),tf.container("train"):
            train_iter = get_iterator("data/sort10000-100.txt", params)
            train_model = PointerNet(train_iter, params, 'train')
        with eval_graph.as_default(), tf.container("eval"):
            eval_iter = get_iterator("data/sort1000-100.txt", params)
            eval_model = PointerNet(eval_iter, params, 'eval')

        if params.use_previous_ckpt:
            train_model.load_model()
        else:
            params.model_dir='model'
            params.model_dir = os.path.join(params.model_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))
            tf.logging.info('model_dir: {}'.format(params.model_dir))

        max_recall = 0
        for i in trange(params.num_epoch):
            result = train_model.train_step()
            if (i + 1) % 50 == 0:
                sum_recall = 0
                count = 0
                max_recall = 0
                min_recall= float('inf')

                train_model.save(result['global_step'])

                eval_model.load_model()
                start_time = datetime.now()
                test = eval_model.eval()
                tf.logging.info("time elapsed: {}".format(datetime.now() - start_time))
                tf.logging.info("loss: {}".format(test['loss']))
                for j in range(len(test['true'])):
                    tf.logging.info("true:      {}".format(test['true'][j]))
                    tf.logging.info("predicted: {}".format(test['predicted'][j]))
                    # true_set = set(test['true'][j]) - set([0])
                    # pred_set = set(test['predicted'][j])
                    # recall = (len(true_set) - len(true_set - pred_set)) / len(true_set)
                    # tf.logging.info("%d / %d = %.2f" % (len(true_set) - len(true_set - pred_set), len(true_set),
                    #                                    recall))
                    # sum_recall += recall
                    # count += 1
                    # max_recall = max(max_recall, recall)
                    # min_recall = min(min_recall, recall)

                # avg_recall = sum_recall / count
                # tf.logging.info("%d average recall = %.2f max = %.2f min = %.2f" % (i+1, avg_recall, max_recall, min_recall))

    elif params.mode == 'infer':
        infer_graph = tf.Graph()
        with infer_graph.as_default(), tf.container('infer'):
            infer_iter = get_infer_iterator("data/CPair-100-10-2d.txt", params)
            infer_model = PointerNet(infer_iter, params, 'infer')

        infer_model.load_model()
        st = datetime.now()
        while True:
            try:
                count = 0
                sum_recall = 0
                start_time = datetime.now()
                result = infer_model.infer()
                tf.logging.info("time elapsed: {}".format(datetime.now() - start_time))
                for i in range(len(result['true'])):
                    tf.logging.info("true:      {}".format(result['true'][i]))
                    tf.logging.info("predicted: {}".format(result['predicted'][i]))

                    true_set = set(result['true'][i]) - set([0])
                    pred_set = set(result['predicted'][i])
                    recall = (len(true_set) - len(true_set - pred_set)) / len(true_set)
                    tf.logging.info("%d / %d = %.2f" % (len(true_set) - len(true_set - pred_set), len(true_set),
                                                        recall))
                    sum_recall += recall
                    count+=1
                avg_recall = sum_recall / count
                tf.logging.info(
                    "%d average recall = %.2f" % (i + 1, avg_recall))
            except tf.errors.OutOfRangeError:
                print("total time elapsed: {}".format(datetime.now() - st))
                exit()

    param_path = os.path.join(params.model_dir, "params.json")
    with open(param_path, 'w') as f:
        json.dump(params.__dict__, f, indent=4, sort_keys=True)


