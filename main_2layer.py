import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
from ptrNetPar import PointerNet
import argparse
import os
from datetime import datetime
from tqdm import trange
from multi_gpu import build_multi_gpu_model

AVG_FILE = "avg10000-64-32-2layer_1GPU.txt"

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int, default=64)
    parser.add_argument("--dropout", type = float, default=0.0)
    parser.add_argument("--num_units", type = int, default=32)
    parser.add_argument("--learning_rate", type = float, default=0.001)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--num_epoch", type=int, default=10000)
    parser.add_argument("--time_major", type = bool, default=False)
    parser.add_argument("--use_previous_ckpt", type=bool, default=False)
    parser.add_argument("--input_dim", type = int, default=2)
    parser.add_argument("--model_dir", type = str, default='model')
    parser.add_argument("--log_step", type = int, default=50)
    parser.add_argument("--max_to_keep", type = int, default=5)
    parser.add_argument("--num_gpus", type=int, default=1)
    return parser

def preprocess(x):
    src_trgt = tf.string_split([x], delimiter="output").values
    src, layer1, layer2 = src_trgt[0], src_trgt[1], src_trgt[2]
    layer1 = tf.string_to_number(tf.string_split([layer1]).values, tf.int32)
    layer2 = tf.string_to_number(tf.string_split([layer2]).values, tf.int32)
    src = tf.string_to_number(tf.string_split([src]).values, tf.float32)
    trgt = tf.concat([layer1, layer2], 0)
    src = tf.reshape(src, shape=(-1,2))
    src = tf.concat([tf.fill([1,params.input_dim],2.0), src], axis = 0)
    trgt_in = tf.gather_nd(src, tf.reshape(trgt, shape=[-1, 1]))
    trgt_out = tf.concat([trgt, tf.fill([1],0)], axis = 0)
    trgt_in = tf.concat([tf.fill([1, params.input_dim], 0.0), trgt_in], axis=0)
    return src, tf.shape(src)[0], trgt_in, trgt_out, tf.shape(trgt_in)[0], layer1



def get_iterator(filename, params):
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.map(preprocess, num_parallel_calls=4)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000))
    dataset = dataset.padded_batch(params.batch_size, padded_shapes=([None, params.input_dim], [],[None,params.input_dim], [None], [], [None]))
    return dataset.make_one_shot_iterator()

def get_infer_iterator(filename, params):
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.map(preprocess, num_parallel_calls=4)
    dataset = dataset.padded_batch(params.batch_size,
                                   padded_shapes=([None, params.input_dim], [], [None, params.input_dim], [None], [], [None]))
    return dataset.make_one_shot_iterator()

#
# parser = get_parser()
# params, unparsed = parser.parse_known_args()
#
# iter = get_iterator("data/skyline1000-10-2d-2layer.txt", params)
# with tf.Session() as sess:
#     print(sess.run(iter.get_next()))



if __name__ =='__main__':
    parser = get_parser()
    params, unparsed = parser.parse_known_args()
    f = open(AVG_FILE,'w')
    f.close()

    if params.mode == 'train':

        train_graph = tf.Graph()
        eval_graph = tf.Graph()

        with train_graph.as_default(),tf.container("train"),tf.device('/cpu:0'):
            train_iter = get_iterator("data/skyline10000-10000-2d-2layer.txt", params)
            train_op, loss= build_multi_gpu_model(params, train_iter, is_2layer=True)
            train_saver = tf.train.Saver(tf.global_variables(), max_to_keep=params.max_to_keep)
            init = tf.global_variables_initializer()
        with eval_graph.as_default(), tf.container("eval"), tf.device('/cpu:0'):
            eval_iter = get_iterator("data/skyline1000-10000-2d-2layer.txt", params)
            eval_model = PointerNet(params, 'eval')
            src, src_len, trgt_in, trgt_out, trgt_len, layer1= eval_iter.get_next()
            next = (src, src_len, trgt_in, trgt_out, trgt_len)
            eval_model.build_graph(next)
            eval_loss = eval_model.compute_loss()
            eval_saver = tf.train.Saver()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        train_sess = tf.Session(config=config, graph=train_graph)
        eval_sess = tf.Session(config=config, graph=eval_graph)



        if params.use_previous_ckpt:
            latest_ckpt = tf.train.latest_checkpoint(params.model_dir)
            train_saver.restore(train_sess, latest_ckpt)
        else:
            train_sess.run(init)
            params.model_dir='model'
            params.model_dir = os.path.join(params.model_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))
            tf.logging.info('model_dir: {}'.format(params.model_dir))

        max_recall = 0
        for i in trange(params.num_epoch):
            result = train_sess.run({"update":train_op, "loss":loss})
            if (i + 1) % 50 == 0:
                train_saver.save(sess=train_sess, save_path=params.model_dir, global_step=tf.train.get_global_step(graph=train_graph))


                eval_saver.restore(sess=eval_sess, save_path= tf.train.latest_checkpoint(params.model_dir))
                test = eval_sess.run({"loss":eval_loss, "true": eval_model.trgt_out, 'predicted':eval_model.sample_ids, 'layer1': layer1})
                tf.logging.info("loss: {}".format(test['loss']))
                sum_recall = 0
                count = 0
                for j in range(len(test['true'])):
                    tf.logging.info("layer1:    {}".format(test['layer1'][j]))
                    tf.logging.info("true:      {}".format(test['true'][j]))
                    tf.logging.info("predicted: {}".format(test['predicted'][j]))
                    true_set = set(test['layer1'][j]) - set([0])
                    pred_set = set(test['predicted'][j])
                    recall = (len(true_set) - len(true_set - pred_set)) / len(true_set)
                    tf.logging.info("%d / %d = %.2f" % (len(true_set) - len(true_set - pred_set), len(true_set),
                                                        recall))
                    sum_recall += recall
                    count += 1
                    avg_recall = sum_recall / count
                    max_recall = max(max_recall, avg_recall)
                tf.logging.info("%d average recall = %.2f max = %.2f" % (i + 1, avg_recall, max_recall))
                with open(AVG_FILE, "a+") as fp:
                    fp.write("%d, %.2f\n" % (i + 1, avg_recall))


    elif params.mode == 'infer':
        infer_graph = tf.Graph()
        with infer_graph.as_default(), tf.container('infer'):
            infer_iter = get_infer_iterator("data/1k-100.txt", params)
            infer_model = PointerNet(infer_iter, 'infer')

        infer_model.load_model()
        while True:
            try:
                start_time = datetime.now()
                result = infer_model.infer()
                tf.logging.info(datetime.now()-start_time)
                for i in range(len(result['true'])):
                    tf.logging.info("true:      {}".format(result['true'][i]))
                    tf.logging.info("predicted: {}".format(result['predicted'][i]))
            except tf.errors.OutOfRangeError:
                exit()
