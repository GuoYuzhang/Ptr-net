import tensorflow as tf
from ptrNetPar import PointerNet

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):

    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def build_multi_gpu_model(params, iter, is_2layer = False):
    with tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
        opt = tf.train.AdamOptimizer(params.learning_rate)
        train_iter = iter
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(params.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % (i)) as scope:
                        model = PointerNet(params, 'train')
                        if is_2layer:
                            src, src_len, trgt_in, trgt_out, trgt_len, layer1 = train_iter.get_next()
                            next = (src, src_len, trgt_in, trgt_out, trgt_len)
                        else:
                            next = train_iter.get_next()
                        model.build_graph(next)
                        loss = model.compute_loss()
                        tf.get_variable_scope().reuse_variables()

                        gradients = tf.gradients(loss, tf.trainable_variables())
                        clipped_grad,_= tf.clip_by_global_norm(gradients, 5)

                        clipped_grad = zip(clipped_grad, tf.trainable_variables())

                        # Keep track of the gradients across all towers.
                        tower_grads.append(clipped_grad)

        grads = average_gradients(tower_grads)

        train_op = opt.apply_gradients(grads, global_step=global_step)
    return train_op, loss