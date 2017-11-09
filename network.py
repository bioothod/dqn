import tensorflow as tf
import numpy as np

class network(object):
    def __init__(self, config):
        self.summary_writer = config.get('summary_writer')
        self.summary_update_steps = config.get('summary_update_steps')
        self.train_steps = 0

        self.summary_all = []

        input_shape = config.get('input_shape')
        state_steps = config.get('state_steps')
        actions = config.get('actions')
        states = tf.placeholder(tf.float32, [None, state_steps, input_shape[0], input_shape[1]], name='states')
        qvals = tf.placeholder(tf.float32, [None, actions], name='qvals')

        input_layer = tf.reshape(states, [-1, input_shape[0], input_shape[1], state_steps])

        prelu_alpha = 0.0001

        c1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=8, strides=4, padding='same',
                activation=tf.contrib.keras.layers.PReLU(alpha_initializer=tf.constant_initializer(prelu_alpha)))
        c2 = tf.layers.conv2d(inputs=c1, filters=32, kernel_size=4, strides=2, padding='same',
                activation=tf.contrib.keras.layers.PReLU(alpha_initializer=tf.constant_initializer(prelu_alpha)))

        flat = tf.reshape(c2, [-1, np.prod(c2.get_shape().as_list()[1:])])

        kinit = tf.contrib.layers.xavier_initializer()

        self.dense = tf.layers.dense(inputs=flat, units=config.get('dense_layer_units'),
                activation=tf.contrib.keras.layers.PReLU(alpha_initializer=tf.constant_initializer(prelu_alpha)),
                use_bias=True, name='dense_layer',
                kernel_initializer=kinit,
                bias_initializer=tf.random_normal_initializer(0, 0.1))

        self.output = tf.layers.dense(inputs=self.dense, units=actions, use_bias=True, name='output_layer',
                kernel_initializer=kinit,
                bias_initializer=tf.random_normal_initializer(0, 0.1))

        self.loss = tf.reduce_mean(tf.square(self.output - qvals))
        self.summary_all.append(tf.summary.scalar('loss', self.loss))

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = 0.00001 + tf.train.exponential_decay(config.get('learning_rate_start'), self.global_step,
            config.get('learning_rate_step'), config.get('learning_rate_factor'), staircase=True)

        self.summary_all.append(tf.summary.scalar('learning_rate', self.learning_rate))
        self.summary_all.append(tf.summary.scalar('global_step', self.global_step))

        self.summary_merged = tf.summary.merge(self.summary_all)

        opt = tf.train.RMSPropOptimizer(self.learning_rate,
                decay=0.99,
                name='optimizer')
        self.optimizer_step = opt.minimize(self.loss, global_step=self.global_step)

        config=tf.ConfigProto(
                intra_op_parallelism_threads = 8,
                inter_op_parallelism_threads = 8,
            )
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        self.session.run(init)

    def train(self, states, qvals):
        ret = self.session.run([self.summary_merged, self.optimizer_step, self.global_step], feed_dict={
                'states:0': states,
                'qvals:0': qvals,
            })

        self.train_steps += 1
        if self.train_steps % self.summary_update_steps == 0:
            summary = ret[0]
            global_step = ret[2]
            self.summary_writer.add_summary(summary, global_step)

    def predict(self, states):
        ret = self.session.run([self.output], feed_dict={
                'states:0': states,
            })

        return ret[0]
