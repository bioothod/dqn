import tensorflow as tf
import numpy as np

import collections

def get_param_name(s):
    return s.split('/', 1)[1].replace('/', 'X').split(':')[0]
def get_scope_name(s):
    return s.split('/')[0].split(':')[0]
def get_transform_placeholder_name(s):
    return get_param_name(s) + '_ext'

class network(object):
    def __init__(self, scope, config):
        self.scope = scope
        self.summary_writer = config.get('summary_writer')
        self.summary_update_steps = config.get('summary_update_steps')
        self.train_steps = 0

        self.summary_all = []

        input_shape = config.get('input_shape')
        state_steps = config.get('state_steps')
        actions = config.get('actions')
        states = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], input_shape[2]*state_steps], name='states')
        qvals = tf.placeholder(tf.float32, [None, actions], name='qvals')

        rewards = tf.placeholder(tf.float32, [None], name='episode_rewards')
        rewards_mva = tf.placeholder(tf.float32, [], name='episode_rewards_mva')
        self.last_rewards = collections.deque(maxlen=100)
        rewards_summary = []
        rewards_summary.append(tf.summary.scalar("episode_rewards_mva", rewards_mva))
        rewards_summary.append(tf.summary.scalar("episode_rewards_mean", tf.reduce_mean(rewards)))
        rewards_summary.append(tf.summary.scalar("episode_rewards_max", tf.reduce_max(rewards)))
        rewards_summary.append(tf.summary.scalar("episode_rewards_min", tf.reduce_min(rewards)))
        self.update_rewards_ops = tf.summary.merge(rewards_summary)

        input_layer = tf.reshape(states, [-1, input_shape[0], input_shape[1], input_shape[2]*state_steps])

        c1 = tf.layers.conv2d(inputs=states, filters=32, kernel_size=8, strides=4, padding='same',
                activation=tf.nn.relu)
        c2 = tf.layers.conv2d(inputs=c1, filters=64, kernel_size=4, strides=2, padding='same',
                activation=tf.nn.relu)

        flat = tf.reshape(c2, [-1, np.prod(c2.get_shape().as_list()[1:])])

        kinit = tf.contrib.layers.xavier_initializer()
        #kinit = tf.random_normal_initializer(0, 0.01)

        self.dense = tf.layers.dense(inputs=flat, units=config.get('dense_layer_units'),
                activation=tf.nn.relu,
                use_bias=False, name='dense_layer')

        self.output = tf.layers.dense(inputs=self.dense, units=actions,
                use_bias=False, name='output_layer')

        mse = tf.reduce_mean(tf.square(self.output - qvals))
        self.loss = mse
        self.summary_all.append(tf.summary.scalar('loss', self.loss))

        self.transform_variables = []
        self.assign_ops = []
        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope):
            ev = tf.placeholder(tf.float32, None, name=get_transform_placeholder_name(v.name))
            self.assign_ops.append(tf.assign(v, ev, validate_shape=False))

            self.transform_variables.append(v)

        for i in range(actions):
            x = tf.one_hot(i, actions)
            out = self.output * x
            sum = tf.reduce_sum(out, axis=-1)
            self.summary_all.append(tf.summary.scalar("qvals_pred_{0}".format(i), tf.reduce_mean(sum)))

            out_in = qvals * x
            sum_in = tf.reduce_sum(out_in, axis=-1)
            self.summary_all.append(tf.summary.scalar("qvals_input_{0}".format(i), tf.reduce_mean(sum_in)))

        self.summary_all.append(tf.summary.histogram("actions_pred", tf.argmax(qvals, axis=1)))
        self.summary_all.append(tf.summary.histogram("qvals_input", qvals))
        self.summary_all.append(tf.summary.histogram("qvals_pred", self.output))

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = 0.00001 + tf.train.exponential_decay(config.get('learning_rate_start'), self.global_step,
            config.get('learning_rate_step'), config.get('learning_rate_factor'), staircase=True)

        self.summary_all.append(tf.summary.scalar('learning_rate', self.learning_rate))
        self.summary_all.append(tf.summary.scalar('global_step', self.global_step))

        train_steps_tensor = tf.placeholder(tf.int32, [], name='train_steps')
        self.summary_all.append(tf.summary.scalar('train_steps', train_steps_tensor))

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
        self.train_steps += 1
        ret = self.session.run([self.summary_merged, self.optimizer_step, self.global_step], feed_dict={
                self.scope + '/states:0': states,
                self.scope + '/qvals:0': qvals,
                self.scope + '/train_steps:0': self.train_steps,
            })

        if self.train_steps % self.summary_update_steps == 0:
            summary = ret[0]
            global_step = ret[2]
            self.summary_writer.add_summary(summary, global_step)

    def predict(self, states):
        ret = self.session.run([self.output], feed_dict={
                self.scope + '/states:0': states,
            })

        return ret[0]

    def export_params(self):
        res = self.session.run(self.transform_variables)
        d = {}
        for k, v in zip(self.transform_variables, res):
            d[k.name] = v
        return d

    def import_params(self, d, self_rate):
        def name(name):
            return self.scope + '/' + get_transform_placeholder_name(name) + ':0'

        ext_d = {}
        for k, ext_v in d.iteritems():
            ext_d[name(k)] = ext_v

            #if 'policy_layer/kernel' in k:
            #    print "exported {0}: name: {1}, value: {2}".format(k, name(k), ext_v)

        import_d = {}
        for k, self_v in self.export_params().iteritems():
            tn = name(k)

            ext_var = ext_d.get(tn, self_v)

            import_d[tn] = self_v * self_rate + ext_var * (1. - self_rate)

            #if 'policy_layer/kernel' in k:
            #    print "import: scope: {0}, name: {1}, self_rate: {2}, self_v: {3}, ext_var: {4}, saving: {5}".format(
            #            self.scope, tn, self_rate, self_v, ext_var, import_d[tn])

        #print("{0}: imported params: {1}, total params: {2}".format(self.scope, len(d), len(d1)))
        self.session.run(self.assign_ops, feed_dict=import_d)

    def update_rewards(self, rewards):
        self.last_rewards.extend(rewards)

        feed_dict = {
            self.scope + '/episode_rewards:0': rewards,
            self.scope + '/episode_rewards_mva:0': np.mean(self.last_rewards),
        }

        s = self.session.run([self.update_rewards_ops, self.global_step], feed_dict)
        self.summary_writer.add_summary(s[0], s[1])
