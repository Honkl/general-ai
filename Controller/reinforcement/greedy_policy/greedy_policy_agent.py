import numpy as np
import tensorflow as tf

from reinforcement.replay_buffer import ReplayBuffer

REPLAY_BUFFER_SIZE = 100000


class GreedyPolicyAgent():
    """
    Greedy-policy reinforcemnet learning agent.
    """

    def __init__(self, parameters, q_network, state_size, action_size, logdir, threads):
        """
        Initializes a new agent.
        :param reinfoce_params: GreedyPolicyParameters.
        :param q_network: Q-network to be used.
        :param state_size: Size of the environment state.
        :param logdir: Logdir where to store logs.
        :param threads: Number of threads to use (in tensorflow).
        """
        self.batch_size = parameters.batch_size
        batch_size = self.batch_size
        self.gamma = parameters.gamma
        self.epsilon = parameters.epsilon
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = q_network
        self.logdir = logdir
        self.saver = None
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        self.sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                     intra_op_parallelism_threads=threads,
                                                     allow_soft_placement=True))
        with tf.device('/gpu:0'):
            with tf.variable_scope('agent') as scope:
                self.state = tf.placeholder(shape=[None, state_size], dtype=tf.float32, name="state")

                self.selected_action, self.estimated_reward = self.select_best_action(self.state)
                self.selected_action = tf.squeeze(self.selected_action)
                self.estimated_reward = tf.squeeze(self.estimated_reward)
                scope.reuse_variables()

                self.new_state = tf.placeholder(shape=[batch_size, state_size], dtype=tf.float32, name="new_state")
                self.last_reward = tf.placeholder(shape=[batch_size], dtype=tf.float32, name="last_reward")
                self.last_action = tf.placeholder(shape=[batch_size, action_size], dtype=tf.float32, name="last_action")
                # self.last_estimated_reward = tf.placeholder(shape=[batch_size], dtype=tf.float32, name="last_estimated_reward")
                self.done = tf.placeholder(shape=[batch_size], dtype=tf.bool, name="done")

                _, new_estimated_reward = self.select_best_action(self.new_state)
                last_estimated_reward = self.get_estimated_reward(self.state, self.last_action)

                self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

                # loss = (r + γ*max_a'Q(s',a';θ) - Q(s,a;θ))^2
                self.losses = (self.last_reward + self.gamma * new_estimated_reward - last_estimated_reward) ** 2
                self.loss = tf.reduce_mean(self.losses)
                self.optimizer = self.get_optimizer(parameters.optimizer)
                self.training = self.optimizer(parameters.learning_rate).minimize(self.loss,
                                                                                  global_step=self.global_step)

                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

                self.summary_writer = tf.summary.FileWriter(logdir,
                                                            graph=self.sess.graph,
                                                            flush_secs=10)

    def get_optimizer(self, opt_str):
        if opt_str == "adam":
            return tf.train.AdamOptimizer
        if opt_str == "rmsprop":
            return tf.train.RMSPropOptimizer
        raise NotImplementedError

    def play(self, env_state):
        # Epsilon-Greedy policy (random action with probability of epsilon)
        if np.random.random() < self.epsilon:
            estimated_reward = self.sess.run(self.estimated_reward, {self.state: [env_state]})
            n_actions = self.q_network.output_size
            selected_action = []
            for i in range(n_actions):
                selected_action.append(np.random.random())
        else:
            selected_action, estimated_reward = self.sess.run([self.selected_action, self.estimated_reward],
                                                              {self.state: [env_state]})
        return selected_action, estimated_reward

    def select_best_action(self, state):
        result = self.Q(state)
        # selected_action = tf.argmax(result, 1)
        selected_action = result
        estimated_reward = tf.reduce_max(result, 1)

        return selected_action, estimated_reward

    def get_estimated_reward(self, state, action):
        result = self.Q(state)
        amax = tf.arg_max(action, 1)
        indices = amax + tf.cast(tf.range(tf.shape(result)[0]) * tf.shape(result)[1], tf.int64)
        reshaped = tf.reshape(result, [-1])
        estimated_reward = tf.gather(reshaped, indices)
        return estimated_reward

    def Q(self, state):
        return self.q_network.forward_pass(state)

    def learn(self, old_state, action, reward, new_state, done):
        self.replay_buffer.add(old_state, action, reward, new_state, done)
        loss = None
        if (self.replay_buffer.count() >= self.batch_size):
            minibatch = self.replay_buffer.get_batch(self.batch_size)
            old_state_batch = np.asarray([data[0] for data in minibatch])
            action_batch = np.asarray([data[1] for data in minibatch])
            reward_batch = np.asarray([data[2] for data in minibatch])
            new_state_batch = np.asarray([data[3] for data in minibatch])
            done_batch = np.asarray([data[4] for data in minibatch])

            _, loss = self.sess.run([self.training, self.loss],
                                    {self.new_state: np.array(new_state_batch).reshape(self.batch_size,
                                                                                       self.state_size),
                                     self.last_reward: np.array(reward_batch),
                                     self.state: np.array(old_state_batch).reshape(self.batch_size,
                                                                                   self.state_size),
                                     # self.last_estimated_reward: np.array(estm_reward_batch).reshape(self.batch_size),
                                     self.last_action: np.array(action_batch).reshape(self.batch_size,
                                                                                      self.action_size),
                                     self.done: np.array(done_batch.reshape(self.batch_size))})
        return loss
