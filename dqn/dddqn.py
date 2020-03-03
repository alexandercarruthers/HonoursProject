import tensorflow as tf  # Deep Learning library
import numpy as np  # Handle matrices
import vizdoom
import random  # Handling random number generation
from skimage import transform  # Help us to preprocess the frames
from collections import deque  # Ordered collection with ends
import json  # for hyperparameters
import warnings  # ignore all the warning messages that are normally printed during the training because of skiimage
from dqn import shared

warnings.filterwarnings('ignore')
game_mode, network, initial_ammo, new, log_path, json_path, writer_path = shared.get_variables()
writer = tf.compat.v1.summary.FileWriter(writer_path)


def create_environment():
    game = vizdoom.DoomGame()
    game.load_config("../scenarios/" + game_mode + ".cfg")  # Load the correct configuration
    game.set_doom_scenario_path("../scenarios/" + game_mode + ".wad")  # Load the correct scenario
    game.init()
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    return game, possible_actions


game, possible_actions = create_environment()

state_size = [84, 84, 4]  # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)
action_size = game.get_available_buttons_size()  # 7 possible actions
last_episode = 0
last_explore_start = 0
# hyper params
learning_rate = 0.0002  # Alpha (aka learning rate)
total_episodes = 51  # Total episodes for training
max_steps = 2100  # Max possible steps in an episode
batch_size = 32
max_tau = 10000  # Tau is the C step where we update our target network FIXED Q TARGETS HYPERPARAMETERS
explore_start = 1.0  # exploration probability at start EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0001  # 0.00005  # exponential decay rate for exploration prob
gamma = 0.95  # Discounting rate # Q LEARNING hyperparameters
memory_size = 1000000  # Number of experiences the Memory can keep If you have GPU change to 1million

hyperparameter_dict = {"learning_rate": str(learning_rate),
                       "total_episodes": str(total_episodes),
                       "max_steps": str(max_steps),
                       "batch_size": str(batch_size),
                       "max_tau": str(max_tau),
                       "explore_start": str(explore_start),
                       "explore_stop": str(explore_stop),
                       "decay_rate": str(decay_rate),
                       "gamma": str(gamma),
                       "memory_size": str(memory_size)
                       }
shared.log_hyperparameters(writer=writer, hyperpara_dict=hyperparameter_dict)

pretrain_length = batch_size  # 100000  # Number of experiences stored in the Memory when initialized for the first time
training = True  # MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
episode_render = False  # TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT


def preprocess_frame(frame):  # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)
    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame  # frame[30:-10, 30:-30]
    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame


stack_size = 4  # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:  # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        stacked_frames.append(frame)  # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)  # Stack the frames
    else:
        stacked_frames.append(frame)  # Append frame to deque, automatically removes the oldest frame
        stacked_state = np.stack(stacked_frames, axis=2)  # Build the stacked state
    return stacked_state, stacked_frames


class DDDQNNet:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name
        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.compat.v1.variable_scope(self.name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 100, 120, 4]
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *state_size], name="inputs")  #
            self.ISWeights_ = tf.compat.v1.placeholder(tf.float32, [None, 1], name='IS_weights')
            self.actions_ = tf.compat.v1.placeholder(tf.float32, [None, action_size], name="actions_")
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.compat.v1.placeholder(tf.float32, [None], name="target")
            """
            First convnet:
            CNN
            ELU
            """
            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")
            self.conv1_out = tf.nn.relu(self.conv1, name="conv1_out")
            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")
            self.conv2_out = tf.nn.relu(self.conv2, name="conv2_out")
            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=64,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_out = tf.nn.relu(self.conv3, name="conv3_out")
            self.flatten = tf.layers.flatten(self.conv3_out)
            ## Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = tf.layers.dense(inputs=self.flatten,
                                            units=512,
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")
            self.value = tf.layers.dense(inputs=self.value_fc,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name="value")
            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                units=512,
                                                activation=tf.nn.relu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")
            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantages")
            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage,
                                                   tf.reduce_mean(self.advantage, axis=1, keepdims=True))
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            # The loss is modified because of PER
            self.absolute_errors = tf.abs(self.target_Q - self.Q)  # for updating Sumtree
            # self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


tf.compat.v1.reset_default_graph()  # Reset the graph

# Instantiate the DQNetwork
DQNetwork = DDDQNNet(state_size, action_size, learning_rate, name="DQNetwork")
# Instantiate the target network
TargetNetwork = DDDQNNet(state_size, action_size, learning_rate, name="TargetNetwork")


class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0
        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    # original from: https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    PER_b_increment_per_sampling = 0.001
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])  # Find the max priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.tree.add(max_priority, experience)  # set the max p for new p

    def sample(self, n):
        memory_b = []  # Create a sample array that will contains the minibatch
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        priority_segment = self.tree.total_priority / n  # priority segment
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)
        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get_leaf(value)
            sampling_probabilities = priority / self.tree.total_priority
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight
            b_idx[i] = index
            experience = [data]
            memory_b.append(experience)
        return b_idx, memory_b, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


memory = Memory(memory_size)
game.new_episode()

for i in range(pretrain_length):
    if i == 0:
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    action = random.choice(possible_actions)
    reward = game.make_action(action)
    done = game.is_episode_finished()
    if done:
        next_state = np.zeros(state.shape)
        experience = state, action, reward, next_state, done
        memory.store(experience)
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    else:
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        experience = state, action, reward, next_state, done
        memory.store(experience)
        state = next_state


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    # EPSILON GREEDY STRATEGY
    exp_exp_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if (explore_probability > exp_exp_tradeoff):
        action = random.choice(possible_actions)  # Make a random action (exploration)
    else:
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)  # Take the biggest Q value (= the best action)
        action = possible_actions[int(choice)]
    return action, explore_probability


def update_target_graph():
    from_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    to_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")
    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder


tf.compat.v1.summary.scalar("Loss", DQNetwork.loss)  # Losses
write_op = tf.compat.v1.summary.merge_all()
saver = tf.compat.v1.train.Saver()

if training is True:
    with tf.compat.v1.Session() as sess:
        # Initialize the variables # or restore
        if new is True:
            sess.run(tf.compat.v1.global_variables_initializer())
            last_episode = 0
            tf.compat.v1.summary.FileWriter(writer_path, sess.graph)
        if new is False:
            saver.restore(sess, log_path)
            # restore last hyper parameters
            lines = [line.rstrip('\n') for line in open(json_path)]
            last_line = json.loads(lines[-1])
            last_episode = last_line['episode']
            last_explore_start = float(last_line['explore_probability'])
            explore_start = last_explore_start
        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0
        tau = 0
        game.init()
        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)
        next_episode = last_episode + 1
        for episode in range(next_episode, total_episodes + next_episode):
            step = 0
            episode_rewards = []  # Initialize the rewards of the episode
            game.new_episode()  # Make a new episode and observe the first state
            state = game.get_state().screen_buffer
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            last_ammo_value = 0
            monsters_killed = 0
            while step < max_steps:
                step += 1
                tau += 1
                decay_step += 1
                # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions)
                reward = game.make_action(action)  # Do the action
                done = game.is_episode_finished()  # Look if the episode is finished
                episode_rewards.append(reward)  # Add the reward to total reward
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros((84, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    step = max_steps  # Set step = max_steps to end the episode
                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)
                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)
                    # calculate ammo used against the ammo player started with
                    ammo_used = initial_ammo - last_ammo_value
                    accuracy = shared.calculate_accuracy(monsters_killed, ammo_used)
                    # log to std out
                    shared.log_episode_std_out(loss,
                                               episode,
                                               explore_probability,
                                               total_reward,
                                               ammo_used,
                                               monsters_killed,
                                               accuracy)
                    # Log to tensorboard
                    writer.close()
                    writer = tf.summary.FileWriter(writer_path)
                    shared.log_episode_tensorboard(writer,
                                                   episode,
                                                   explore_probability,
                                                   total_reward,
                                                   ammo_used,
                                                   monsters_killed,
                                                   accuracy,
                                                   loss=loss)
                    writer.flush()
                    writer.close()

                    # shared.log_episode_csv(json_path,
                    #                        episode,
                    #                        explore_probability,
                    #                        total_reward,
                    #                        ammo_used,
                    #                        monsters_killed,
                    #                        accuracy,
                    #                        loss=loss)
                    # Log to txt file in json
                    shared.log_episode_json(json_path,
                                            episode,
                                            explore_probability,
                                            total_reward,
                                            ammo_used,
                                            monsters_killed,
                                            accuracy,
                                            loss=loss)

                else:
                    last_ammo_value = game.get_state().game_variables[0]
                    monsters_killed = game.get_state().game_variables[2]
                    # Get the next state
                    next_state = game.get_state().screen_buffer
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)
                    # st+1 is now our current state
                    state = next_state
                ### LEARNING PART
                # Obtain random mini-batch from memory
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
                states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch])
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
                dones_mb = np.array([each[0][4] for each in batch])
                target_Qs_batch = []
                ### DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')
                # Get Q values for next_state
                q_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})
                # Calculate Qtarget for all actions that state
                q_target_next_state = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states_mb})
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    # We got a'
                    action = np.argmax(q_next_state[i])
                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        # Take the Qtarget for action a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)
                targets_mb = np.array([each for each in target_Qs_batch])
                _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                                    feed_dict={DQNetwork.inputs_: states_mb,
                                                               DQNetwork.target_Q: targets_mb,
                                                               DQNetwork.actions_: actions_mb,
                                                               DQNetwork.ISWeights_: ISWeights_mb})

                # Update priority
                memory.batch_update(tree_idx, absolute_errors)
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                        DQNetwork.target_Q: targets_mb,
                                                        DQNetwork.actions_: actions_mb,
                                                        DQNetwork.ISWeights_: ISWeights_mb})
                writer.add_summary(summary, episode)
                writer.flush()
                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")
            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, log_path)  # log path from restored
                print("Model Saved")

with tf.Session() as sess:
    game = vizdoom.DoomGame()
    # Load the correct configuration (TESTING)
    game.load_config("../scenarios/" + game_mode + ".cfg")
    # Load the correct scenario (in our case deadly_corridor scenario)
    game.set_doom_scenario_path("../scenarios/" + game_mode + ".wad")
    game.init()
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    game.init()
    for i in range(10):
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        while not game.is_episode_finished():
            ## EPSILON GREEDY STRATEGY
            # Choose action a from state s using epsilon greedy.
            ## First we randomize a number
            exp_exp_tradeoff = np.random.rand()
            explore_probability = 0.01
            if (explore_probability > exp_exp_tradeoff):
                # Make a random action (exploration)
                action = random.choice(possible_actions)
            else:
                # Get action from Q-network (exploitation)
                # Estimate the Qs values state
                Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
                # Take the biggest Q value (= the best action)
                choice = np.argmax(Qs)
                action = possible_actions[int(choice)]
            game.make_action(action)
            done = game.is_episode_finished()
            if done:
                break
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
        score = game.get_total_reward()
        print("Score: ", score)
    game.close()
