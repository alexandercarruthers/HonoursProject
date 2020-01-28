import tensorflow as tf  # Deep Learning library
import numpy as np  # Handle matrices
import vizdoom
import random  # Handling random number generation
from skimage import transform  # Help us to preprocess the frames
from collections import deque  # Ordered collection with ends
import warnings  # This ignore all the warning messages that are normally printed during the training because of skimage
import json  # for hyperparameters
from dqn import shared
warnings.filterwarnings('ignore')


def create_environment():
    game = vizdoom.DoomGame()  # Load the correct configuration
    game.load_config("../scenarios/" + game_mode + ".cfg") # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("../scenarios/" + game_mode + ".wad")
    game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
    game.init()
    # Here our possible actions
    left = [1, 0, 0]  # Can also be True, False, False
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    return game, possible_actions


def preprocess_frame(frame):
    cropped_frame = frame  # frame[30:-10, 30:-30]
    normalized_frame = cropped_frame / 255.0  # Normalize Pixel Values
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])  # Resize
    return preprocessed_frame


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


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    # EPSILON GREEDY STRATEGY Choose action a from state s using epsilon greedy
    exp_exp_tradeoff = np.random.rand()
    # improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    if explore_probability > exp_exp_tradeoff:
        action = random.choice(possible_actions)  # Make a random action (exploration)
    else:
        # Get action from Q-network (exploitation) & Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)  # Take the biggest Q value (= the best action)
        action = possible_actions[int(choice)]
    return action, explore_probability


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        with tf.compat.v1.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.compat.v1.placeholder(tf.float32, [None, 3], name="actions_")
            self.target_Q = tf.compat.v1.placeholder(tf.float32, [None], name="target")  # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            # First convnet: CNN ReLU
            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_, filters=32, kernel_size=[8, 8], strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")
            self.conv1_out = tf.nn.relu(self.conv1, name="conv1_out")
            # Output --> [20, 20, 32]
            # Second convnet: CNN ReLU
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64, kernel_size=[4, 4], strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")
            self.conv2_out = tf.nn.relu(self.conv2, name="conv2_out")
            ## --> [9, 9, 64]
            # Third convnet: CNN ReLU
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, filters=64, kernel_size=[3, 3], strides=[1, 1],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")
            self.conv3_out = tf.nn.relu(self.conv3, name="conv3_out")
            ## --> [3, 3, 128]
            self.flatten = tf.layers.flatten(self.conv3_out)
            ## --> [1152]
            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="fc1")
            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=3,
                                          activation=None)
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]


game_mode, network, initial_ammo, new, log_path, json_path, writer_path = shared.get_variables()
writer = tf.compat.v1.summary.FileWriter(writer_path)
game, possible_actions = create_environment()
# PREVIOUS PARAMETERS
last_episode = 0
last_explore_start = 0
# MODEL HYPERPARAMETERS
state_size = [84, 84, 4]  # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = game.get_available_buttons_size()  # 3 possible actions: left, right, shoot
# TRAINING HYPERPARAMETERS
learning_rate = 0.00025  # Alpha (aka learning rate)
total_episodes = 500  # Total episodes for training
max_steps = 300  # Max possible steps in an episode
batch_size = 32
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0001  # 00 #0  # exponential decay rate for exploration prob
gamma = 0.99  # Discounting rate for Q learning
memory_size = 1000000  # Number of experiences the Memory can keep 1 million

hyperparameter_dict = {"learning_rate": str(learning_rate),
                       "total_episodes": str(total_episodes),
                       "max_steps": str(max_steps),
                       "batch_size": str(batch_size),
                       "explore_start": str(explore_start),
                       "explore_stop": str(explore_stop),
                       "decay_rate": str(decay_rate),
                       "gamma": str(gamma),
                       "memory_size": str(memory_size)
                       }
shared.log_hyperparameters(writer=writer, hyperpara_dict=hyperparameter_dict)
# MEMORY HYPERPARAMETERS
pretrain_length = batch_size  # Number of experiences stored in the Memory when initialized for the first time
training = True  # MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
episode_render = True  # TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
stack_size = 4  # We stack 4 frames
# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
tf.compat.v1.reset_default_graph()  # Reset the graph
DQNetwork = DQNetwork(state_size, action_size, learning_rate)  # Instantiate the DQNetwork
memory = Memory(max_size=memory_size)  # Instantiate memory
game.new_episode()  # Render the environment

for i in range(pretrain_length):
    if i == 0:
        state = game.get_state().screen_buffer  # First we need a state
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    action = random.choice(possible_actions)  # Random action
    reward = game.make_action(action)  # Get the rewards
    done = game.is_episode_finished()
    if done:  # If we're dead
        next_state = np.zeros(state.shape) # We finished the episode
        memory.add((state, action, reward, next_state, done))  # Add experience to memory
        game.new_episode() # Start a new episode
        state = game.get_state().screen_buffer  # First we need a state
        state, stacked_frames = stack_frames(stacked_frames, state, True)  # Stack the frames
    else:
        next_state = game.get_state().screen_buffer # Get the next state
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        memory.add((state, action, reward, next_state, done)) # Add experience to memory
        state = next_state # Our state is now the next_state

tf.compat.v1.summary.scalar("Loss", DQNetwork.loss)  # Losses
write_op = tf.compat.v1.summary.merge_all()
saver = tf.compat.v1.train.Saver() # Saver will help us to save our model

if training:
    with tf.compat.v1.Session() as sess:
        if new is True:
            sess.run(tf.compat.v1.global_variables_initializer())
            tf.compat.v1.summary.FileWriter(writer_path, sess.graph)
            last_episode = 0
        if new is False:
            saver.restore(sess, log_path)
            # restore last hyper parameters
            lines = [line.rstrip('\n') for line in open(json_path)]
            last_line = json.loads(lines[-1])
            last_episode = last_line['episode']
            last_explore_start = float(last_line['explore_probability'])  # saved as a string need to be a float
            explore_start = last_explore_start
        decay_step = 0  # Initialize the decay rate (that will use to reduce epsilon)
        game.init() # Init the game
        next_episode = last_episode + 1
        for episode in range(next_episode, total_episodes + next_episode):
            step = 0
            episode_rewards = []  # Initialize the rewards of the episode
            game.new_episode()  # Make a new episode and observe the first state
            state = game.get_state().screen_buffer # that stack frame function also calls our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            last_ammo_value = 0
            monsters_killed = 0
            while step < max_steps:
                step += 1
                decay_step += 1 # Increase decay_step
                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
                reward = game.make_action(action) # Do the action
                done = game.is_episode_finished() # Look if the episode is finished
                episode_rewards.append(reward) # Add the reward to total reward
                if done: # If the game is finished
                    next_state = np.zeros((84, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    step = max_steps  # Set step = max_steps to end the episode
                    total_reward = np.sum(episode_rewards)  # Get the total reward of the episode
                    memory.add((state, action, reward, next_state, done))
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
                    shared.log_episode_tensorboard(writer,
                                                   episode,
                                                   explore_probability,
                                                   total_reward,
                                                   ammo_used,
                                                   monsters_killed,
                                                   accuracy,
                                                   loss=loss)
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
                    next_state = game.get_state().screen_buffer # Get the next state
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    memory.add((state, action, reward, next_state, done))  # Add experience to memory
                    state = next_state # st+1 is now our current state
                # LEARNING PART
                # Obtain random mini-batch from memory
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])
                target_Qs_batch = []
                # Get Q values for next_state
                Qs_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                targets_mb = np.array([each for each in target_Qs_batch])
                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                   feed_dict={DQNetwork.inputs_: states_mb,
                                              DQNetwork.target_Q: targets_mb,
                                              DQNetwork.actions_: actions_mb})
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                        DQNetwork.target_Q: targets_mb,
                                                        DQNetwork.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()
            if episode % 5 == 0:  # Save model every 5 episodes
                save_path = saver.save(sess, log_path)  # log path from restored
                print("Model Saved")

with tf.compat.v1.Session() as sess:
    game, possible_actions = create_environment()
    totalScore = 0
    saver.restore(sess, log_path)
    game.init()
    for i in range(10):
        done = False
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        while not game.is_episode_finished():
            # Take the biggest Q value (= the best action)
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[int(choice)]
            game.make_action(action)
            done = game.is_episode_finished()
            score = game.get_total_reward()
            if done:
                break
            else:
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
        score = game.get_total_reward()
        print("Score: ", score)
    game.close()