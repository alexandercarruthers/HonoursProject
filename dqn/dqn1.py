import tensorflow as tf  # Deep Learning library
import numpy as np  # Handle matrices
from vizdoom import *  # Doom Environment
import vizdoom
import random  # Handling random number generation
import time  # Handling time calculation
from skimage import transform  # Help us to preprocess the frames
import datetime
from collections import deque  # Ordered collection with ends
import matplotlib.pyplot as plt  # Display graphs
from dqn import shared
import \
    warnings  # This ignore all the warning messages that are normally printed during the training because of skiimage

warnings.filterwarnings('ignore')

game_mode = "basic"
def create_environment():
    game = vizdoom.DoomGame()


    game.load_config("../scenarios/" + game_mode + ".cfg")
    game.set_doom_scenario_path("../scenarios/" + game_mode + ".wad")
    #game.load_config("basic.cfg")
    #game.set_doom_scenario_path("basic.wad")
    game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
    game.init()
    # Here our possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]
    return game, possible_actions


def test_environment():
    game = vizdoom.DoomGame()
    game.load_config("../scenarios/" + game_mode + ".cfg")
    game.set_doom_scenario_path("../scenarios/" + game_mode + ".wad")
    game.init()
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print("\treward:", reward)
            time.sleep(0.02)
        print("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()


game, possible_actions = create_environment()

def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)
    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame[30:-10, 30:-30]
    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame



stack_size = 4  # We stack 4 frames

# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames



### MODEL HYPERPARAMETERS
state_size = [84, 84, 4]  # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = game.get_available_buttons_size()  # 3 possible actions: left, right, shoot
learning_rate = 0.0002  # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 500  # Total episodes for training
max_steps = 100  # Max possible steps in an episode
batch_size = 64

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0001  # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.99  # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size  # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000  # Number of experiences the Memory can keep
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

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


# ## Step 5: Create our Deep Q-learning Neural Network model 🧠
# <img src="assets/model.png" alt="Model" />
# This is our Deep Q-learning model:
# - We take a stack of 4 frames as input
# - It passes through 3 convnets
# - Then it is flatened
# - Finally it passes through 2 FC layers
# - It outputs a Q value for each actions

# In[8]:


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 3], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            BatchNormalization
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

            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [20, 20, 32]

            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [9, 9, 64]

            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                                 training=True,
                                                                 epsilon=1e-5,
                                                                 name='batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            ## --> [3, 3, 128]

            self.flatten = tf.layers.flatten(self.conv3_out)
            ## --> [1152]

            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
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

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

# Reset the graph
tf.reset_default_graph()
# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


# Instantiate memory
memory = Memory(max_size=memory_size)

# Render the environment
game.new_episode()

for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        # First we need a state
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    # Random action
    action = random.choice(possible_actions)

    # Get the rewards
    reward = game.make_action(action)

    # Look if the episode is finished
    done = game.is_episode_finished()

    # If we're dead
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Start a new episode
        game.new_episode()

        # First we need a state
        state = game.get_state().screen_buffer

        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        # Get the next state
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Our state is now the next_state
        state = next_state

# Setup TensorBoard Writer
network = "dqn"
date = datetime.datetime.now().strftime("%d-%m-%Y")
time = datetime.datetime.now().strftime("%H-%M")
writer_path = "/tensorboard/" + game_mode + "/" + network + "/" + date + "/" + time
writer = tf.summary.FileWriter(writer_path)
generic_path = "./models/" + game_mode + "/" + network + "/" + date + "-" + time
log_path = generic_path + "-model.ckpt"

shared.log_hyperparameters(writer=writer, hyperpara_dict=hyperparameter_dict)
writer.close()
## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

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

    return action, explore_probability


saver = tf.train.Saver()


if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        tf.compat.v1.summary.FileWriter(writer_path, sess.graph)

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        # Init the game
        game.init()

        for episode in range(total_episodes):
            # Set step to 0
            step = 0

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state
            game.new_episode()
            state = game.get_state().screen_buffer

            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            monsters_killed = 0
            while step < max_steps:
                step += 1

                # Increase decay_step
                decay_step += 1

                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state,
                                                             possible_actions)

                # Do the action
                reward = game.make_action(action)

                # Look if the episode is finished
                done = game.is_episode_finished()

                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # the episode ends so no next state
                    next_state = np.zeros((84, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    # total_reward = np.sum(episode_rewards)
                    #
                    # print('Episode: {}'.format(episode),
                    #       'Total reward: {}'.format(total_reward),
                    #       'Training loss: {:.4f}'.format(loss),
                    #       'Explore P: {:.4f}'.format(explore_probability),
                    #       'Monsters killed: {}'.format(monsters_killed))
                    #
                    #
                    #
                    # writer = tf.summary.FileWriter(writer_path)
                    # summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="monsters_killed", simple_value=monsters_killed)])
                    # writer.add_summary(summary, episode)
                    # writer.flush()
                    memory.add((state, action, reward, next_state, done))
                else:
                    # Get the next state
                    monsters_killed = game.get_state().game_variables[2]
                    next_state = game.get_state().screen_buffer

                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state

                ### LEARNING PART
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

            total_reward = np.sum(episode_rewards)

            print('Episode: {}'.format(episode),
                  'Total reward: {}'.format(total_reward),
                  'Training loss: {:.4f}'.format(loss),
                  'Explore P: {:.4f}'.format(explore_probability),
                  'Monsters killed: {}'.format(monsters_killed))

            memory.add((state, action, reward, next_state, done))

            writer = tf.summary.FileWriter(writer_path)
            summary = tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(tag="monsters_killed", simple_value=monsters_killed)])
            writer.add_summary(summary, episode)
            writer.flush()

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, log_path)
                print("Model Saved")

# ## Step 9: Watch our Agent play 👀
# Now that we trained our agent, we can test it

# In[ ]:


with tf.Session() as sess:
    game, possible_actions = create_environment()

    totalScore = 0

    # Load the model
    saver.restore(sess, log_path)
    game.init()
    for i in range(1):

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
                print("else")
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state

        score = game.get_total_reward()
        print("Score: ", score)
    game.close()

