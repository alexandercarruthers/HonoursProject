import tensorflow as tf  # Deep Learning library
import numpy as np  # Handle matrices
from skimage import transform  # Help us to preprocess the frames
from collections import deque  # Ordered collection with ends
import warnings  # This ignore all the warning messages from  skiimage
import vizdoom
import json
from dqn import shared
warnings.filterwarnings('ignore')


def create_environment():
    game = vizdoom.DoomGame()  # Load the correct configuration
    game_mode = "basic"
    game.load_config("../scenarios/" + game_mode + ".cfg")  # Load the correct scenario
    game.set_doom_scenario_path("../scenarios/" + game_mode + ".wad")
    game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
    game.init()
    possible_actions = np.identity(3, dtype=int).tolist()
    return game, possible_actions


def preprocess_frame(frame):
    cropped_frame = frame  # frame[30:-10, 30:-30]
    normalized_frame = cropped_frame / 255.0  # Normalize Pixel Values
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])  # Resize
    return preprocessed_frame


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)     # Preprocess frame
    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)  # clear
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


# discount_and_normalize_rewards 💰
# This function is important, because we are in a Monte Carlo situation. <br>
# We need to **discount the rewards at the end of the episode**.
# This function takes, the reward discount it
# **then normalize them** (to avoid a big variability in rewards).
def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    return discounted_episode_rewards


last_ammo_value = 0
monsters_killed = 0


def make_batch(batch_size, stacked_frames):
    # Initialize lists: states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards
    states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []
    # Reward of batch is also a trick to keep track of how many timestep we made.
    # We use to to verify at the end of each episode if > batch_size or not.
    # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)
    episode_num = 1
    # Launch a new episode
    game.new_episode()
    # Get a new state
    state = game.get_state().screen_buffer
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    while True:
        # Run State Through Policy & Calculate Action
        action_probability_distribution = sess.run(PGNetwork.action_distribution, feed_dict={PGNetwork.inputs_: state.reshape(1, *state_size)})
        # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
        # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
        # 30% chance that we take action a2)
        action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
        action = possible_actions[action]
        # Perform action
        reward = game.make_action(action)
        done = game.is_episode_finished()
        # Store results
        states.append(state)
        actions.append(action)
        rewards_of_episode.append(reward)
        if done:
            # The episode ends so no next state
            next_state = np.zeros((84, 84), dtype=np.int)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            # Append the rewards_of_batch to reward_of_episode
            rewards_of_batch.append(rewards_of_episode)
            # Calculate gamma Gt
            discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode))
            # If the number of rewards_of_batch > batch_size stop the minibatch creation
            # (Because we have sufficient number of episode mb)
            # Remember that we put this condition here, because we want entire episode (Monte Carlo)
            # so we can't check that condition for each step but only if an episode is finished
            if len(np.concatenate(rewards_of_batch)) > batch_size:
                break
            # Reset the transition stores
            rewards_of_episode = []
            # Add episode
            episode_num += 1
            # Start a new episode
            game.new_episode()
            # First we need a state
            state = game.get_state().screen_buffer
            # Stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        else:
            global last_ammo_value
            global monsters_killed
            last_ammo_value = game.get_state().game_variables[0]
            monsters_killed = game.get_state().game_variables[2]
            # If not done, the next_state become the current state
            next_state = game.get_state().screen_buffer
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state
    return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewards_of_batch), np.concatenate(discounted_rewards), episode_num

game_mode, network, initial_ammo, new, log_path, json_path, writer_path = shared.get_variables()
writer = tf.compat.v1.summary.FileWriter(writer_path)
game, possible_actions = create_environment()
stack_size = 4  # We stack 4 frames
# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
# ENVIRONMENT HYPERPARAMETERS
state_size = [84, 84, 4]  # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = game.get_available_buttons_size()  # 3 possible actions: turn left, turn right, move forward
# TRAINING HYPERPARAMETERS
learning_rate = 0.002
total_episodes = 500  # Total epochs for training
batch_size = 16  # Each 1 is a timestep (NOT AN EPISODE) # YOU CAN CHANGE TO 5000 if you have GPU
gamma = 0.95  # Discounting rate
hyperparameter_dict = {"learning_rate": str(learning_rate),
                       "total_episodes": str(total_episodes),
                       "batch_size": str(batch_size),
                       "gamma": str(gamma)
                       }
shared.log_hyperparameters(writer=writer, hyperpara_dict=hyperparameter_dict)
training = True  # MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
# Quick note: Policy gradient methods like reinforce **are on-policy method which can not be updated from experience replay.**


class PGNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='PGNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        with tf.compat.v1.variable_scope(name):
            with tf.name_scope("inputs"):
                # We create the placeholders
                # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
                # [None, 84, 84, 4]
                self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *state_size], name="inputs_")
                self.actions = tf.compat.v1.placeholder(tf.int32, [None, action_size], name="actions")
                self.discounted_episode_rewards_ = tf.compat.v1.placeholder(tf.float32, [None, ],
                                                                  name="discounted_episode_rewards_")
                # Add this placeholder for having this variable in tensorboard
                self.mean_reward_ = tf.compat.v1.placeholder(tf.float32, name="mean_reward")

            with tf.name_scope("conv1"):
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
            with tf.name_scope("conv2"):
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
            with tf.name_scope("conv3"):
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
            with tf.name_scope("flatten"):
                self.flatten = tf.layers.flatten(self.conv3_out)
                ## --> [1152]
            with tf.name_scope("fc1"):
                self.fc = tf.layers.dense(inputs=self.flatten,
                                          units=512,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="fc1")
            with tf.name_scope("logits"):
                self.logits = tf.layers.dense(inputs=self.fc,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              units=3,
                                              activation=None)

            with tf.name_scope("softmax"):
                self.action_distribution = tf.nn.softmax(self.logits)

            with tf.name_scope("loss"):
                # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
                # If you have single-class labels, where an object can only belong to one class, you might now consider using
                # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array.
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.actions)
                self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_episode_rewards_)

            with tf.name_scope("train"):
                self.train_opt = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


tf.compat.v1.reset_default_graph()  # Reset the graph
PGNetwork = PGNetwork(state_size, action_size, learning_rate)  # Instantiate the PGNetwork
sess = tf.compat.v1.Session() # Initialize Session
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
# Setup TensorBoard Writer
writer = tf.compat.v1.summary.FileWriter(writer_path)
tf.compat.v1.summary.scalar("Loss", PGNetwork.loss)  # Losses
tf.compat.v1.summary.scalar("Reward_mean", PGNetwork.mean_reward_) # Reward mean
write_op = tf.compat.v1.summary.merge_all()





allRewards = []  # Keep track of all rewards total for each batch
total_rewards = 0
maximumRewardRecorded = 0
mean_reward_total = []
episode = 1
average_reward = []
saver = tf.compat.v1.train.Saver()  # Saver

if training:
    if new is True:
        tf.compat.v1.summary.FileWriter(writer_path, sess.graph)
    if new is False:
        saver.restore(sess, log_path) # restore from last run
        # restore last hyper parameters
        lines = [line.rstrip('\n') for line in open(json_path)]
        last_line = json.loads(lines[-1])
        last_episode = last_line['episode']
        episode = last_episode
    while episode < total_episodes + 1:
        states_mb, actions_mb, rewards_of_batch, discounted_rewards_mb, nb_episodes_mb = make_batch(batch_size, stacked_frames)
        # These part is used for analytics
        # Calculate the total reward ot the batch
        total_reward_of_that_batch = np.sum(rewards_of_batch)
        allRewards.append(total_reward_of_that_batch)
        # Calculate the mean reward of the batch
        # Total rewards of batch / nb episodes in that batch
        mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, nb_episodes_mb)
        mean_reward_total.append(mean_reward_of_that_batch)
        # Calculate the average reward of all training
        # mean_reward_of_that_batch / epoch
        average_reward_of_all_training = np.divide(np.sum(mean_reward_total), episode)
        # Calculate maximum reward recorded
        maximumRewardRecorded = np.amax(allRewards)
        print("==========================================")
        print("Epoch: ", episode, "/", total_episodes)
        print("-----------")
        print("Number of training episodes: {}".format(nb_episodes_mb))
        print("Total reward: {}".format(total_reward_of_that_batch, nb_episodes_mb))
        print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
        print("Average Reward of all training: {}".format(average_reward_of_all_training))
        print("Max reward for a batch so far: {}".format(maximumRewardRecorded))
        # Feedforward, gradient and backpropagation
        loss, _ = sess.run([PGNetwork.loss, PGNetwork.train_opt],
                            feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 84, 84, 4)),
                                       PGNetwork.actions: actions_mb,
                                       PGNetwork.discounted_episode_rewards_: discounted_rewards_mb
                                       })

        print("Training Loss: {}".format(loss))

        # Write TF Summaries
        summary = sess.run(write_op, feed_dict={PGNetwork.inputs_: states_mb.reshape((len(states_mb), 84, 84, 4)),
                                                PGNetwork.actions: actions_mb,
                                                PGNetwork.discounted_episode_rewards_: discounted_rewards_mb,
                                                PGNetwork.mean_reward_: mean_reward_of_that_batch
                                                })
        # summary = sess.run(write_op, feed_dict={x: s_.reshape(len(s_),84,84,1), y:a_, d_r: d_r_, r: r_, n: n_})
        writer.add_summary(summary, episode)
        writer.flush()

        ammo_used = initial_ammo - last_ammo_value
        accuracy = shared.calculate_accuracy(monsters_killed, ammo_used)
        # log to std out
        shared.log_episode_std_out(loss,
                                   episode,
                                   explore_probability=0,
                                   total_reward=0,
                                   ammo_used=ammo_used,
                                   monsters_killed=monsters_killed,
                                   accuracy=accuracy)
        # Log to tensorboard
        shared.log_episode_tensorboard(writer,
                                       episode,
                                       explore_probability=0,
                                       total_reward=0,
                                       ammo_used=ammo_used,
                                       monsters_killed=monsters_killed,
                                       accuracy=accuracy,
                                       loss=loss)
        shared.log_episode_json(json_path,
                                episode,
                                explore_probability=0,
                                total_reward=0,
                                ammo_used=ammo_used,
                                monsters_killed=monsters_killed,
                                accuracy=accuracy,
                                loss=loss)

        # Save Model
        if episode % 5 == 0:
            saver.save(sess, log_path)
            print("Model saved")
        episode += 1
saver = tf.train.Saver()
with tf.compat.v1.Session() as sess:
    game = vizdoom.DoomGame()
    game.load_config("health_gathering.cfg")
    game.set_doom_scenario_path("health_gathering.wad")
    saver.restore(sess, "./models/model.ckpt")
    game.init()
    for i in range(10):
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        while not game.is_episode_finished():
            # Run State Through Policy & Calculate Action
            action_probability_distribution = sess.run(PGNetwork.action_distribution,
                                                       feed_dict={PGNetwork.inputs_: state.reshape(1, *state_size)})
            # REMEMBER THAT WE ARE IN A STOCHASTIC POLICY SO WE DON'T ALWAYS TAKE THE ACTION WITH THE HIGHEST PROBABILITY
            # (For instance if the action with the best probability for state S is a1 with 70% chances, there is
            # 30% chance that we take action a2)
            action = np.random.choice(range(action_probability_distribution.shape[1]),
                                      p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
            action = possible_actions[action]
            reward = game.make_action(action)
            done = game.is_episode_finished()
            if done:
                break
            else:
                # If not done, the next_state become the current state
                next_state = game.get_state().screen_buffer
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
        print("Score for episode ", i, " :", game.get_total_reward())
    game.close()