import datetime  # Logging
import tensorflow as tf  # Deep Learning library
import numpy as np  # Handle matrices
import vizdoom
import random  # Handling random number generation
from skimage import transform  # Help us to preprocess the frames
from collections import deque  # Ordered collection with ends
import warnings  # This ignore all the warning messages that are normally printed during the training because of skimage
import json # for hyperparameters
warnings.filterwarnings('ignore')


def create_environment():
    game = vizdoom.DoomGame()
    # Load the correct configuration
    game.load_config("../scenarios/" + game_mode + ".cfg")
    # Load the correct scenario (in our case basic scenario)
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
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)
    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame[30:-10, 30:-30]
    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])
    return preprocessed_frame


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


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    # EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    # First we randomize a number
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


class DQNetwork:
    '''
    Cannot update
    flatten, batch norm, conv2d
    '''
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

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.compat.v1.placeholder(tf.float32, [None], name="target")

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


# GAME MODE CHOICE
game_mode = "basic"  # defend_the_center
initial_ammo = 50 # basic = 50 def = 26
network = "DQN"
new = False

text = input("1 to load previous checkpoint\n"
             "2 for new model\n"
             "3 exit\n")
if text == "1":
    print("checkpoint name in format ./models/gamemode/network/DD-MM-YY-HH-MM-model.ckpt")
    game_mode = input("game mode: ")
    network = input("network: ")
    date = input("DD-MM-YYYY: ")
    time = input("HH-MM: ")
    log_path = "./models/" + game_mode + "/" + network + "/" + date + "-" + time + "-model.ckpt"
    json_log = "./models/" + game_mode + "/" + network + "/" + date + "-" + time + "log.txt"
    writer = tf.compat.v1.summary.FileWriter("/tensorboard/" + game_mode + "/" + network + "/" + date + "/" + time)
    new = False
if text == "2":
    today_date = datetime.datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.datetime.now().strftime("%H-%M")
    log_path = "./models/" + game_mode + "/" + network + "/" + today_date + "-" + current_time + "-model.ckpt"
    json_log = "./models/" + game_mode + "/" + network + "/" + today_date + "-" + current_time + "log.txt"
    writer = tf.compat.v1.summary.FileWriter("/tensorboard/" + game_mode + "/" + network + "/" + today_date + "/" + current_time)
    new = True
if text == "3":
    exit()


game, possible_actions = create_environment()
# PREVIOUS PARAMETERS
last_episode = 0
last_explore_start = 0
# MODEL HYPERPARAMETERS
state_size = [84, 84, 4]  # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = game.get_available_buttons_size()  # 3 possible actions: left, right, shoot
learning_rate = 0.0002  # Alpha (aka learning rate)
# TRAINING HYPERPARAMETERS
total_episodes = 900  # Total episodes for training
max_steps = 300  # Max possible steps in an episode
batch_size = 16
# Exploration parameters for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.00001 #00 #0  # exponential decay rate for exploration prob
# Q learning hyperparameters
gamma = 0.95  # Discounting rate

# MEMORY HYPERPARAMETERS
pretrain_length = batch_size  # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000  # Number of experiences the Memory can keep 1 million
# MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True
# TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = True
stack_size = 4  # We stack 4 frames
# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)
# Reset the graph
tf.compat.v1.reset_default_graph()
# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)
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

## Losses
tf.compat.v1.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.compat.v1.summary.merge_all()

# Saver will help us to save our model
saver = tf.compat.v1.train.Saver()

if training:
    with tf.compat.v1.Session() as sess:
        # Initialize the variables # or restore
        if new == True:
            sess.run(tf.compat.v1.global_variables_initializer())
            last_episode = 0
        if new == False:
            saver.restore(sess, log_path)
            # restore last hyper parameters
            lines = [line.rstrip('\n') for line in open(json_log)]
            last_line = json.loads(lines[-1])
            last_episode = last_line['episode']
            last_explore_start = last_line['explore_probability']
            explore_start = last_explore_start

            # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0
        # Init the game
        game.init()
        next_episode = last_episode + 1
        for episode in range(next_episode, total_episodes + next_episode):
            # Set step to 0
            step = 0
            # Initialize the rewards of the episode
            episode_rewards = []
            # Make a new episode and observe the first state
            game.new_episode()
            state = game.get_state().screen_buffer
            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            last_ammo_value = 0
            monsters_killed = 0
            while step < max_steps:
                step += 1
                # Increase decay_step
                decay_step += 1
                # Predict the action to take and take it
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
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
                    total_reward = np.sum(episode_rewards)
                    # calculate ammo used against the ammo player started with
                    ammo_used = initial_ammo - last_ammo_value
                    accuracy = (monsters_killed / ammo_used) * 100
                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability),
                          'Ammo used: {:.4f}'.format(ammo_used),
                          'monsters killed: {:.4f}'.format(monsters_killed),
                          'Accuracy: {:.4f}'.format(accuracy))
                    memory.add((state, action, reward, next_state, done))
                    # Log episode data
                    log_titles = ['explore_probability', 'total_reward', 'ammo_used', 'monsters_killed', 'accuracy']
                    log_values = [explore_probability, total_reward, ammo_used, monsters_killed, accuracy]
                    for log, values in zip(log_titles, log_values):
                        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=log, simple_value=values)])
                        writer.add_summary(summary, episode)
                    write_op = tf.compat.v1.summary.merge_all()
                    writer.flush()
                    # write to .txt in json
                    hyperparameter_data = {'episode': episode, 'explore_probability': explore_probability, 'total_reward': total_reward, 'ammo_used': ammo_used, 'monsters_killed': monsters_killed, 'accuracy': accuracy}
                    y = json.dumps(hyperparameter_data)
                    with open(json_log, 'a') as outfile:
                        outfile.write(y)
                        outfile.write("\n")
                else:
                    last_ammo_value = game.get_state().game_variables[0]
                    monsters_killed = game.get_state().game_variables[2]
                    # Get the next state
                    next_state = game.get_state().screen_buffer
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))
                    # st+1 is now our current state
                    state = next_state
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

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, log_path) # log path from restored
                print("Model Saved")


with tf.compat.v1.Session() as sess:
    game, possible_actions = create_environment()
    totalScore = 0
    # Load the model
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