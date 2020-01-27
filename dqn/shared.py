import datetime
import json
import tensorflow as tf


'''def log():
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.datetime.now().strftime("%H-%M")
    hyperparameter_data = {'episode': episode,
                           'explore_probability': explore_probability,
                           'total_reward': total_reward,
                           'ammo_used': ammo_used,
                           'monsters_killed': monsters_killed,
                           'accuracy': accuracy,
                           'date': current_date,
                           'time': current_time}
'''


def calculate_accuracy(frags, ammo_used):
    accuracy = (frags / ammo_used) * 100
    return accuracy


def log_episode(path, episode, explore_probability, total_reward, ammo_used, monsters_killed, accuracy):
    output_data = {'episode': episode,
                   'explore_probability': explore_probability,
                   'total_reward': total_reward,
                   'ammo_used': ammo_used,
                   'monsters_killed': monsters_killed,
                   'accuracy': accuracy}
    # Get current date / time
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.datetime.now().strftime("%H-%M")
    output_data['date'] = current_date
    output_data['time'] = current_time
    # dictionary to json
    output_data_as_json = json.dumps(output_data)
    with open(path, 'a') as outfile:
        outfile.write(output_data_as_json)
        outfile.write("\n")


def log_episode_tensorboard(writer, episode, explore_probability, total_reward, ammo_used, monsters_killed, accuracy, loss):
    log_titles = ['loss','explore_probability', 'total_reward', 'ammo_used', 'monsters_killed', 'accuracy']
    log_values = [loss, explore_probability, total_reward, ammo_used, monsters_killed, accuracy]
    for log, values in zip(log_titles, log_values):
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=log, simple_value=values)])
        writer.add_summary(summary, episode)
    # write_op = tf.compat.v1.summary.merge_all()
    writer.flush()


def log_episode_std_out(loss, episode, explore_probability, total_reward, ammo_used, monsters_killed, accuracy):
    print('Episode: {}'.format(episode),
          'Total reward: {}'.format(total_reward),
          'Training loss: {:.4f}'.format(loss),
          'Explore P: {:.4f}'.format(explore_probability),
          'Ammo used: {}'.format(ammo_used),
          'monsters killed: {}'.format(monsters_killed),
          'Accuracy: {}'.format(accuracy))


def get_variables():
    game_mode = ""  # defend_the_center
    initial_ammo = 0  # basic = 50 def = 26
    network = ""  # DQN DDDQN PolicyGradient
    new = False

    # Choose network
    network_input = input("1: DQN\n"
                          "2: DDDQN\n"
                          "3: Policy Gradient\n")
    if network_input == "1":
        network = "DQN"
    elif network_input == "2":
        network = "DDDQN"
    elif network_input == "3":
        network = "PolicyGradient"

    # Choose game mode
    game_mode_input = input("1: Basic\n"
                            "2: Defend_the_center\n")
    if game_mode_input == "1":
        game_mode = "basic"
        initial_ammo = 50
    elif game_mode_input == "2":
        game_mode = "defend_the_center"
        initial_ammo = 26

    # Choose new or previous model
    model_input = input("1: Load previous checkpoint\n"
                        "2: New model\n")
    if model_input == "1":  # Previous model
        date = input("DD-MM-YYYY: ")
        time = input("HH-MM: ")
        new = False
    if model_input == "2":  # New model
        date = datetime.datetime.now().strftime("%d-%m-%Y")
        time = datetime.datetime.now().strftime("%H-%M")
        new = True
    generic_path = "./models/" + game_mode + "/" + network + "/" + date + "-" + time
    log_path = generic_path + "-model.ckpt"
    json_path = generic_path + "log.txt"
    writer_path = "/tensorboard/" + game_mode + "/" + network + "/" + date + "/" + time
    return game_mode, network, initial_ammo, new, log_path, json_path, writer_path

def log_hyperparameters(writer, hyperpara_dict):
    summary_writer = writer
    for k, v in hyperpara_dict.items():
        value = k + v
        text_tensor = tf.make_tensor_proto(value, dtype=tf.string)
        meta = tf.SummaryMetadata().plugin_data.plugin_name = "text"
        summary = tf.Summary().value.add(tag="Hyper Parameters", metadata=meta, tensor=text_tensor)
        summary_writer.add_summary(summary)
