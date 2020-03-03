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
    accuracy = int(accuracy)
    return accuracy

def log_episode_csv(path,
                     episode,
                     explore_probability,
                     total_reward,
                     ammo_used,
                     monsters_killed,
                     accuracy,
                     loss):
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.datetime.now().strftime("%H-%M")
    output_data = {'episode': str(episode),
                   'total_reward': str(int(total_reward)),
                   'loss': str(round(loss,4)),
                   'explore_probability': str(round(explore_probability,4)),
                   'ammo_used': str(int(ammo_used)),
                   'monsters_killed': str(int(monsters_killed)),
                   'accuracy': str(accuracy),
                   'date': current_date,
                   'time': current_time}
    line = ""
    for key, value in output_data.items():
        line = line + value + ","
    #output_data_as_json = json.dumps(output_data)  # dictionary to json
    with open(path, 'a') as outfile:
        outfile.write(line)
        outfile.write("\n")
def log_episode_json(path,
                     episode,
                     explore_probability,
                     total_reward,
                     ammo_used,
                     monsters_killed,
                     accuracy,
                     loss):
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.datetime.now().strftime("%H-%M")
    output_data = {'episode': episode,
                   'total_reward': str(int(total_reward)),
                   'loss': str(round(loss,4)),
                   'explore_probability': str(round(explore_probability,4)),
                   'ammo_used': str(int(ammo_used)),
                   'monsters_killed': str(int(monsters_killed)),
                   'accuracy': str(accuracy),
                   'date': current_date,
                   'time': current_time}
    output_data_as_json = json.dumps(output_data)  # dictionary to json
    with open(path, 'a') as outfile:
        outfile.write(output_data_as_json)
        outfile.write("\n")


def log_episode_tensorboard(writer,
                            episode,
                            explore_probability,
                            total_reward,
                            ammo_used,
                            monsters_killed,
                            accuracy,
                            loss):
    log_titles = ['loss', 'explore_probability', 'total_reward', 'ammo_used', 'monsters_killed', 'accuracy']
    log_values = [loss, explore_probability, total_reward, ammo_used, monsters_killed, accuracy]
    for log, values in zip(log_titles, log_values):
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=log, simple_value=values)])
        writer.add_summary(summary, episode)
    writer.flush()


def log_episode_std_out(loss, episode, explore_probability, total_reward, ammo_used, monsters_killed, accuracy):
    print('Episode: {}'.format(episode),
          'Total reward: {:.0f}'.format(total_reward),
          'Training loss: {:.4f}'.format(loss),
          'Explore P: {:.4f}'.format(explore_probability),
          'Ammo used: {:.0f}'.format(ammo_used),
          'monsters killed: {:.0f}'.format(monsters_killed),
          'Accuracy: {:.0f}'.format(accuracy))


def get_variables():
    game_mode = ""  # defend_the_center
    initial_ammo = 0  # basic = 50 def = 26
    network = ""  # DQN DDDQN PolicyGradient
    new = False

    # Choose network
    network_input = input("1: DQN\n"
                          "2: DDDQN\n"
                          "3: Policy Gradient\n"
                          "4: A3C\n")
    if network_input == "1":
        network = "DQN"
    elif network_input == "2":
        network = "DDDQN"
    elif network_input == "3":
        network = "PolicyGradient"
    elif network_input == "4":
        network = "A3C"

    # Choose game mode
    game_mode_input = input("1: Basic\n"
                            "2: Defend_the_center\n"
                            "3: predict_position\n"
                            "4: defend_the_line\n")
    if game_mode_input == "1":
        game_mode = "basic"
        initial_ammo = 50
    elif game_mode_input == "2":
        game_mode = "defend_the_center"
        initial_ammo = 26
    elif game_mode_input == "3":
        game_mode = "predict_position"
        initial_ammo = 1
    elif game_mode_input == "4":
        game_mode = "defend_the_line"
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
    for hyperparameter, value in hyperpara_dict.items():
        text_to_write = hyperparameter + ": " + value
        text_tensor = tf.compat.v1.make_tensor_proto(text_to_write, dtype=tf.string)
        meta = tf.compat.v1.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.compat.v1.Summary()
        summary.value.add(tag="Hyper Parameters", metadata=meta, tensor=text_tensor)
        summary_writer.add_summary(summary)

def log_episode(writer, hyperpara_dict,episode):
    #summary_writer = writer
    for keys, values in hyperpara_dict.items():
        #text_to_write = hyperparameter + ": " + value
        #text_tensor = tf.compat.v1.make_tensor_proto(text_to_write, dtype=tf.string)
        #meta = tf.compat.v1.SummaryMetadata()
        #meta.plugin_data.plugin_name = "text"
        #summary = tf.compat.v1.Summary()
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=keys, simple_value=values)])
        #summary.value.add(tag="Hyper Parameters", metadata=meta, tensor=text_tensor)
        writer.add_summary(summary, episode)


