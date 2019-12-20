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


def log_episode_tensorboard(writer, episode, explore_probability, total_reward, ammo_used, monsters_killed, accuracy):
    log_titles = ['explore_probability', 'total_reward', 'ammo_used', 'monsters_killed', 'accuracy']
    log_values = [explore_probability, total_reward, ammo_used, monsters_killed, accuracy]
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