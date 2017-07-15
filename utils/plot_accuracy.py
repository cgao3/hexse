

def plot_training(self, accuracies_file):
    '''
    :param self:
    :param accuracies_file: path to train accuracy file
    :return:

    file format:
    each line contains two numbers, (train_step_num batch_accuracy)
    '''
    reader = open(accuracies_file, "r")
    accuracies = []
    prev_step = 0
    step_num = 0
    for line in reader:
        tmp = line.strip().split()
        prev_step = step_num
        step_num, acc = int(tmp[0]), float(tmp[1])
        accuracies.append(acc)
    reader.close()

    scale = step_num - prev_step
    import matplotlib.pyplot as plt
    fig, ax = plt.subplot()
    ax.plot(range(0, len(accuracies) * scale, scale), accuracies, "plain cnn")
    ax.set_xlabel('Training step')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0.1, 1.0])
    ax.set_title('plain CNN' + repr(self.num_hidden_layers) + ' hidden layer, 3x3-' +
                 repr(self.num_filters) + 'convolution,accuracy')
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_acc_path', type=str, default='')

    args = parser.parse_args()
    import os
    if not os.path.isfile(args.train_acc_path):
        print("please indicate train accuracy file path")
        exit(0)

    plot_training(args.train_acc_path)