import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt

def tf_lower_and_split_punct(text):
    # Split accecented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '\(\)[^ a-z.?!,¿="]', '')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

def plot_wer(trainning_wers, test_wers):
    x = range(len(trainning_wers))
    y1 = trainning_wers
    y2 = test_wers
    plt.plot(x, y1, label = "Training")
    plt.plot(x, y2, label = "Test")
    plt.title("Word Error Rate")
    plt.legend(["Training", "Test"])
    plt.xticks(x)
    plt.grid(axis='y')
    return plt

def plot_accuracy(trainning_accuracies, test_accuracies):
    x = range(len(trainning_accuracies))
    y1 = trainning_accuracies
    y2 = test_accuracies
    plt.plot(x, y1, label = "Training")
    plt.plot(x, y2, label = "Test")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training", "Test"])
    plt.xticks(x)
    plt.grid(axis='y')
    return plt