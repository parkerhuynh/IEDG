from tensorflow.keras import callbacks
import tensorflow_text as tf_text
import tensorflow as tf
from config import data_config, model_config
import pandas as pd
from utils import tf_lower_and_split_punct, plot_wer, plot_accuracy
from Attention_Model import  TrainTranslator, Translator, ShapeChecker
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
from loss import MaskedLoss
from callback import BatchLogs
from wer import WordErrorRate
from accuracy import Accuracy
import time
import datetime
import json
import os

logtime = datetime.datetime.now()


if __name__ == '__main__':
    #Load data
    log = {}
    data = pd.read_csv(data_config["data_dir"])
    training_set = data[:len(data)-data_config["test_sample"]]
    test_set = data[len(data)-data_config["test_sample"]:]
    buffer_size = len(training_set)

    training_ds = tf.data.Dataset.from_tensor_slices((training_set["text"], training_set["query"])).shuffle(buffer_size)
    training_ds = training_ds.batch(data_config["batch_size"])

    #Text processing
    input_text_processor = tf.keras.layers.TextVectorization(
        standardize= tf_lower_and_split_punct,
        max_tokens= data_config["max_vocab_size"])
    input_text_processor.adapt(data["text"])

    #CDQL processing
    output_text_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=data_config["max_vocab_size"])
    output_text_processor.adapt(data["query"])
    
    #Generate model
    train_translator = TrainTranslator(
        model_config["embedding_dim"], model_config["hidden_units"],
        input_text_processor=input_text_processor,
        output_text_processor=output_text_processor)
    output_text_processor.adapt(data["query"])
    
    #Compile model
    opt = tf.keras.optimizers.Adam(learning_rate=model_config["learning_rate"])
    train_translator.compile(
        optimizer=opt,
        loss=MaskedLoss()
    )

    #WER
    #test_wer = WordErrorRate(test_set, input_text_processor, output_text_processor)
    #train_wer = WordErrorRate(training_set, input_text_processor, output_text_processor)
    
    #WER
    test_accuracy = Accuracy(test_set, input_text_processor, output_text_processor)
    train_accuracy = Accuracy(training_set, input_text_processor, output_text_processor)

    #Train model
    batch_loss = BatchLogs('batch_loss')
    start_time = time.time()
    training_history = train_translator.fit(training_ds.take(1), epochs=model_config["epoch"], callbacks=[batch_loss, train_accuracy, test_accuracy])
    training_time = time.time() - start_time

    #Save trainning history
    os.mkdir(model_config["save_dir"] + model_config["model_name"] + "/")
    plot_history(training_history)
    plt.savefig(model_config["save_dir"] + model_config["model_name"] + "/epochLosses")
    plt.show()

    plt.plot(batch_loss.logs)
    plt.xlabel('Batch')
    plt.title('Batch loss')
    plt.ylabel('Batch loss')
    plt.savefig(model_config["save_dir"] + model_config["model_name"] + "/batchLosses")
    plt.show()

    #plot_wer(train_wer.wer, test_wer.wer)
    #plt.savefig(model_config["save_dir"] + model_config["model_name"] + "/wer")

    plot_accuracy(train_accuracy.accuracy, test_accuracy.accuracy)
    plt.savefig(model_config["save_dir"] + model_config["model_name"] + "/accuracy")

    #save Translator
    translator = Translator(
        encoder=train_translator.encoder,
        decoder=train_translator.decoder,
        input_text_processor=input_text_processor,
        output_text_processor=output_text_processor,
    )
    log[str(logtime)] = {}
    log[str(logtime)]["data_config"] = data_config
    log[str(logtime)]["model_config"] = model_config
    log[str(logtime)]["trainning_time"] = training_time
    with open(model_config["save_dir"] + model_config["model_name"] +'/logs.txt', 'w') as outfile:
        json.dump(log, outfile)
    tf.saved_model.save(translator, model_config["save_dir"] +model_config["model_name"],
                        signatures={'serving_default': translator.tf_translate})