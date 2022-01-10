from Attention_Model import  Translator
import tensorflow as tf
import pandas as pd
import time
from jiwer import wer
from random import randrange
from config import data_config
from sklearn.metrics import accuracy_score

class Metrics(tf.keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""
    def __init__(self, dataset, input_text_processor, output_text_processor):
        super().__init__()
        self.dataset = dataset
        self.wer  = []
        self.accuracy  = []
        self.running_time  = 0
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

    def on_epoch_end(self, epoch: int, logs=None):
        start_time = time.time()
        translator = Translator(
            encoder=self.model.encoder,
            decoder=self.model.decoder,
            input_text_processor=self.input_text_processor,
            output_text_processor=self.output_text_processor)
        if len(self.dataset) > data_config["test_sample"]:
            dataset = self.dataset.sample(frac=1)[:data_config["test_sample"]]
            data_type = "Train"
        else:
            dataset = self.dataset
            data_type = "Test"

        inputs = tf.constant(dataset["text"])
        outputs = list(dataset["query"])
        predictions = []
        result = translator.tf_translate(inputs)['text']
        for i, tr in enumerate(result):
            outputs[i] = outputs[i].lower()
            tr = tr.numpy().decode()
            predictions.append(tr)
        random_id = randrange(data_config["test_sample"])
        #Compute WER
        wer_score = wer(outputs, predictions)
        self.wer.append(wer_score)
        
        #Compute WER
        acc = accuracy_score(outputs, predictions)
        self.accuracy.append(acc)

        #Print results
        print()
        print(f"{data_type} WER (Epoch {epoch + 1}): {wer_score}")
        print(f"{data_type} Accuracy (Epoch {epoch + 1}): {acc}")
        print(f"Target ({data_type} set): {outputs[random_id]}")
        print(f"Prediction ({data_type} set): {predictions[random_id]}")
        print()
        if data_type == "Test":
            print("*"*150)
        self.running_time += time.time() - start_time