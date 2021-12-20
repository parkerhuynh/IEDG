from Attention_Model import  Translator
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score
from random import randrange
from config import data_config

class Accuracy(tf.keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""
    def __init__(self, dataset, input_text_processor, output_text_processor):
        super().__init__()
        self.dataset = dataset
        self.accuracy  = []
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

    def on_epoch_end(self, epoch: int, logs=None):
        translator = Translator(
            encoder=self.model.encoder,
            decoder=self.model.decoder,
            input_text_processor=self.input_text_processor,
            output_text_processor=self.output_text_processor)
        if len(self.dataset) > data_config["test_sample"]:
            dataset = self.dataset.sample(frac=1)[:data_config["test_sample"]]
            data_type = "Training"
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
        _accuracy = accuracy_score(outputs, predictions)
        self.accuracy.append(_accuracy)
        print()
        print(f"{data_type} Accuracy (Epoch {epoch + 1}): {_accuracy}")
        if data_type == "Test":
            print(f"Target: {outputs[random_id]}")
            print(f"Prediction: {predictions[random_id]}")
        return self.accuracy