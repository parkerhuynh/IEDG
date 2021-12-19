from Attention_Model import  Translator
import tensorflow as tf
import pandas as pd
from jiwer import wer
from random import randrange
from config import data_config

class WordErrorRate(tf.keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""
    def __init__(self, dataset, input_text_processor, output_text_processor):
        super().__init__()
        self.dataset = dataset
        self.wer  = []
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

    def on_epoch_end(self, epoch: int, logs=None):
        translator = Translator(
            encoder=self.model.encoder,
            decoder=self.model.decoder,
            input_text_processor=self.input_text_processor,
            output_text_processor=self.output_text_processor)
        if len(self.dataset) > data_config["test_sample"]:
            dataset = self.dataset.sample(frac=1)[:300]
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
        random_id = randrange(10)
        wer_score = wer(outputs, predictions)
        self.wer.append(wer_score)
        print()
        print(f"{data_type} WER (Epoch {epoch + 1}): {wer_score}")
        if data_type == "Test":
            print(f"Target: {outputs[random_id]}")
            print(f"Prediction: {predictions[random_id]}")
        return self.wer