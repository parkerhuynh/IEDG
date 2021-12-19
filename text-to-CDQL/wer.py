from Attention_Model import  Translator
import tensorflow as tf
import pandas as pd
from jiwer import wer

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
      if len(self.dataset) > 300:
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
          tr = tr.numpy().decode()
          predictions.append(tr)
      wer_score = wer(outputs, predictions)
      self.wer.append(wer_score)
      print()
      print(f"{data_type} WER: {wer_score}")
      return self.wer