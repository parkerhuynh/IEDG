import tensorflow_text as tf_text
import tensorflow as tf
from config import data_config, model_config
import pandas as pd
from utils import tf_lower_and_split_punct
from Attention_Model import  TrainTranslator, Translator, ShapeChecker
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
from loss import MaskedLoss

data = pd.read_csv(data_config["data_dir"])
buffer_size = len(data)
dataset = tf.data.Dataset.from_tensor_slices((data["text"], data["query"])).shuffle(buffer_size)


dataset = dataset.batch(data_config["batch_size"])

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

train_translator = TrainTranslator(
    model_config["embedding_dim"], model_config["hidden_units"],
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor)

train_translator.compile(
    optimizer=tf.optimizers.Adam(),
    loss=MaskedLoss(),
)
training_history = train_translator.fit(dataset.take(1),
                                        epochs=model_config["epoch"])


plot_history(training_history)
plt.savefig(model_config["save_dir"] +model_config["model_name"]+"png")
translator = Translator(
    encoder=train_translator.encoder,
    decoder=train_translator.decoder,
    input_text_processor=input_text_processor,
    output_text_processor=output_text_processor,
)
tf.saved_model.save(translator, model_config["save_dir"] +model_config["model_name"],
                    signatures={'serving_default': translator.tf_translate})