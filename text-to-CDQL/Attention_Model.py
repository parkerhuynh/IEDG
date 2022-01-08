from abc import abstractmethod
import numpy as np
import typing
from typing import Any, Tuple
import tensorflow as tf
import tensorflow_text as tf_text
from config import model_config

class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.input_vocab_size = input_vocab_size

        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size,
                                                embedding_dim)
        # The GRU RNN layer processes those vectors sequentially.
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                    # Return the sequence and state
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, tokens, state=None):
        # 2. The embedding layer looks up the embedding for each token.
        vectors = self.embedding(tokens)
        # 3. The GRU processes the embedding sequence.
        output, state = self.gru(vectors, initial_state=state)
        # 4. Returns the new sequence and its state.
        return output, state

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)
        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        w1_query = self.W1(query)
        w2_key = self.W2(value)
        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask
        context_vector, attention_weights = self.attention(
            inputs = [w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores = True,
        )
        return context_vector, attention_weights

class LuongAttention(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(hidden_dim) 

    def call(self,query,value):
        score = tf.matmul(query, self.wa(value), transpose_b=True)
        attention_weights = tf.keras.activations.softmax(score, axis=-1)
        context_vector = tf.matmul(attention_weights, value)
        return context_vector, attention_weights

class DecoderInput(typing.NamedTuple):
    new_tokens: Any
    enc_output: Any
    mask: Any

class DecoderOutput(typing.NamedTuple):
    logits: Any
    attention_weights: Any

class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,
                                                embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        if model_config["attention"] == "Bahdanau":
            self.attention = BahdanauAttention(self.dec_units)
        elif model_config["attention"] == "Luong":
            self.attention = LuongAttention(self.dec_units)
        elif model_config["attention"] == None:
            pass
        self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                        use_bias=False)
        self.fc = tf.keras.layers.Dense(self.output_vocab_size)
        
    def call(self,inputs: DecoderInput, state=None) -> Tuple[DecoderOutput, tf.Tensor]:
        vectors = self.embedding(inputs.new_tokens)
        rnn_output, state = self.gru(vectors, initial_state=state)
        if model_config["attention"] == None:
            logits = self.fc(rnn_output)
            attention_weights = None
        elif model_config["attention"] == "Bahdanau":
            context_vector, attention_weights = self.attention(query=rnn_output, value=inputs.enc_output, mask=inputs.mask)
            context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)
            attention_vector = self.Wc(context_and_rnn_output)
            logits = self.fc(attention_vector)
        elif model_config["attention"] == "Luong":
            context_vector, attention_weights = self.attention(query=rnn_output, value=inputs.enc_output)
            context_and_rnn_output = tf.concat([context_vector,rnn_output], axis=-1)
            attention_vector = self.Wc(context_and_rnn_output)
            logits = self.fc(attention_vector)
        return DecoderOutput(logits, attention_weights), state

class TrainTranslator(tf.keras.Model):
    def __init__(self, embedding_dim, units,
                input_text_processor,
                output_text_processor, 
                use_tf_function=True):
        super().__init__()
        encoder = Encoder(input_text_processor.vocabulary_size(),
                        embedding_dim, units)
        decoder = Decoder(output_text_processor.vocabulary_size(),
                        embedding_dim, units)
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.use_tf_function = use_tf_function

    def train_step(self, inputs):
        if self.use_tf_function:
            return self._tf_train_step(inputs)
        else:
            return self._train_step(inputs)
    def _preprocess(self, input_text, target_text):
        # Convert the text to token IDs
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)
        # Convert IDs to masks.
        input_mask = input_tokens != 0
        target_mask = target_tokens != 0
        return input_tokens, input_mask, target_tokens, target_mask

    def _train_step(self, inputs):
        input_text, target_text = inputs  
        (input_tokens, input_mask, target_tokens, target_mask) = self._preprocess(input_text, target_text)
        max_target_length = tf.shape(target_tokens)[1]
        with tf.GradientTape() as tape:
            enc_output, enc_state = self.encoder(input_tokens)
            dec_state = enc_state
            loss = tf.constant(0.0)
            for t in tf.range(max_target_length-1):
                new_tokens = target_tokens[:, t:t+2]
                step_loss, dec_state = self._loop_step(new_tokens, input_mask,enc_output, dec_state)
                loss = loss + step_loss
            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))
            variables = self.trainable_variables 
            gradients = tape.gradient(average_loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return {'batch_loss': average_loss}

    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]
        decoder_input = DecoderInput(new_tokens=input_token,
                                    enc_output=enc_output,
                                    mask=input_mask)
        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)
        y = target_token
        y_pred = dec_result.logits
        step_loss = self.loss(y, y_pred)
        return step_loss, dec_state
  
    @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),
                                tf.TensorSpec(dtype=tf.string, shape=[None])]])                    
    def _tf_train_step(self, inputs):
        return self._train_step(inputs)


#Extract model
class Translator(tf.Module):
    def __init__(self, encoder, decoder, input_text_processor,
                output_text_processor):
        self.encoder = encoder
        self.decoder = decoder
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

        self.output_token_string_from_index = (
            tf.keras.layers.StringLookup(
                vocabulary=output_text_processor.get_vocabulary(),
                mask_token='',
                invert=True))
        # The output should never generate padding, unknown, or start.
        index_from_string = tf.keras.layers.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token='')
        token_mask_ids = index_from_string(['', '[UNK]', '[START]']).numpy()

        token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=np.bool)
        token_mask[np.array(token_mask_ids)] = True
        self.token_mask = token_mask

        self.start_token = index_from_string(tf.constant('[START]'))
        self.end_token = index_from_string(tf.constant('[END]'))

    def tokens_to_text(self, result_tokens):
        result_text_tokens = self.output_token_string_from_index(result_tokens)
        result_text = tf.strings.reduce_join(result_text_tokens, axis=1, separator=' ')
        result_text = tf.strings.strip(result_text)
        return result_text

    def translate(self,input_text, *,
                    max_length=50,
                    return_attention=True,
                    temperature=1.0):
        batch_size = tf.shape(input_text)[0]
        input_tokens = self.input_text_processor(input_text)
        enc_output, enc_state = self.encoder(input_tokens)
        dec_state = enc_state
        new_tokens = tf.fill([batch_size, 1], self.start_token)
        result_tokens = []
        if model_config["attention"] != None:
            attention = []
        done = tf.zeros([batch_size, 1], dtype=tf.bool)

        for _ in range(max_length):
            dec_input = DecoderInput(new_tokens=new_tokens,
                                    enc_output=enc_output,
                                    mask=(input_tokens!=0))
            dec_result, dec_state = self.decoder(dec_input, state=dec_state)
            if model_config["attention"] != None:
                attention.append(dec_result.attention_weights)
            new_tokens = self.sample(dec_result.logits, temperature)
            # If a sequence produces an `end_token`, set it `done`
            done = done | (new_tokens == self.end_token)
            # Once a sequence is done it only produces 0-padding.
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)
            # Collect the generated tokens
            result_tokens.append(new_tokens)
            if tf.executing_eagerly() and tf.reduce_all(done):
                break
        # Convert the list of generates token ids to a list of strings.
        result_tokens = tf.concat(result_tokens, axis=-1)
        result_text = self.tokens_to_text(result_tokens)
        if return_attention and model_config["attention"] != None:
            attention_stack = tf.concat(attention, axis=1)
            return {'text': result_text, 'attention': attention_stack}
        else:
            return {'text': result_text}

    def sample(self, logits, temperature):
        # 't' is usually 1 here.
        token_mask = self.token_mask[tf.newaxis, tf.newaxis, :]
        # Set the logits for all masked tokens to -inf, so they are never chosen.
        logits = tf.where(self.token_mask, -np.inf, logits)
        if temperature == 0.0:
            new_tokens = tf.argmax(logits, axis=-1)
        else: 
            logits = tf.squeeze(logits, axis=1)
            new_tokens = tf.random.categorical(logits/temperature,
                                                num_samples=1)
        return new_tokens

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def tf_translate(self, input_text):
        return self.translate(input_text)