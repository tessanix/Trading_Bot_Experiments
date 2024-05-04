import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Layer, Dense, ReLU, Dropout


# https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras/
# https://ai.stackexchange.com/questions/41505/which-situation-will-helpful-using-encoder-or-decoder-or-both-in-transformer-mod

def positional_encoding(length, depth):
    # print("lentgh", depth)
    depth = depth/2
    # print("lentgh", length)
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
    # print("depths", depths.shape)
    # print("positions", positions.shape)
    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    # print("pos_encoding", pos_encoding.shape)

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(Layer):
  def __init__(self,  length, d_model): #, vocab_size,):
    super().__init__()
    self.d_model = d_model
    # self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=length, depth=d_model)

#   def compute_mask(self, *args, **kwargs):
#     return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    # length = tf.shape(x)[1]
    # x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    # print(f"x shape: {tf.shape(x)}")
    # pos_enc = self.pos_encoding[tf.newaxis, :length, :]
    # print(f"pos_enc shape: {tf.shape(pos_enc)}")
    x = x + self.pos_encoding
    return x
  
# Implementing the Add & Norm Layer
class AddNormalization(Layer):
    def __init__(self):
        super(AddNormalization, self).__init__()
        self.layer_norm = LayerNormalization()  # Layer normalization layer

    def call(self, x, sublayer_x):
        # The sublayer input and output need to be of the same shape to be summed
        add = x + sublayer_x

        # Apply layer normalization to the sum
        return self.layer_norm(add)

# Implementing the Feed-Forward Layer
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, ):
        super(FeedForward, self).__init__()
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU()  # ReLU activation layer

    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        x_fc1 = self.fully_connected1(x)

        return self.fully_connected2(self.activation(x_fc1))

# Implementing the Encoder Layer
class EncoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate): 
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(num_heads=h, key_dim=d_k, value_dim=d_v, dropout=rate)
        self.dropout1 = Dropout(rate)

        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
                
        self.dropout2 = Dropout(rate)

        self.add_norm2 = AddNormalization()

    def call(self, x, training:bool):
        # Multi-head attention layer

        multihead_output = self.multihead_attention(query=x, value=x, key=x, training=training)
        # Expected output shape = (batch_size, sequence_length, d_model)
        # Followed by an Add & Norm layer
        multihead_output = self.dropout1(multihead_output, training=training)

        addnorm_output = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)
        # Expected output shape = (batch_size, sequence_length, d_model)
        feedforward_output = self.dropout2(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)

# Implementing the Encoder
class Encoder(Layer):
    def __init__(self, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate): 
        super(Encoder, self).__init__()
        self.pos_encoding = PositionalEmbedding(sequence_length, d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

    def call(self, input_sentence, training:bool):
        # Generate the positional encoding
        x = self.pos_encoding(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)
        x = self.dropout(x, training=training)

        # Pass on the positional encoded values to each encoder layer
        for layer in self.encoder_layer:
            x = layer(x, training=training)

        return x
    


class ActorNetworkTransformerCategorical(Model):
    def __init__(self,  fc_dims1=1, fc_dims2=128, fc_dims3=3, base_dir='trading_agent/ppo/tmp/', chkpt_dir='model_x', name='trading_bot_AC', encoderParams={}):
        super(ActorNetworkTransformerCategorical, self).__init__()
        self.fc_dims1 = fc_dims1
        self.fc_dims2 = fc_dims2
        self.fc_dims3 = fc_dims3

        self.model_name = name
        self.base_dir = base_dir
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.base_dir, self.checkpoint_dir, name)

        self.encoder  = Encoder(**encoderParams)
        self.fc1      = Dense(self.fc_dims1,  activation='relu')
        self.dropout1 = Dropout(encoderParams['rate'])
        self.fc2      = Dense(self.fc_dims2,  activation='relu')
        self.dropout2 = Dropout(encoderParams['rate'])
        self.fc3      = Dense(self.fc_dims3,  activation='tanh')

    def call(self, state:tuple[tf.Tensor, tf.Tensor, tf.Tensor], training:bool):
        marketPrice, slAndTp, entryPrice = state

        # state shape will be: [batch, [open,high,low,close], N] == [batch, 4, N]
        # print(f"marketPrice shape: {marketPrice.shape}")
        # print(f"slAndTp shape: {slAndTp.shape}")

        # batch_size = state.shape[0]

        value = self.encoder(marketPrice) # shape == [batch, N, 4]
        # print(f"value1 shape: {value.shape}")
        value = self.fc1(value) # shape == [batch, N, 1]
        value = self.dropout1(value, training=training)

        value = tf.squeeze(value, axis=2) # shape == [batch, N]

        value = tf.concat([value, slAndTp, entryPrice], axis=1)

        # print(f"value2 shape: {value.shape}")
        value = self.fc2(value)    # shape == [batch, fc_dims2]
        value = self.dropout2(value, training=training)

        pi = self.fc3(value)    # shape == [batch, fc_dims3]
  
        return pi
    
    
class CriticNetworkTransformerCategorical(Model):
    def __init__(self,  fc_dims1=1, fc_dims2=128, fc_dims3=3, base_dir='trading_agent/ppo/tmp/', chkpt_dir='model_x', name='trading_bot_AC', encoderParams={}):
        super(CriticNetworkTransformerCategorical, self).__init__()
        self.fc_dims1 = fc_dims1
        self.fc_dims2 = fc_dims2
        self.fc_dims3 = fc_dims3

        self.model_name = name
        self.base_dir = base_dir
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.base_dir, self.checkpoint_dir, name)

        self.encoder = Encoder(**encoderParams)
        self.fc1      = Dense(self.fc_dims1,  activation='relu')
        self.dropout1 = Dropout(encoderParams['rate'])
        self.fc2      = Dense(self.fc_dims2,  activation='relu')
        self.dropout2 = Dropout(encoderParams['rate'])
        self.v        = Dense(1, activation=None)


    def call(self, state:tuple[tf.Tensor, tf.Tensor, tf.Tensor], training:bool):
        marketPrice, slAndTp, entryPrice = state

        # state shape will be: [batch, [open,high,low,close], N] == [batch, 4, N]
        # print(f"marketPrice shape: {marketPrice.shape}")
        # print(f"slAndTp shape: {slAndTp.shape}")

        # batch_size = state.shape[0]

        value = self.encoder(marketPrice) # shape == [batch, N, 4]
        # print(f"value1 shape: {value.shape}")
        value = self.fc1(value) # shape == [batch, N, 1]
        value = self.dropout1(value, training=training)

        value = tf.squeeze(value, axis=2) # shape == [batch, N]
        # print(f"value shape:{value.shape}, slAndTp shape:{slAndTp.shape}, entryPrice shape:{entryPrice.shape}")

        value = tf.concat([value, slAndTp, entryPrice], axis=1)

        # print(f"value2 shape: {value.shape}")
        value = self.fc2(value)    # shape == [batch, fc_dims2]
        value = self.dropout2(value, training=training)
  
        v = self.v(value)

        return v
    