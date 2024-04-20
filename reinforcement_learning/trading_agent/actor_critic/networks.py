import os
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.activations import leaky_relu, tanh
from tensorflow.keras.layers import Dense, LSTM
import tensorflow_probability as tfp

def my_custom_leaky_relu(x):
    return leaky_relu(x, alpha=0.2)

def my_custom_tanh(x):
    return 5*tanh(x)

class ActorCriticNetwork(Model):
    def __init__(self, lstm1_dims=256, lstm2_dims=128,  fc_dims1=64, fc_dims2=5,
                base_dir='trading_agent/actor_critic/tmp/', chkpt_dir='model_x', name='trading_bot_AC',):
        super(ActorCriticNetwork, self).__init__()
        self.lstm1_dims = lstm1_dims
        self.lstm2_dims = lstm2_dims
        self.fc_dims1 = fc_dims1
        self.fc_dims2 = fc_dims2
        self.model_name = name
        self.base_dir = base_dir
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.base_dir, self.checkpoint_dir, name)

        self.lstm1 = LSTM(self.lstm1_dims, return_sequences=True, activation='sigmoid')
        self.lstm2 = LSTM(self.lstm2_dims,  activation='tanh')
        self.fc1  = Dense(self.fc_dims1,  activation='tanh')
        self.fc2  = Dense(self.fc_dims2,  activation=my_custom_tanh)
        self.v   = Dense(1, trainable=True, activation=None)

    def call(self, state:tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        marketPrice, slAndTp, entryPrice = state

        # state shape will be: [batch, [open,high,low,close], N] == [batch, 4, N]
        # print(f"marketPrice shape: {marketPrice.shape}")
        # print(f"slAndTp shape: {slAndTp.shape}")

        # batch_size = state.shape[0]

        value = self.lstm1(marketPrice) # shape == [batch, N, lstm1_dims]
        # print(f"value1 shape: {value.shape}")
        value = self.lstm2(value) # shape == [batch, lstm2_dims]

        value = tf.concat([value, slAndTp, entryPrice], axis=1)

        value = self.fc1(value) # shape == [batch, fc_dims1]
        # print(f"value2 shape: {value.shape}")
        value = self.fc2(value)    # shape == [batch, fc_dims2]

        mu  = tf.slice(value, [0, 0], [1, 2]) # shape == [batch, 2]
        cov = tf.slice(value, [0, 1], [1, 4]) # shape == [batch, 4]
        cov = tf.reshape(cov, [1, 2, 2]) # shape == [batch, 2, 2]

        dist = tfp.distributions.MultivariateNormalTriL(
            loc=mu,
            scale_tril=cov
        )

        v = self.v(value)

        return v, dist

