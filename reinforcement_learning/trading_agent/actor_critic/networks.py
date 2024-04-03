import os
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.keras.layers import Dense, LSTM
import tensorflow_probability as tfp

class ActorCriticNetwork(Model):
    def __init__(self, lstm1_dims=1024, lstm2_dims=512,  fc_dims=5,
                 name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.lstm1_dims = lstm1_dims
        self.lstm2_dims = lstm2_dims
        self.fc_dims = fc_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')

        self.lstm1 = LSTM(self.lstm1_dims, return_sequences=True, activation='sigmoid')
        self.lstm2 = LSTM(self.lstm2_dims,  activation='sigmoid')
        self.fc  = Dense(self.fc_dims,  activation='sigmoid')
        self.v   = Dense(1, trainable=True, activation=None)

    def call(self, state:tf.Tensor):
        # state shape will be: [batch, [open,high,low,close], N] == [batch, 4, N]
        # print(f"state shape: {state.shape}")
        # batch_size = state.shape[0]

        value = self.lstm1(state) # shape == [batch, N, lstm1_dims]
        # print(f"value1 shape: {value.shape}")
        value = self.lstm2(value) # shape == [batch, lstm2_dims]
        # print(f"value2 shape: {value.shape}")
        value = self.fc(value)    # shape == [batch, fc_dims]

        mu  = tf.slice(value, [0, 0], [1, 2]) # shape == [batch, 2]
        cov = tf.slice(value, [0, 1], [1, 4]) # shape == [batch, 4]
        cov = tf.reshape(cov, [1, 2, 2]) # shape == [batch, 2, 2]

        dist = tfp.distributions.MultivariateNormalTriL(
            loc=mu,
            scale_tril=cov
        )

        v = self.v(value)

        return v, dist

