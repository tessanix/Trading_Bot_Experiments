import os
from tensorflow.python.keras import Model
from tensorflow.keras.layers import Dense

class ActorNetwork(Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(n_actions, activation='softmax')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        probs = self.fc3(value)
        return probs

    def save_model(self):
        print("--- saving model ---")
        self.save_weights(self.checkpoint_file)

    def load_weights(self):
        print("--- loading models ---")
        self.load_weights(self.checkpoint_file)




class CriticNetwork(Model):
    def __init__(self, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.fc3 = Dense(1)

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        value = self.fc3(value)
        return value

    def save_model(self):
        print("--- saving model ---")
        self.save_weights(self.checkpoint_file)

    def load_weights(self):
        print("--- loading models ---")
        self.load_weights(self.checkpoint_file)
