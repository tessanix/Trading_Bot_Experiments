import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam
from reinforcement_learning.trading_agent.actor_critic.networks import ActorCriticNetwork
from reinforcement_learning.trading_agent.actor_critic.encoderTransformerNetwork import ActorCriticNetworkTransformerCategorical


class Agent:
    def __init__(self, alpha=0.0001, gamma=0.99, base_dir='trading_agent/actor_critic/tmp/', chkpt_dir="model_x", name='trading_bot_AC', transformer=False, transParams={}, norm=True):
        self.gamma = gamma
        self.action = None
        self.norm = norm

        self.actor_critic = ActorCriticNetworkTransformerCategorical(base_dir=base_dir, chkpt_dir=chkpt_dir, name=name, encoderParams=transParams) if transformer else ActorCriticNetwork(base_dir=base_dir, chkpt_dir=chkpt_dir, name=name)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha)) 
    
    def choose_action(self, _observation:np.ndarray, _maxSlInPips:float, _entryPrice:float):
        if self.norm:
            _observation = self.minMaxNorm(350.0, 6000.0, _observation)
            _entryPrice = self.minMaxNorm(350.0, 6000.0, _entryPrice)
            _maxSlInPips = _maxSlInPips/1000.0

        observation   = tf.convert_to_tensor([_observation], dtype=tf.float32)
        slAndTpInPips = tf.convert_to_tensor([[_maxSlInPips]], dtype=tf.float32)
        entryPrice    = tf.convert_to_tensor([[_entryPrice]], dtype=tf.float32)
        # print(f"observation shape: {observation.shape}")
        _, dist = self.actor_critic((observation, slAndTpInPips, entryPrice))    
       
        action = dist.sample()
        # print(f"action shape: {action}")
        self.action = action
        # print(f'action: {action}')
        return action[0]
    
    def save_models(self):
        print("--- saving models ---")
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_weights(self):
        print("--- loading models ---")
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def minMaxNorm(self, _min, _max, _data):
        return (_data-_min)/(_max-_min)    
    
    
    def learn(self, state:np.ndarray, reward, state_:np.ndarray, maxSlInPips, _entryPrice, done):
        # print(f'state shape: {state.shape}')
        # print(f'state_ shape: {state_.shape}')

        if self.norm:
            _entryPrice = self.minMaxNorm(350.0, 6000.0, _entryPrice)
            state = self.minMaxNorm(350.0, 6000.0, state)
            state_ = self.minMaxNorm(350.0, 6000.0, state_)
            maxSlInPips = maxSlInPips/1000.0

        slAndTp    = tf.convert_to_tensor([[maxSlInPips]], dtype=tf.float32)
        entryPrice = tf.convert_to_tensor([[_entryPrice]], dtype=tf.float32)
        state      = tf.convert_to_tensor([state], dtype=tf.float32)
        state_     = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward     = tf.convert_to_tensor([reward], dtype=tf.float32)

        with tf.GradientTape() as tape:
            state_value, dist = self.actor_critic((state, slAndTp, entryPrice))
            state_value_, _ = self.actor_critic((state_, slAndTp, entryPrice))
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            log_prob = dist.log_prob(self.action)

            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            actor_loss = -log_prob*delta
            critic_loss = delta**2

            total_loss = actor_loss + critic_loss

                
            gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
            # gradient = [tf.Variable(grad)  for grad in gradient]
            trainables = [tf.Variable(tr)  for tr in self.actor_critic.trainable_variables]
            # for grad, var in zip(gradient, self.actor_critic.trainable_variables):
            #     print(f"Gradient shape: {tf.Variable(grad).shape}, Variable shape: {var.shape}")
            self.actor_critic.optimizer.apply_gradients(zip(gradient, trainables))
            