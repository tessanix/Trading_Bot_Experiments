import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam
from reinforcement_learning.trading_agent.actor_critic.networks import ActorCriticNetwork


class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99):
        self.gamma = gamma
        self.action = None

        self.actor_critic = ActorCriticNetwork()
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha)) # type: ignore
    
    def choose_action(self, observation):
        # print(f"observation shape: {observation.shape}")
        _, dist = self.actor_critic(observation)    
       
        action = dist.sample()
        # print(f"action shape: {action}")
        self.action = action
        a1, a2 = action[0][0], action[0][1]
        # print(f'action1: {a1}, action2: {a2}')
        return a1, a2
    
    def save_models(self):
        print("--- saving models ---")
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    
    def updateSlAndTp(self, df:pd.DataFrame, _maxSlInPips:float, _maxTpInPips:float):

        slAndTp      = tf.convert_to_tensor([[_maxSlInPips, _maxTpInPips]])
        observation = tf.convert_to_tensor([df.to_numpy()])

        sl, tp = self.choose_action((observation, slAndTp))
        # tp belongs to [0, +inf[
        # sl belongs to [0, +inf[

        tp = tp*_maxTpInPips if 0 < tp else _maxTpInPips
       
        sl = sl*(-_maxSlInPips) + _maxSlInPips
       
        return sl, tp
    
    def learn(self, state, reward, state_, maxSlInPips, maxTpInPips, done):
        # print(f'state shape: {state.shape}')
        # print(f'state_ shape: {state_.shape}')
        slAndTp = tf.convert_to_tensor([[maxSlInPips, maxTpInPips]], dtype=tf.float32)
        state  = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)

        with tf.GradientTape() as tape:
            state_value, dist = self.actor_critic((state, slAndTp))
            state_value_, _ = self.actor_critic((state_, slAndTp))
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
            