import numpy as np
import tensorflow as tf 
import tensorflow_probability as tfp
from tensorflow.python.keras.losses import MSE
from tensorflow.python.keras.optimizer_v2.adam import Adam
from reinforcement_learning.tests.CartPoleGame_with_PPO.networks import ActorNetwork, CriticNetwork
from reinforcement_learning.tests.CartPoleGame_with_PPO.memory import PPOMemory

class Agent:
    def __init__(self, n_actions, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        
        self.gamma       = gamma
        self.n_epochs    = n_epochs
        self.gae_lambda  = gae_lambda
        self.policy_clip = policy_clip

        self.memory = PPOMemory(batch_size)
        self.actor  = ActorNetwork(n_actions)
        self.critic = CriticNetwork()

        self.actor.compile(optimizer=Adam(learning_rate=alpha)) 
        self.critic.compile(optimizer=Adam(learning_rate=alpha)) 

       
    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_model()
        self.critic.save_model()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights()
        self.critic.load_weights()

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])

        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs=probs)

        value = self.critic(state)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        log_prob = log_prob.numpy()[0]
        action = action.numpy()[0]
        value = value.numpy()[0]

        return action, log_prob, value
    
    @tf.function
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]* (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:

                    states    = tf.convert_to_tensor(state_arr[batch])
                    actions   = tf.convert_to_tensor(action_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])

                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs=probs)
                    critic_value = self.critic(states)

                    critic_value = tf.squeeze(critic_value, 1)

                    new_probs = dist.log_prob(actions)
                    prob_ratio = tf.math.exp(new_probs - old_probs)

                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(
                                                prob_ratio, 
                                                1-self.policy_clip,
                                                1+self.policy_clip
                                            )
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    critic_loss = MSE(critic_value, returns)
                
                # actor_params = self.actor.trainable_variables
                # critic_params = self.critic.trainable_variables
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

                print(type(actor_grads[0]))
                print(type(self.actor.trainable_variables[0]))

                actor_params = [tf.Variable(tr) for tr in self.actor.trainable_variables]
                critic_params = [tf.Variable(tr) for tr in self.critic.trainable_variables]

                self.actor.optimizer.apply_gradients(zip(actor_grads, self.critic.trainable_variables))
                self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        self.memory.clear_memory()         