import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax
from reinforcement_learning.trading_agent.ppo.memory import PPOMemory
from reinforcement_learning.trading_agent.ppo.encoderTransformerNetwork import ActorNetworkTransformerCategorical, CriticNetworkTransformerCategorical

def softmax_filtered( input, mask):
    masked_input = input + mask
    # print(f'masked_input: {masked_input}, mask: {mask}')
    return softmax(masked_input)


class Agent:
    def __init__(self, alpha=0.0001, gamma=0.8, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10,
                 base_dir='trading_agent/ppo/tmp/', chkpt_dir="model_x",
                 transParams={}, norm=True):
        
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.norm = norm

        self.memory = PPOMemory(batch_size)
        self.actor = ActorNetworkTransformerCategorical(base_dir=base_dir, chkpt_dir=chkpt_dir, name='actor_ppo.keras', encoderParams=transParams) 
        self.critic = CriticNetworkTransformerCategorical(base_dir=base_dir, chkpt_dir=chkpt_dir, name='critic_ppo.keras', encoderParams=transParams) 

        self.actor.compile(optimizer=Adam(learning_rate=alpha)) 
        self.critic.compile(optimizer=Adam(learning_rate=alpha)) 

    def store_transition(self, state, maxSlInPips, entryPrice, action, probs, vals, reward, done):
        self.memory.store_memory(state,  maxSlInPips, entryPrice, action, probs, vals, reward, done)

    def choose_action(self, _observation:np.ndarray, _maxSlInPips:float, _entryPrice:float, action_mask):
        if self.norm:
            _observation = self.minMaxNorm(350.0, 6000.0, _observation)
            _entryPrice = self.minMaxNorm(350.0, 6000.0, _entryPrice)
            _maxSlInPips = _maxSlInPips/1000.0

        observation   = tf.convert_to_tensor([_observation])
        slAndTpInPips = tf.convert_to_tensor([[_maxSlInPips]])
        entryPrice    = tf.convert_to_tensor([[_entryPrice]])
  
        probs = self.actor((observation, slAndTpInPips, entryPrice))    
        probs = softmax_filtered(probs, action_mask)
        dist = tfp.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        value = self.critic((observation, slAndTpInPips, entryPrice))

        # self.action = action
        return action[0], log_prob[0], value[0]
    
    def save_models(self):
        print("--- saving models ---")
        self.actor.save(self.actor.checkpoint_file)
        self.critic.save(self.critic.checkpoint_file)

    # def load_weights(self):
    #     print("--- loading models ---")
    #     self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def minMaxNorm(self, _min, _max, _data):
        return (_data-_min)/(_max-_min)    
    


    def learn(self):
        for _ in range(self.n_epochs):
            state_arr,  maxSlInPips_arr, entryPrice_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
            
            state_arr       = self.minMaxNorm(350.0, 6000.0, state_arr)
            entryPrice_arr  = self.minMaxNorm(350.0, 6000.0, entryPrice_arr)
            maxSlInPips_arr = maxSlInPips_arr/1000.0

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:

                    states        = tf.convert_to_tensor(state_arr[batch])
                    slAndTpInPips = tf.convert_to_tensor(maxSlInPips_arr[batch])
                    entryPrice    = tf.convert_to_tensor(entryPrice_arr[batch])
                    old_probs     = tf.convert_to_tensor(old_prob_arr[batch])
                    actions       = tf.convert_to_tensor(action_arr[batch])


                    slAndTpInPips = tf.expand_dims(slAndTpInPips, -1)
                    entryPrice    = tf.expand_dims(entryPrice, -1)

                    # print(f"states shape:{states.shape}, slAndTpInPips shape:{slAndTpInPips.shape}, entryPrice shape:{entryPrice.shape}")
                    probs = self.actor((states, slAndTpInPips, entryPrice))
                    dist = tfp.distributions.Categorical(softmax(probs))
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic((states, slAndTpInPips, entryPrice))

                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    # critic_loss = tf.math.reduce_mean(tf.math.pow(returns-critic_value, 2))
                    critic_loss = MSE(critic_value, returns)

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

        self.memory.clear_memory()
            