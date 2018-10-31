'''
NOTE:
This is an implementation of the Advantage actor-critic agent for Cartpole.
The main source for this one would be in this repo: https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
This was done by rlcode guys, similar to the one that gave us DQN-Cartpole earlier.
'''

# Import dependencies

import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

Episodes = 100

#NOTE: Start of the Cartpole Agent using A2C (Advantage Actor-Critic)
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False # For rendering the cartpole model
        self.load_model = False # Set if you want to load a previous checkpoint
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # Policy Gradient hyperparameters
        # NOTE: Read more on Policy Gradient

        self.discount_factor = 0.99 # For the entire update statement
        self.actor_lr = 0.001 # For the Optimizer of Actor
        self.critic_lr = 0.005 # Why is it higher? For stability?

        # Call the building blocks
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        # Check if we need to load a model
        if self.load_model:
            self.actor.load_weights("./save_model/cartpole_actor.h5")
            self.critic.load_weights("./save_model/cartpole_critic.h5")
    # We then create the Neural Network for the approximation of the actor and critic values
    # i.e. policy and value for the model.

    # NOTE: Actor module: Input of states and outputs the probability of an action (softmax)
    def build_actor(self):
        actor = Sequential() # Define our model
        actor.add(Dense(24 , input_dim = self.state_size, activation= 'relu', kernel_initializer= 'he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',kernel_initializer='he_uniform'))
        actor.summary()
        actor.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = self.actor_lr))
        return actor
    # NOTE: Critic module: Input is also state but the output is also state(linear)
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim = self.state_size, activation= 'relu', kernel_initializer= 'he_uniform'))
        critic.add(Dense(self.value_size,activation= 'linear', kernel_initializer= 'he_uniform'))
        critic.summary()
        critic.compile(loss = 'mse', optimizer= Adam(lr=self.critic_lr))# Loss is MSE since we want to give out a value and not a probability.
        return critic
    # NOTE: We do the function on how the agent will pick the next action and policy based on stochastics(probability)
    def get_action(self,state):
        policy = self.actor.predict(state,batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]
    # NOTE: We do the update for the network policy.
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1,self.value_size)) # Initialize the policy targets matrix
        advantages = np.zeros((1,self.action_size)) # Initialize the advantages matrix

        value = self.critic.predict(state)[0] # Get value for this state
        next_value = self.critic.predict(next_state)[0] # Get value for the next state

        # update the advantages and value tables if done
        if done:
            advantages[0][action] = reward - value # Basically, what do we gain by choosing the action, will it improve or worsen the advantage
            target[0][0] = reward # Fill in the target value to see if we can still improve it in the policy making
        else:
            advantages[0][action] = reward + self.discount_factor*(next_value) - value # If not yet done, then simply update for the current step.
            target[0][0] = reward + self.discount_factor*next_value
        # Once we are done with the episode, we then update the weights
        self.actor.fit(state,advantages,epochs=1,verbose=0)
        self.critic.fit(state,target,epochs=1,verbose=0)

if __name__ == '__main__':
    # TODO: Create an environment
    env = gym.make('CartPole-v1')
    # TODO: Get the action and state sizes
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    # TODO: Make the agent by calling the function earlier
    agent = A2CAgent(state_size,action_size)
    # TODO: Initialize our scores and episodes list
    scores, episodes = [], []

    # TODO: Create the training loop
    for e in range(Episodes):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state,[1,state_size])

        while not done:
            # Check if we want to render
            if agent.render:
                env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state,[1,state_size])
            # Give immediate penalty for an action that terminates the episode immediately, Since we want to maximize the time
            # Note that the max for the cartpole is 499 and it will reset, otherwise we keep the current score if it is not yet done, and if it ended we give a -100 reward
            reward = reward if not done or score == 499 else -100
            # We now train the model based on the results of our action taken
            agent.train_model(state,action,reward,next_state,done)
            score += reward
            state = next_state

            if done:
                score = score if score == 500.0 else score +100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes,scores,'b')
                pylab.savefig("./save_graph/A2C-CartPole.png")
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()
        if e % 50 ==0:
            agent.actor.save_weights("./save_model/cartpole_actor.h5")
            agent.critic.save_weights("./save_model/cartpole_critic.h5")
            print("episode: {} score: {}".format(e,score))
