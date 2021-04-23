# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey

X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
    """
    This agent jumps randomly.
    """

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        # We initialize our Q-value grid that has an entry for each action and
        # state.
        # (action, rel_x, rel_y, other features)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE,Y_SCREEN // Y_BINSIZE,4))#, ,Y_SCREEN // Y_BINSIZE4 gravity options available -> binned.Y_SCREEN // Y_BINSIZE, 2
        
        #hyperparameters
        self.eta = 0.2
        self.epsilon = 0.005
        self.gamma = 1#for this problem seems like an infinte problem.

        #need to init gravity
        self.gravity = 0

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)#wdistance to tree
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)#distance to top of tree
        
        #adding more features
        #print((state['monkey']['bot']))
        #abs_y_monkey = int((state['monkey']['bot']) // Y_BINSIZE)
        abs_vel_monkey = int((state['monkey']['vel']) // Y_BINSIZE)
        if (self.last_state != None) and (self.last_state != state) and (self.last_action !=1):
            self.gravity = int ((state['monkey']['vel']-self.last_state['monkey']['vel']))
        ##print(self.gravity)
        return (rel_x, rel_y, abs_vel_monkey,self.gravity)#abs_y_monkey,

    def Q_policy(self, ds):
        return np.argmax(self.Q[:,ds[0],ds[1],ds[2],ds[3]])

    def action_policy(self,state, ds):
        if npr.rand() > self.epsilon/(1+np.exp(0.01*(state['score']-250))):#Q action with prob 1-eps #implemented decay epsilon
            action = self.Q_policy(ds)
        else:#random action with prob eps
            action = npr.randint(2)
        return int(action)

    def learnQ(self, ds, prev_ds):
        TD = self.last_reward + self.gamma*np.max(self.Q[:,ds[0],ds[1],ds[2],ds[3]]) - self.Q[self.last_action,prev_ds[0],prev_ds[1],prev_ds[2],prev_ds[3]]#temporal difference
        return TD

    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        # TODO (currently monkey just jumps around randomly)
        # 1.  Discretize 'state' to get your transformed 'current state'
        # features.
      
        ds = self.discretize_state(state)

        # 2.  Perform the Q-Learning update using 'current state' and the 'last
        # state'.
      
       
        if self.last_state != None and (self.last_action == 0 or self.last_action ==1):
            prev_ds = self.discretize_state(self.last_state)
            self.Q[self.last_action,prev_ds[0],prev_ds[1],prev_ds[2],prev_ds[3]] += self.eta*self.learnQ(ds,prev_ds)

        # 3.  Choose the next action using an epsilon-greedy policy.
        new_action = self.action_policy(state, ds) #decides on action based on epsilon greedy model.
        new_state = state
   
        self.last_action = new_action
        self.last_state = new_state

        return self.last_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        #print("New Game")
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games.  You can update t_len to be smaller to run it faster.
    run_games(agent, hist, 100, 0)
    print(hist)

    # Save history.
    np.save('hist', np.array(hist))
