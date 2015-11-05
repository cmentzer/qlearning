# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qv = util.Counter() 

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #if we have seen this state, return its qv
        if (state, action) in self.qv:
          return self.qv[(state,action)]
        # else return 0
        else: 
          return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        # " if there are no legal actions, return 0.0 "
        if len(self.getLegalActions(state)) == 0:
          return 0.0
        # otherwise
        # init output to be none
        out = None
        # initialize our best Q value to a very negative number
        oldQ = -1000000
        # for all possible actions, 
        for a in self.getLegalActions(state):
          # get the new q value
          newQ = self.getQValue(state, a)
          # compare them, if the new is better, save it. 
          if newQ > oldQ:
            oldQ = newQ
        # return the best Q
        return oldQ

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        # " if there are no legal actions, return None " 
        if len(self.getLegalActions(state)) == 0:
          return None

        # otherwise
        # initialize a list of the best actions to keep track of ties
        # and a variable to track max qvalue
        oldQ = -1000000
        bestAs = []

        # for each possible action
        for a in self.getLegalActions(state):
          # get the new q value
          newQ = self.getQValue(state, a)
          # compare it to the old Q value. if newQ is better save it
          if newQ > oldQ:
            oldQ = newQ
            # replace the list of actions with this (better) action
            bestAs = []
            bestAs.append(a)
          # if the two are equal, do the same thing but add to list instead of replace
          if newQ == oldQ:
            bestAs.append(a)
        # return a random element from the best actions list
        if len(bestAs) == 1:
          return bestAs[0]
        else:
          return random.choice(bestAs)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # " if there are no legal actions, return None " 
        if len(self.getLegalActions(state)) == 0:
          return None

        # we want to pick a random action epsilon percent of the time
        if (util.flipCoin(self.epsilon)):
          return random.choice(self.getLegalActions(state))
        # otherwise return one of the best actions
        else: 
          return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        # the qvalue is given by our getQValue function, 
        qvalue = self.getQValue(state, action)
        # the learning rate, alpha
        a = self.alpha
        # discount 
        discount = self.discount
        # next state's value
        nextQ = self.computeValueFromQValues(nextState)

        # use formula Qt+1 = ((reward + discount * future value) - qvalue) * a
        self.qv[(state,action)] = ((reward + discount * nextQ) - qvalue) * a + qvalue

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # featureVector * w
        # multiply weights by the feature extraction of the state and action
        return self.weights * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        #get the features from this state
        features = self.featExtractor.getFeatures(state, action)
        # the qvalue is given by our getQValue function, 
        qvalue = self.getQValue(state, action)
        # discount 
        discount = self.discount
        # next state's value
        nextQ = self.computeValueFromQValues(nextState)

        # for each feature, compute the weights using the formula
        # weight = old weight + alpha * (reward + discount + nextQValue) 
        # - qvalue * features[f]
        for f in features:
          self.weights[f] = self.weights[f] + \
          (self.alpha * ((reward + discount * nextQ) - qvalue) * features[f])

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
