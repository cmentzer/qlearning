# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # keep track of the number of iterations we have done so far
        i = 0
        # final output value
        v = 0
        # get all the states
        states = mdp.getStates()
        # for each of the specified iterations:
        while i < iterations:
          # save the current self.values
          oldSV = self.values.copy()
          # increment our variable for number of iterations
          i = i + 1
          # for each of the states, 
          for s in states:
            # get the value at this state
            v = util.Counter()
            # look at all possible actions from that state
            actions = mdp.getPossibleActions(s)
            # for each state action pair ...
            for a in actions:
              # get the transition states and the probablilities of 
              # reaching those states
              tStatesAndProbs = mdp.getTransitionStatesAndProbs(s, a)
              # keep track of the number of pairs we have seen so far
              j = 0
              # print tStatesAndProbs
              # for each pair in tStatesAndProbs, 
              while j < len(tStatesAndProbs):
                # extract tState and Prob from this member of the list
                tState = tStatesAndProbs[j][0]
                prob = tStatesAndProbs[j][1]
                # set the value associated with that move
                # make sure to account for prob and discount
                v[a] = v[a] + (mdp.getReward(s, a, tState) + discount * oldSV[tState]) * prob
                # increment
                j = j + 1
            # return 
            self.values[s] = v[v.argMax()]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # get the discount
        discount = self.discount
        # get the values
        values = self.values
        # get the mdp
        mdp = self.mdp
        # set initial q value
        qv = 0
        # 
        tStatesAndProbs = mdp.getTransitionStatesAndProbs(state, action)
        # keep track of pairs seen so far
        j = 0 
        while j < len(tStatesAndProbs):
          # extract tState and Prob from this member of the list
          tState = tStatesAndProbs[j][0]
          prob = tStatesAndProbs[j][1]
          # calcuate the qv the same we we calculated v above
          qv = qv + ((discount * values[tState]) + mdp.getReward(state, action, tState)) * prob
          # increment
          j = j + 1

        return qv

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # get mdp 
        mdp = self.mdp
        # list the possible actions
        actions = mdp.getPossibleActions(state)
        # init v
        v = 0
        # init output
        out = None

        # "note that if there are no legal actions, return none"
        if len(actions) == 0:
          return None

        # for each a in actions, compute the value of that action and return the highest
        for a in actions:
          # if v is not set yet, set it
          if v == 0:
            v = self.computeQValueFromValues(state, a)
            out = a
          # otherwise compare new v and old v
          elif self.computeQValueFromValues(state, a) > v:
            v = self.computeQValueFromValues(state, a)
            out = a

        # return the best option
        return out

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
