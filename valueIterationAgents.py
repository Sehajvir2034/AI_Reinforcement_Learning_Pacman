# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
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
        
        for i in range(0,iterations):
            #make a copy of values
            copyOfValues = self.values.copy()
            states = self.mdp.getStates()
            # check if the state is a terminal state, if yes then continue
            for state in states:
                if self.mdp.isTerminal(state):
                    continue
                # initialize a value max to compare
                max_value = float("-inf")
                for action in self.mdp.getPossibleActions(state):
                    tempValues = self.getQValue(state,action)
                    # compute max between the max_value computed before and the value we get from qvalues
                    max_value = max(max_value,tempValues)

                copyOfValues[state] = max_value

            self.values = copyOfValues



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
        #initialize qvalue with 0
        Qvalue = 0
        # Initialize transition states to use next state and probability in the equation
        # which is modified in the for loop further
        TransitionStates = self.mdp.getTransitionStatesAndProbs(state,action)

        # Add up qvalues as we iterate to the next state
        for nextState ,probability in TransitionStates:
            Qvalue += probability * (self.mdp.getReward(state, action, nextState) + (self.discount * self.values[nextState]))

        #return the accumulated qvalue computed using the values
        return Qvalue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = util.Counter()
        #check about the terminal state, if true return None
        legalActions = self.mdp.getPossibleActions(state)
        if len(legalActions) == 0:
            return None
        # Possible actions for the state
        possibleActions = self.mdp.getPossibleActions(state)
        # append qvalues into a array for further arg max implementation to
        # get the bast action that is the policy
        for action in possibleActions:
            actions[action] = (self.getQValue(state,action))

        policy = actions.argMax()
        #return the policy (best action) using argmax
        return policy

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
