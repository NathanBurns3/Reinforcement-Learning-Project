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
import collections

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
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Iterate over the number of iterations
        for i in range(self.iterations):
            # Copy the current values to a new variable
            newValues = self.values.copy()
            # Iterate over all states in the Markov Decision Process (MDP)
            for state in self.mdp.getStates():
                # If the state is a terminal state, skip to the next iteration
                if self.mdp.isTerminal(state):
                    continue
                # Compute the best action based on the current values
                bestAction = self.computeActionFromValues(state)
                # Compute the Q-value for the state and the best action, and update the new values
                newValues[state] = self.computeQValueFromValues(state, bestAction)
            # Update the values with the new values
            self.values = newValues

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
        # Initialize the Q-value to 0
        qValue = 0

        # Iterate over all possible next states and their probabilities given the current state and action
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # Update the Q-value based on the reward for the current state, action, and next state,
            # as well as the discounted value of the next state
            qValue += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])

        # Return the computed Q-value
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Check if the state is a terminal state
        if self.mdp.isTerminal(state):
            # If it is, return None as there are no actions to take in a terminal state
            return None

        # Initialize the best action to None and the best Q-value to negative infinity
        bestAction = None
        bestQValue = float('-inf')

        # Iterate over all possible actions in the current state
        for action in self.mdp.getPossibleActions(state):
            # Compute the Q-value for the current state and action
            qValue = self.computeQValueFromValues(state, action)
            # If the computed Q-value is better than the current best Q-value, update the best action and Q-value
            if qValue > bestQValue:
                bestQValue = qValue
                bestAction = action

        # Return the best action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Get all states from the Markov Decision Process (MDP)
        states = self.mdp.getStates()

        # Iterate over the number of iterations
        for i in range(self.iterations):
            # Select a state from the list of states, wrapping around if the number of iterations exceeds the number of states
            state = states[i % len(states)]
            
            # If the state is a terminal state, skip to the next iteration
            if self.mdp.isTerminal(state):
                continue
            
            # Compute the best action based on the current values
            bestAction = self.computeActionFromValues(state)
            
            # Compute the Q-value for the state and the best action, and update the value of the state
            self.values[state] = self.computeQValueFromValues(state, bestAction)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Initialize a dictionary to store the predecessors of each state
        predecessors = {}

        # Iterate over all states in the Markov Decision Process (MDP)
        for state in self.mdp.getStates():
            # Initialize the set of predecessors for the current state
            predecessors[state] = set()

        # Iterate over all states in the MDP again
        for state in self.mdp.getStates():
            # Iterate over all possible actions in the current state
            for action in self.mdp.getPossibleActions(state):
                # Iterate over all possible next states and their probabilities given the current state and action
                for nextState, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    # Add the current state to the set of predecessors of the next state
                    predecessors[nextState].add(state)

        # Initialize a priority queue to store states based on their value difference
        queue = util.PriorityQueue()

        # Iterate over all states in the MDP again
        for state in self.mdp.getStates():
            # If the state is not a terminal state
            if not self.mdp.isTerminal(state):
                # Compute the difference between the current value of the state and the maximum Q-value for any action in the state
                diff = abs(self.values[state] - max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]))
                # Add the state to the queue with priority equal to the negative difference (so states with larger differences are dequeued first)
                queue.push(state, -diff)

        # Iterate over the number of iterations
        for _ in range(self.iterations):
            # If the queue is empty, break the loop
            if queue.isEmpty():
                break
            # Dequeue a state from the queue
            state = queue.pop()
            # If the state is not a terminal state
            if not self.mdp.isTerminal(state):
                # Update the value of the state to the maximum Q-value for any action in the state
                self.values[state] = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
            # Iterate over all predecessors of the state
            for predecessor in predecessors[state]:
                # Compute the difference between the current value of the predecessor and the maximum Q-value for any action in the predecessor
                diff = abs(self.values[predecessor] - max([self.computeQValueFromValues(predecessor, action) for action in self.mdp.getPossibleActions(predecessor)]))
                # If the difference is greater than a threshold
                if diff > self.theta:
                    # Update the priority of the predecessor in the queue to the negative difference
                    queue.update(predecessor, -diff)