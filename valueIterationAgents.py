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


import collections

import mdp
import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # repeat for every iteration
        for i in range(self.iterations):
            update_values = self.values.copy()

            # calculate the Q value for every state available
            for state in self.mdp.getStates():
                q_vals = [float('-inf')]

                # check if we at a terminal state
                if self.mdp.isTerminal(state):
                    update_values[state] = 0
                else:
                    possible_actions = self.mdp.getPossibleActions(state)

                    for action in possible_actions:
                        q_vals.append(self.getQValue(state, action))

                    update_values[state] = max(q_vals)

            self.values = update_values


    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        Q*(s, a) = sum_s' of T(s, a, s') [R(s, a, s') + gamma V*(s')]
        """
        "*** YOUR CODE HERE ***"
        possible_states = self.mdp.getTransitionStatesAndProbs(state, action)
        qValues = []

        for S, T in possible_states:
            reward = self.mdp.getReward(state, action, S)
            result = (T * (reward + (self.discount * self.getValue(S))))
            qValues.append(result)
        
        return sum(qValues)


    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possible_actions = self.mdp.getPossibleActions(state)
        action_vals = []
        best_action = 0

        # check for no legal actions
        if len(possible_actions) == 0:
            return None

        # get the q-value for every possible action
        for action in possible_actions:
            qval = self.getQValue(state, action)
            action_vals.append((action, qval))

        best_action = max(action_vals, key=lambda x: x[1])[0]
        return best_action

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

    def __init__(self, mdp, discount=0.9, iterations=1000):
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
        self.runValueIteration()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        # initalize all state vals to 0
        for state in states:
            self.values[state] = 0

        # repeat per iteration
        for i in range(self.iterations):
            state_len = len(states)
            state_index = i % state_len

            curr_state = states[state_index]

            # calculate qvals until terminal
            if not self.mdp.isTerminal(curr_state):
                action = self.getAction(curr_state)
                qval = self.getQValue(curr_state, action)

                self.values[curr_state] = qval



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
        Your prioritized sweeping value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = {}

        # Initialize an empty priority queue.
        fringe = util.PriorityQueue()

        # Compute predecessors of all states.
        for s in states:
            predecessors[s] = self.getPredecessors(s)
            self.values[s] = 0

        for s in states:
            if not self.mdp.isTerminal(s):
                # Compute max q-value for state 's'
                max_qval = self.getMaxQval(s)
                diff = abs(self.values[s] - max_qval)
                fringe.update(s, -diff)
            
        for i in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if fringe.isEmpty():
                return

            curr_state = fringe.pop()
            if not self.mdp.isTerminal(curr_state):
                self.values[curr_state] = self.getMaxQval(curr_state)

            for p in predecessors[curr_state]:
                diff = abs(self.values[p] - self.getMaxQval(p))

                if diff > self.theta:
                    fringe.update(p, -diff)
            

            


    def getMaxQval(self, state):
        return max([self.getQValue(state, a) for a in self.mdp.getPossibleActions(state)])
        

    def getPredecessors(self, state):
        # get all states
        states = self.mdp.getStates()
        predecessors = set()
        actions = ['north', 'south', 'east', 'west']

        # check we're not trying to get the predecessors of a terminal state
        if not self.mdp.isTerminal(state):

            # compute predecessors for all states
            for curr_state in states:

                possible_moves = self.mdp.getPossibleActions(curr_state)

                # making sure the current state is not terminal
                if not self.mdp.isTerminal(curr_state):

                    # need to check every possible move on this state to see if they lead to 'state'
                    for move in actions:
                        if move in possible_moves:
                            # get possible moves and check if resulting states have probability > 0 of reaching 'state'
                            possibilites = self.mdp.getTransitionStatesAndProbs(curr_state, move)

                            for s, p in possibilites:
                                if (s == state) and (p > 0):
                                    predecessors.add(curr_state)            

        return predecessors