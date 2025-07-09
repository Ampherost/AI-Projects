# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to University of California, Riverside and the authors.
# 
# Authors: Pei Xu (peixu@stanford.edu) and Ioannis Karamouzas (ioannis@cs.ucr.edu)
"""
The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In q_learning, once a terminal state is reached, 
the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
You can also only update the visualization of the v values by
    logger.log(i, v)
"""

# Todo: Finish implementing the 3 functions


# use random library if needed
import random


def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the infinity norm between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 (theta parameter in the pseudocode) is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    # Initialize value function and policy
    v = [0.0] * NUM_STATES
    pi = [0] * NUM_STATES

    # Log initial values and policy
    logger.log(0, v, pi)

    # Value iteration loop
    for iteration in range(1, max_iterations + 1):
        delta = 0.0
        v_old = v.copy()

        for s in range(NUM_STATES):
            # Compute Q(s, a) for all actions
            Q_sa = []
            for a in range(NUM_ACTIONS):
                q_value = 0.0
                for prob, next_state, reward, done in TRANSITION_MODEL[s][a]:
                    q_value += prob * (reward + gamma * v_old[next_state])
                Q_sa.append(q_value)

            # Update value and policy
            best_value = max(Q_sa)
            best_action = Q_sa.index(best_value)
            v[s] = best_value
            pi[s] = best_action

            # Track maximum change for convergence
            delta = max(delta, abs(v[s] - v_old[s]))

        # Log after this iteration
        logger.log(iteration, v, pi)

        # Check for convergence
        if delta < 1e-4:
            break

    return pi


def policy_iteration(env, gamma, max_iterations, logger):
    """
    Optional: Implement policy iteration to return a deterministic policy for all states.
    See lines 20-30 for details.  

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the infinity norm 
        between the values before and after an iteration is small enough. 
        In the gridworld environments, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of values by simply calling logger.log(i, v, pi).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model
    
    # Initialize value function and random policy
    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS - 1) for _ in range(NUM_STATES)]

    # Log initial values and policy
    logger.log(0, v, pi)

    for iteration in range(1, max_iterations + 1):
        # Policy Evaluation
        while True:
            delta = 0.0
            v_old = v.copy()
            for s in range(NUM_STATES):
                action = pi[s]
                new_value = 0.0
                for prob, next_state, reward, done in TRANSITION_MODEL[s][action]:
                    new_value += prob * (reward + gamma * v_old[next_state])
                delta = max(delta, abs(new_value - v[s]))
                v[s] = new_value

            # Log evaluation step (values with current policy)
            logger.log(iteration, v, pi)
            if delta < 1e-4:
                break

        # Policy Improvement
        policy_stable = True
        for s in range(NUM_STATES):
            old_action = pi[s]
            # Compute Q(s, a) for all actions
            Q_sa = []
            for a in range(NUM_ACTIONS):
                q_value = 0.0
                for prob, next_state, reward, done in TRANSITION_MODEL[s][a]:
                    q_value += prob * (reward + gamma * v[next_state])
                Q_sa.append(q_value)
            best_action = Q_sa.index(max(Q_sa))
            pi[s] = best_action
            if old_action != best_action:
                policy_stable = False

        # Log after policy improvement
        logger.log(iteration, v, pi)

        if policy_stable:
            break

    return pi


def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states using sample updates.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n

    # Initialize Q-table, value estimate, and policy
    Q = [[0.0] * NUM_ACTIONS for _ in range(NUM_STATES)]
    v = [0.0] * NUM_STATES
    pi = [0] * NUM_STATES
    logger.log(0, v, pi)

    # Hyperparameters
    eps = 1.0            # exploration rate
    alpha = 0.1          # learning rate
    state = env.reset()

    for t in range(1, max_iterations + 1):
        # Epsilon-greedy action selection
        if random.random() < eps:
            action = random.randint(0, NUM_ACTIONS - 1)
        else:
            action = max(range(NUM_ACTIONS), key=lambda a: Q[state][a])

        # Take step in environment
        next_state, reward, done, _ = env.step(action)

        # Q-update
        best_next = 0.0 if done else max(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

        # Update value and policy estimates
        for s in range(NUM_STATES):
            v[s] = max(Q[s])
            pi[s] = max(range(NUM_ACTIONS), key=lambda a: Q[s][a])

        logger.log(t, v, pi)

        # Prepare for next iteration
        state = env.reset() if done else next_state
        eps = max(0.1, 1.0 - float(t) / max_iterations)

    return pi



if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "Q-Learning": q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            ["_", "_", "_", "_", "_"],
            ["s", "_", "_", "_", 1],
            [-100, -100, -100, -100, -100],
        ],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()