# Artificial-Intelligence
 
This repository contains a series of small projects made with applications of Artificial Intelligence.
 
## Projects
 
### Project 1 — Search Algorithms
 
Implementations of classic graph search algorithms applied to pathfinding on a 2D grid. Given a start cell, a goal cell, and a set of obstacles, each algorithm finds a path through the grid while visualizing the search process via a tkinter interface. The project includes Depth-First Search (DFS) using a stack-based frontier, Breadth-First Search (BFS) using a queue-based frontier, Uniform-Cost Search (UCS) with a priority queue ordered by cumulative path cost, and A* Search using the Manhattan distance heuristic combined with path cost. The grid includes weighted regions where movement is more expensive, allowing UCS and A* to demonstrate cost-sensitive routing that DFS and BFS cannot account for.
 
### Project 2 — Markov Decision Processes and Reinforcement Learning
 
Solutions for sequential decision-making in stochastic environments. The project covers both model-based and model-free approaches applied to two domains: a grid world with terminal reward and penalty states, and a simulated robot crawler that learns to move forward by adjusting its arm joints. Value Iteration computes optimal state values by iteratively applying the Bellman optimality equation until convergence. Policy Iteration alternates between evaluating a fixed policy and greedily improving it, often converging in fewer iterations than value iteration. Q-Learning learns an action-value function from environment samples using an epsilon-greedy exploration strategy with a decaying exploration rate, requiring no knowledge of the transition model.
 
### Project 3 — Adversarial Search
 
Depth-limited adversarial search algorithms for playing Connect-4. A custom evaluation function scores board states by counting aligned disk segments and weighting them by length. Minimax explores the full game tree up to the depth limit, assuming optimal play from both sides. Alpha-Beta Pruning extends minimax by eliminating branches that cannot influence the final decision, significantly reducing the number of nodes evaluated. Expectimax replaces the minimizing player with a chance node that averages over all legal moves uniformly, modeling a non-optimal or randomized opponent.
 