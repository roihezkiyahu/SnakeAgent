In this repo I implement several RL algorithms (DQN, DDQN, Dueling DDQN, A2C, PPO) in order to teach the agent to play snake.

Along the way, I decided to generalize my code so that it can handle any Atari game from Gymnasium, and also so that I can run multiple configurations with ease to achieve a perfect score.

The best-performing snake agent was the A2C with a mean score of 78 and a median score of 100!

Here are two of the best agent's games using A2C:

| Current AVG game A2C | Perfect score A2C |
|:---------------------:|:-----------------:|
| ![Current AVG game A2C (Score 78)](A2C_78.gif) | ![Perfect score A2C (Score 100)](A2C_100.gif) |

**Current Results Dueling DDQN**

![Current Results Dueling DDQN (Score 45)](Score_Dueling_DDQN.gif)
