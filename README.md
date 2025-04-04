# Generative Architectural Design with RL-PLAN: A Reinforcement Learning Based and Knowledge-informed Method for Floor Layout Design

This repository contains the code for RL-PLAN, developed as part of the paper "Generative Architectural Design with RL-PLAN: A Reinforcement Learning-Based and Knowledge-Informed Method for Floor Layout Design."
* RL-PLAN is an apartment layout generation algorithm based on Deep Reinforcement Learning (DRL).
* As the target metric, it uses Visibility Graph Analysis (VGA) from Space Syntax theory (Hillier & Hanson, 1989; Turner et al., 2001).

The room placement follows a fixed sequence:
* (Living room) - (Master bedroom) - (Master bathroom) - (Second bedroom) - (Bathroom)
The room connection matrix is defined as follows:
| 0 | livingroom | Master bedroom | Master bathroom | Second bedroom | Bathroom |
| livingroom | - | 1 | 0 | 1 | 1 |
| Master bedroom | 1 | - | 1 | 0 | 0 |
| Second bedroom | 1 | 0 | - | 0 | 0 |
| Bathroom | 1 | 0 | 0 | 0 | - |

## Training Environment
The agent is trained in a grid-based environment, and two types of data are stored during training for reward computation:
1. Grid pixel information (used for adjacency calculations)
2. Room coordinates (used for IoU calculations)
![image](https://github.com/user-attachments/assets/156e9f1e-6504-4b1a-b212-b5e4ca041584)

## Reward Structure
The agent receives three types of rewards:
1. Adjacency reward — based on whether program requirements (e.g., required room-to-room connections) are met
2. IoU penalty — to discourage overlapping rooms
3. Integration [HH] reward — derived from space syntax metrics to reflect spatial privacy and visibility, where bedrooms and bathrooms are rewarded for lower integration, and living rooms are rewarded for higher integration

* The adjacency and IoU rewards are dense rewards, given each time a room is placed.
* The integration reward is a sparse reward, provided after all rooms are placed.
![image](https://github.com/user-attachments/assets/12d23d22-8653-4737-833d-aee3d997e214)

## Sample Output
Here is an example of a generated layout:
![image](https://github.com/user-attachments/assets/84373741-a631-4d2e-91fd-6979ad301ab3)


# References:
* Hillier, B., Leaman, A., Stansall, P., & Bedford, M. (1976). Space syntax. *Environment and Planning B: Planning and design*, 3(2), 147-185.
* Hillier, B., & Hanson, J. (1989). *The social logic of space*. Cambridge university press.
* Turner, A., Doxa, M., O'sullivan, D., & Penn, A. (2001). From isovists to visibility graphs: a methodology for the analysis of architectural space. *Environment and Planning B: Planning and design*, 28(1), 103-121.
