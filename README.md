# Generative Architectural Design with RL-PLAN: A Reinforcement Learning Based and Knowledge-informed Method for Floor Layout Design

* This repo 는 "Generative Architectural Design with RL-PLAN: A Reinforcement Learning Based and Knowledge-informed Method for Floor Layout Design" 의 RLPLAN 의 코드입니다.
* RLPLAN 은 Deep Reinforcement Learning (DRL) 을 사용해서 apartment layout 을 하는 알고리즘입니다.
* target metrics 로는 Space Syntax (Hillier & Hanson, 1989) 을 기반으로 한 VGA (Turner et al., 2001) 를 사용했습니다.
* ![image](https://github.com/user-attachments/assets/f0d274d7-f320-406f-89fa-59912c1d76d7)


해당 알고리즘은 room sequence 가 고정되어 있으며 이는 다음과 같습니다.
* (Living room) - (Master bedroom) - (Master bathroom) - (Second bedroom) - (Bathroom)
* 그리고 Room connection 은 다음과 같아요.
| 0 | livingroom | Master bedroom | Master bathroom | Second bedroom | Bathroom |
| livingroom | - | 1 | 0 | 1 | 1 |
| Master bedroom | 1 | - | 1 | 0 | 0 |
| Second bedroom | 1 | 0 | - | 0 | 0 |
| Bathroom | 1 | 0 | 0 | 0 | - |

Grid environment 에서 학습되고, 학습기간 동안 데이터는 두 형식으로 저장이 됩니다 for reward calculating, (1) grid pixel info (middle) - adjacency, (2) coordinates (right) - iou.
![image](https://github.com/user-attachments/assets/156e9f1e-6504-4b1a-b212-b5e4ca041584)

Reward 는 3 가지입니다. (1) 설정된 program requirements 에 따른 adjacency, 그리고 겹치기 않게 하기 위해 (2) intersection over union (iou), 그리고 (3) Integration [HH] value per room type (more private for bedroom & bathroom; while public livingroom).
(1), (2) 는 dense reward (when rooms are placed), 그리고 integration 은 the sparse reward 입니다.
![image](https://github.com/user-attachments/assets/12d23d22-8653-4737-833d-aee3d997e214)

## Sample
이런식으로 나왔습니다.
![image](https://github.com/user-attachments/assets/84373741-a631-4d2e-91fd-6979ad301ab3)


# References:
* Hillier, B., Leaman, A., Stansall, P., & Bedford, M. (1976). Space syntax. *Environment and Planning B: Planning and design*, 3(2), 147-185.
* Hillier, B., & Hanson, J. (1989). *The social logic of space*. Cambridge university press.
* Turner, A., Doxa, M., O'sullivan, D., & Penn, A. (2001). From isovists to visibility graphs: a methodology for the analysis of architectural space. *Environment and Planning B: Planning and design*, 28(1), 103-121.
