## Panagou

### Dependencies 

* branch from feature_continuous_action 
* probably need to 'make' for cpp code  

### Run Isolated

```
python panagou.py
```
* from `~/code/benchmark/` 
* should produce a figure where yellow dots are intersections, black are nominal trajectories to goal, orange/blue transparent dots are reachable set, etc. and orange/blue trajectories are chosen path 


### Run with MCTS 

```
python exp3.py
```
* from `~/code/` 
* I would turn off initial velocities, i.e. change line 285 of param.py from `r  = sqrt(random.random())*speed_lim` to `r  = 0*sqrt(random.random())*speed_lim`

#### Error Messages: 

`FileNotFoundError: [Errno 2] No such file or directory: '../current/models/a0.pt'`

* fix: run mice.py from `~/code/` and cancel once it starts making expert demonstration data 


### Notes

* Problem: Panagou has two problems: (i) velocity limits are not implemented, and (ii) numerical solns are not robust 
* Goal: (i) fix problems, (ii) run exp3.py with various tree sizes and identify boundary of where MCTS beats panagou for meaningful plot and (iii) identify cases where MCTS can 'juke' Panagou because of constant acceleration assumption 
* Paper: 'http://www-personal.umich.edu/~dpanagou/assets/documents/MCoon_CDC17.pdf'




