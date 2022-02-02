# Motion planning 

## Modified Intelligent Bidirectional Rapidly-Exploring Random Tree 

**"Modified Intelligent Bidirectional Rapidly-Exploring Random Tree"** was chosen as the motion planner. This algorithm is a modification of the algorithms based on [rapidly exploring random tree](http://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf).
The modification of the algorithm consists in introducing advanced sampling heuristics when adding new vertices (nodes) to trees (graphs). The heuristic is based on determining the position of a potential new vertex in the space between the start and end points of planning. The algorithm was developed to work in conditions of limited computing resources. The modification uses significantly less memory during operation, since it does not add nodes to the trees that will not be used in the pathfinding process, because they are outside the space between the start and end points of the path. 

[Detailed description of the algorithm](https://ieeexplore.ieee.org/document/9666083)

The algorithm was implemented on the basis of the scheduler library [Open Motion Planning Library](https://ompl.kavrakilab.org/). Repository with modified algorithm (MIB-RRT) and installation instructions are available at [link](https://github.com/IDovgopolik/ompl). 

## Calibration of a motion planning algorithm

To speed up the path planning process, the system provides calibration. Her process consists of pre-testing the available schedulers for the kinematics of the robot and the environment. There is a determination of the fastest scheduler for the task, and its most optimal parameters that provide maximum efficiency. Pre-calibration can significantly speed up the process of the system. 

[Detailed description of the algorithm](https://ieeexplore.ieee.org/document/9666096)

[Official repository with calbration module]()