# Map Matching

<img src=./images/map_matching.png width="800" height="400" />

This repository contains an implementation of a map matching algorithm
using hidden markov models in python.


## Installation
First, install python 3.9.2 (osmnx needs >= python 3.9)
```
pyenv install 3.9.2 
pyenv local 3.9.2
```
Then, run
`poetry install`

## Data
GPS Trajectory Dataset of the Region of Hannover, Germany. 
Zourlidou, S., Golze, J. and Sester, M. (2022). 
Dataset: GPS Trajectory Dataset of the Region of Hannover, Germany.
 https://doi.org/10.25835/9bidqxvl
 
 ## Model 
check out my article 
 
## Notebook

before running the notebook : export the module `src` to `PYTHONPATH`
```
export PYTHONPATH=$PYTHONPATH:/path/to/folder/map_matching/src
```
Then,
```
poetry run streamlit run notebooks/notebook_map_matching.py
```

<img src=./images/notebook_example.png width="800" height="900" />
