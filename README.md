# Code to Calculate Divergent Semantic Integration (DSI)
This repository contains code from Johnson et al.'s (2022) paper **Divergent semantic integration (DSI): Extracting creativity from narratives with distributional semantic modeling**.

## Github Repo Navigation
The following is the **top-level directory layout** of this repo:

    .
    ├── DSI.py                 # Main python script to calculate DSI score
    ├── environment.yml        # Configure the environment and installing required packages
    ├── pyDSI tutorial         # A detailed tutorial on how to calculate DSI score
    ├── study1_OSF FINAL.csv   # The original result file from the authors' paper (can also be used to test the code in the first place)
    ├── README.md

## Environment Activation and Code Running
After installing miniconda, the first step is to create the virtual environment: 
```bash
conda env create -f environment.yml
```

The next step is to activate the created environment:
```bash
conda activate dsi
```

The next step is to run `DSI.py` file (using the Python interpreter located at the virtual environment):
```bash
/opt/miniconda3/envs/dsi/bin/python DSI.py 
```

## References
```
@article{johnson2023divergent,
  title={Divergent semantic integration (DSI): Extracting creativity from narratives with distributional semantic modeling},
  author={Johnson, Dan R and Kaufman, James C and Baker, Brendan S and Patterson, John D and Barbot, Baptiste and Green, Adam E and van Hell, Janet and Kennedy, Evan and Sullivan, Grace F and Taylor, Christa L and others},
  journal={Behavior Research Methods},
  volume={55},
  number={7},
  pages={3726--3759},
  year={2023},
  publisher={Springer}
}
```
