# Saliency Driven Image Manipulatoin - CS413 Project

## Introduction

Our task was to reconstruct the method introduced by [Mechrez & al.](https://arxiv.org/pdf/1612.02184) allowing to modify the saliency of a given region of a picture.

The original code and saliency model of the paper are missing from their repository, so we needed to rewrite the whole codebase and find an adequate alternative for the saliency model.

## Python environnement
Conda was used to set-up the needed librairies and the environment can be added by using the *environment.yml* file

## File structure 
To help you trough your exploration, here is a little break-down of the directory orginization of the project.

```bash
├── data
│   ├── background_decluttering
│   ├── distractors_attenuation
│   ├── object_enhancement
│   └── saliency_shift
│   ├── coco_output
│   ├── debug
├── output
├── src
└── Tempsal
    ├── ...
```

### Data directories
We have two pincipal sources of data, the 2017 validation set of Microsoft COCO and the 667 benchmark images created for the paper. Image that we got from coco are placed alongside their masks in the coco_output and debug (for simple images) directories.

The data from the paper is spread amongst the 4 first directories and each one represents a different task. It includes results from their implementation and other state of the art saliency manipulation methods.

### Code directories
The code utilities that we wrote are all in the `src` folder and are called from `main.py`. We tried multiple saliency model and Tempsal required a complex codebase, so we added their [git repository](https://github.com/IVRL/Tempsal) directly in the `Tempsal/`directory. The weights of their model are too heavy for a git repository, but you can find them [here](https://drive.google.com/drive/folders/160WB1YrPAjNYy787jP1pmffl9Xv0gLw6) and put them in `Tempsal/src/checkpoints/`

## Execution
To run the code, please follow the following format:

> `python .\main.py {Path to your image} {Path to the mask} {saliency contrast}`

Example: python .\main.py .\data\debug\easy_apple_small.jpg .\data\debug\easy_apple_small_mask.jpg 0.1


## Benchmark
We compared our results with the available results from the paper's dataset and the comparison can be seen on the following website: [https://jeremyhugentobler.github.io/saliency-manipulation/](https://jeremyhugentobler.github.io/saliency-manipulation/)

If you modify the codebase and want to launch a script that generates an html page for comparison, you can use `python bench.py`


