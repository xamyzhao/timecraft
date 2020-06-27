## timecraft
A learning-based method for synthesizing time lapse videos of paintings. This work was presented at CVPR 2020. This repository contains the authors' implementation, as discussed in the paper. 

If you use our code, please cite:

**Painting Many Pasts: Synthesizing Time Lapse Videos of Paintings**  
[Amy Zhao](https://people.csail.mit.edu/xamyzhao), [Guha Balakrishnan](https://people.csail.mit.edu/balakg/), [Kathleen M. Lewis](https://katiemlewis.github.io/), [Fredo Durand](https://people.csail.mit.edu/fredo), [John Guttag](https://people.csail.mit.edu/guttag), [Adrian V. Dalca](adalca.mit.edu)  
CVPR 2020. [eprint arXiv:2001.01026](https://arxiv.org/abs/2001.01026)

# Getting started
## Prerequisites
To run this code, you will need:
* Python 3.6+ (Python 2.7 may work but has not been tested)
* CUDA 10.0+
* Tensorflow 1.13.1 and Keras 2.2.4
* 1-2 GPUs, each with 12 GB of memory

## Creating time lapse samples
Use the provided script to run our trained model and synthesize a time lapse for a given input image.
```
python scripts/make_timelapse.py cezanne_watermelon_and_pomegranates.jpg
```

## Training your own models
Code coming soon!

<sub>Repo name inspired by Magic: The Gathering.</sub>

![Timecrafting](https://gatherer.wizards.com/Handlers/Image.ashx?multiverseid=129012&type=card)
