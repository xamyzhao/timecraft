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
python make_timelapse.py cezanne_watermelon_and_pomegranates.jpg
```
If you get any interesting or fun results, please let me know on [twitter](https://twitter.com/AmyZhaoMIT)!

## Downloading the dataset
We have organized the dataset into [pickle](https://docs.python.org/3/library/pickle.html) files. 
Each file contains information from a single time lapse video, and has the keys:
* `vid_name`: A shortened name in the rough format \<video owner>-\<video name>  
* `vid_id`: The YouTube or Vimeo video ID. Might be `None` if the video no longer exists. 
* `framenums`: The frame numbers selected from the video. These should be frames that are relatively clean (without 
hands and shadows in the watercolors dataset, for example). However, there might be translations between some of the frames, 
which we store in a separate annotation file `frame_shifts.txt`.

For watercolors, we also provide the keys:
* `frames`: The actual frames extracted from the video, in a `n_frames x h x w x 3` `np.ndarray`. These frames might
not correspond exactly to the `framenums` field above, since they were extracted using Matlab and there could be some 
rounding inconsistencies.

For digital paintings, we provide the keys: 
* `crop_start_xy`, `crop_end_xy`: The coordinates of the bounding box of the painting. Since the digital paintings dataset
is too large to host online, we provide the information and script necessary if you wish to download and preprocess
the dataset on your own.


If you wish to use our preprocessed watercolors dataset, please email me.

We provide information about the digital paintings dataset in this repository, under `./dataset`.
Extract the `.pkl` files. Then, to download the actual video frames, run the preprocessing script `src/dataset/scripts/preprocess_digital_painting_vids.ipynb`.

## Training your own models
The model is trained in two stages, as described in Sections 4.2.1 and 4.2.2 in the paper.

First, to run *pairwise optimization*, train a `TimeLapseFramesPredictor` model using:
```
python main.py tlf --dataset watercolor-example dig-example
```

Once this model has been trained, run *sequence optimization*. Make sure
to pass in the directory of your pre-trained model.  
```
python main.py tls --dataset watercolor-example dig-example -pd <your_trained_model_dir> -pe <epoch_to_load>
```

This repository has been cleaned up from the code we used for the paper; however, it likely still contains some legacy functionality. 
If you run into any problems, please open an issue on Github.  


<sub>Repo name inspired by Magic: The Gathering.</sub>

![Timecrafting](https://gatherer.wizards.com/Handlers/Image.ashx?multiverseid=129012&type=card)
