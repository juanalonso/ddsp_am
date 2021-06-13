# DDSP-FM: differentiable FM synthesis

Master Thesis, May 2021

Author: Juan Alonso

Supervisor: Cumhur Erkut

[Sound and Music Computing](https://www.smc.aau.dk/) - Aalborg University, Copenhagen


### Please, visit [the audio examples page](https://juanalonso.github.io/ddsp_fm/) to listen to the results.

DDSP-FM is a fork of the [official DDSP library](https://github.com/magenta/ddsp). This version includes the following new features:
* a differentiable 4-op FM synthesizer
* a differentiable AM synthesizer
* a new operator, `mult`
* a new TFRecord, suitable for future develpments, such as preset matching
* a flag for stopping the training if losses are NaN

## 💡 Timbre matching demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juanalonso/ddsp_fm/blob/master/ddsp/colab/fm/Timbre_matching.ipynb) This notebook will create a random preset for the FM synth, generate 48 pitches and train a NN that will create a patch as similar as possible as the original one, using only the spectrograms of the original preset.