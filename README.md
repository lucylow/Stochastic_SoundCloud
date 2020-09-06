# Stochastic SoundCloud : Lucy’s New Mozart Mixtape 🔥

Machine Learning Generative Music

<div>
  
  [![Status](https://img.shields.io/badge/status-work--in--progress-success.svg)]()
  [![GitHub Issues](https://img.shields.io/github/issues/lucylow/Stochastic_SoundCloud.svg)](https://github.com/lucylow/Stochastic_SoundCloud/issues)
  [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/lucylow/Stochastic_SoundCloud.svg)](https://github.com/lucylow/Stochastic_SoundCloud/pulls)
  [![License](https://img.shields.io/bower/l/bootstrap)]()

</div>

![CATS ARE SO CUTE](https://github.com/lucylow/Stochastic_SoundCloud/blob/master/images/album_cover.png)
*Album Cover. LUCY's New_Mozart_Mixtape now available on Stochastic_SoundCloud for the LOW price of $3.99. Featuring new up coming rapper TensorFlow_AI*

--------

## Motivation 
* Calculations for stocastic music take a long time when done by hand - let's use machine learning and HPC to make it easier using concepts like law of large numbers, probability theory, game theory, boolean algebra, markov chains, poisson law, group theory and etc to redefine the concept of "harmony"
* To generates original "Mozart" melodies using stochastic processes that will dominate the TOP MUSIC number-one hits 
* Physicist Denis Gabor had the idea of quantum representation for sound - musical quantas of sampled waveforms: “If I were not a physicist, I would probably be a musician. I often think in music. I live my daydreams in music. I see my life in terms of music.” – Albert Einstein"
---
## Stocastic Process 
* Probability theory where a math object is defined with random variables 
* "Stochastic" == "an asymptotic evolution towards a stable state, towards a kind of goal, of stochos"
* Pragmatic Examples:
  * **Bernoulli process** to study the repeatedly flipping of a coin where the probability of obtaining a head is p value is one and value of a tail is zero
  * **Brownian motion** process to study the diffusion of tiny particles suspended in fluid (also used as a solution to the Schrödinger equation)
  * **Poisson process** to study the number of phone calls occurring in a certain period of time
---
## Stochastic SoundCloud
* The determined musical state is only partially determined by the preceding musical state where the concrete musical state n+2 follows after the state n+1 only with some probability
* Control pitch, duration, timbre, & dynamics
* Apply stocashtic, chaotic, or determinatisic curves to different music composizitions and parameters fo create a sound transformation
* Changing the pitch and amplitudes of music waveforms using parameters changing the effects to the spectral domain

---

## Music Dataset
* Use pre-trained data
  * Mozart's Modern Classical composition Download and unzip classical music files from : https://github.com/lucylow/Stochastic_SoundCloud/blob/master/data/classical%20music%20dataset.zip
  * /config where there are four possible .mag bundle files
    * basic_rnn
    * mono_rnn
    * lookback_rnn
    * attention_rnn
    
* Build your own dadaset 
  * Musical Instrument Digital Interface (MIDI) dataset for music analysis. 
  * Choose a dataset then convert MIDI files into NoteSequences:
    * https://composing.ai/dataset
    * http://www.piano-midi.de/
    * https://magenta.tensorflow.org/datasets/nsynth
    * https://magenta.tensorflow.org/datasets/maestro
    * https://magenta.tensorflow.org/datasets/groove
    * https://magenta.tensorflow.org/datasets/e-gmd
    * https://magenta.tensorflow.org/datasets/bach-doodle
    * http://abc.sourceforge.net/NMD/ 
    * http://musedata.stanford.edu/
    * https://github.com/mdeff/fma
    * http://www.piano-midi.de/albeniz.htm
    * http://www.piano-midi.de/bach.htm
    * http://www.piano-midi.de/beeth.htm
    * http://www.piano-midi.de/mozart.htm
  
    """
    INPUT_DIRECTORY=<folder containing MIDI and/or MusicXML files. can have child folders.>

    # Convert to NoteSequences
    SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord

    convert_dir_to_note_sequences \
      --input_dir=$INPUT_DIRECTORY \
      --output_file=$SEQUENCES_TFRECORD \
      --recursive
    """
   * NoteSequences were output to /tmp/notesequences.tfrecord
   
---
## Technical Tools
* Python 3 (>= 3.5)
* [Magenta for Tensorflow](https://magenta.tensorflow.org/) with the 3 pre-trained LSTM models:
  1) Basic RNN (basic one hot encoding)
  2) Lookback RNN
  3) Attention RNN (looks at bunch of previous steps)
* GarageBand for Mac
  
---
## Installation 
Use Anaconda python packages:

> curl https://raw.githubusercontent.com/tensorflow/magenta/master/magenta/tools/magenta-install.sh > /tmp/magenta-install.sh
> bash /tmp/magenta-install.sh

Run Magenta using Python programs or Juypter Notebook
> source activate magenta

Clone this repository
> git clone https://github.com/lucylow/Stochastic_SoundCloud.git

Install the dependencies
> pip install -e .

Run the **melody_rnn_generate script** from the base directory
> python Stochastic_SoundCloud/melody_rnn/melody_rnn_generate --config=...


---
## Machine Learning Model: LSTM RNN 



---
## Results 

1) Basic RNN 


2) Lookback RNN


3) Attention RNN

https://arxiv.org/abs/1409.0473

---

## References
* Iannis Xenakis. Formalized Music 
* Google Tensorflow Magenta: Melody RNN https://github.com/magenta/magenta / https://magenta.tensorflow.org/
* Generating Long-Term Structure in Songs and Stories https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn/
* The Attention Method we use here: Neural Machine Translation by Jointly Learning to Align and Translate https://arxiv.org/abs/1409.0473
* Composing Music With Recurrent Neural Networks http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/
* More on note sequences for dataset building https://developers.google.com/protocol-buffers/
* Carnegie Mellon University https://www.link.cs.cmu.edu/melody-generator/
* https://random-music-generators.herokuapp.com/melody

