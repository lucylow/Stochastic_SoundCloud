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
* Calculations for stocastic music take a long time when done by hand - let's use machine learning and HPC to make it easier to generate original "Mozart" melodies using stochastic processes that will dominate the TOP MUSIC number-one hits 
* Review concepts like law of large numbers, probability theory, game theory, boolean algebra, markov chains, poisson law, group theory and etc 
* Music is older than language - automatic music became "algorithmic" where piano compositions can be broken down into fragments
* Iannis Xenakis and Stocastic Music : Modelling the probability of a note occuring after a sequence of notes since all music is sequential
* *“If I were not a physicist, I would probably be a musician. I often think in music. I live my daydreams in music. I see my life in terms of music.”* – Albert Einstein"
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
* Apply stocashtic, chaotic, or determinatisic curves to different music composizitions and parameters fo create a sound transformation
* Changing the pitch, duration, timbre, dynamics, and amplitudes of music waveforms using parameters changing the effects to the spectral domain

---

## Music Dataset
* **Use pre-trained data:**
  * Mozart's Modern Classical composition Download and unzip classical music files from : https://github.com/lucylow/Stochastic_SoundCloud/blob/master/data/classical%20music%20dataset.zip
  * There are four possible .mag bundle files at https://github.com/lucylow/Stochastic_SoundCloud/tree/master/config%20files
    * Basic_rnn
    * Mono_rnn
    * Lookback_rnn
    * Attention_rnn
    
* **Build your own dataset:**
  * Choose a Musical Instrument Digital Interface (MIDI) dataset from the list below
    * https://composing.ai/dataset
    * http://www.piano-midi.de/
    * https://magenta.tensorflow.org/datasets/nsynth
    * https://magenta.tensorflow.org/datasets/maestro
    * https://magenta.tensorflow.org/datasets/groove
    * http://abc.sourceforge.net/NMD/ 
    * http://musedata.stanford.edu/
    * https://github.com/mdeff/fma
    * http://www.piano-midi.de/albeniz.htm
    * http://www.piano-midi.de/bach.htm
    * http://www.piano-midi.de/beeth.htm
    * http://www.piano-midi.de/mozart.htm
     
 * Convert MIDI into NoteSequence datatype for the Stochastic SoundCloud:

   > INPUT_DIRECTORY=<folder_name_here>
   > SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord
   > convert_dir_to_note_sequences \
   >   --input_dir=$INPUT_DIRECTORY \
   >   --output_file=$SEQUENCES_TFRECORD \
   >   --recursive

* NoteSequences are outputted to /tmp/notesequences.tfrecord
   
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
## LSTM Machine Learning Model parameters
* config = (choose options between basic_rnn, mono_rnn, lookback_rnn, attention_rnn)
* bundle_file = (choose .mag file) 
* output_dir = output directory within Stochastic_Soundloud folder 
* num_outputs = 10 (number of music files you want to output)
* num_steps = 128 
* primer_melody = 60 (middle C on piano)

---
## Results 

1) Basic_rnn
  * Note by note basis 
  * one-hot encoding == melody
  * pitch range [48 to 84]
  * Basic two-three notes
  
2) Lookback_rnn
  * Patterns that occur over one or two measures/bars in a song 
  * Generating Long-term Strucutre in Songs and Stories
  * Less basic than 1) nd more musical structure with actual melodies 
  * Allows custom inputs and labels
  * Results in more "repetitive" beats 
  
3) Attention_rnn
  * Looks at bunch of previous steps to figure out what note to play next (more longer term dependencies)
  * More mathematically complicated 
  * Notes more complex
  
---
## What's next 
* Apply the Music Turing Test to compare outputs to human generated music

---
## Conclusion 

Neural networks used to generate symbolite melodies. This music generated using machine learning techniques using Magenta from Google's Tensorflow AI. Using a LSTM long-short-term-memory model, with three specific RNN examples: Basic RNN, Lookback RNN, and Attention RNN. Outputs ~10 randomly generated output.mid music files that can be opened up on Mac's Garageband.

---

## References
* Google Tensorflow Magenta: Melody RNN https://github.com/magenta/magenta / https://magenta.tensorflow.org/
* Generating Long-Term Structure in Songs and Stories https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn/
* The Attention Method we use here: Neural Machine Translation by Jointly Learning to Align and Translate https://arxiv.org/abs/1409.0473
* Convolutional Generative Adversarial Networks with Binary Neurons for Polyphonic Music Generation
https://arxiv.org/abs/1804.09399
* Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment https://arxiv.org/abs/1709.06298
* Demonstration of GAN based model for generating multi-track piano rolls https://salu133445.github.io/musegan/pdf/musegan-ismir2017-lbd-paper.pdf
* Composing Music With Recurrent Neural Networks http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/
* More on note sequences for dataset building https://developers.google.com/protocol-buffers/
* Carnegie Mellon University https://www.link.cs.cmu.edu/melody-generator/
* https://random-music-generators.herokuapp.com/melody
* The one and only Siraj Raval. AI for Music Composition: https://www.youtube.com/watch?v=NS2eqVsnJKo&ab_channel=SirajRaval
* A Hierarchical Recurrent Neural Network for Symbolic Melody Generation https://arxiv.org/pdf/1712.05274.pdf


