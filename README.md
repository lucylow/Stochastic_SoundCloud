# Stochastic SoundCloud : Lucy’s New Mozart Mixtape 🔥

Machine Learning Generative Music using RNN LSTMs.

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
* Calculations for **stocastic music** take a long time when done by hand. Stochastic SoundCloud uses machine learning to make it easier to generate melodies while reviewing basic math concepts like law of large numbers, probability theory, game theory, boolean algebra, markov chains, poisson law, and group theory
* ***“If I were not a physicist, I would probably be a musician. I often think in music. I live my daydreams in music. I see my life in terms of music.”* – Albert Einstein**
* In 1958 Iannis Xenakis used **Markov Chains, a stochastic process to make predictions on the future based on its present state**. He composed "Analogique" (pictured below) - The first musical composition that **models the probability of a note occuring after a sequence of notes**
  ![](https://github.com/lucylow/Stochastic_SoundCloud/blob/master/images/Iannis%20Xenakis%20Markov%20Chains.png)

---

## Stocastic Process 
* Probability theory where a math object is defined with random variables
* **"Stochastic"** == "an asymptotic evolution towards a stable state, towards a kind of goal, of stochos"

  ![](https://github.com/lucylow/Stochastic_SoundCloud/blob/master/images/Wiener_process_animated.gif)

* Examples:
  * **Bernoulli process** to study the repeatedly flipping of a coin where the probability of obtaining a head is p value is one and value of a tail is zero
  * **Weiner Brownian motion** process to study the diffusion of tiny particles suspended in fluid (also used as a solution to the Schrödinger equation)
  * **Poisson process** to study the number of phone calls occurring in a certain period of time

---

## Stochastic SoundCloud : Music is Sequential 

* Music is older than language - automatic music became "algorithmic" where piano compositions can be broken down into fragments
* The determined musical state is only partially determined by the preceding musical state where the concrete musical state n+2 follows after the state n+1 only with some probability
* Changing the pitch, duration, timbre, dynamics, and amplitudes of music waveforms using parameters changing the effects to the spectral domain


---
## Theory Classical Music Concepts  

In order to generate classical music for Lucy’s New Mozart Mixtape, we need to understand more about how a computer interprets music notes. Reading sheet music is like learning a new language where the symbols represent  pitch, speed, and rhythm of the melody. It is a sequential sucession of musical notes read in linear order. **How would you abstract a musical melody into numerical data that can be trained with a neural network?**

**Piano scales**

A scale is made of eight consecutive notes. The C major scale is composed of C, D, E, F, G, A, B, C. This is imporatant because we can transpose all of the musical pieces to key C, while maintaining the relative relationship between notes. The generated pieces can be transposed to any key. 

![](https://github.com/lucylow/Stochastic_SoundCloud/blob/master/images/Screen%20Shot%202020-09-07%20at%202.47.54%20AM.png)

**Piano Roll Representation**
* Binary-valued scoresheet-like matrix representing music notes over different time steps
* M-track piano roll representation: 
  * one bar is represented as a tensor x ∈ {0, 1} R×S×M where R == time steps in a bar and S == the number of note candidates 
  * T bars is represented as x_hat = {−x_hat(t)} from t =1 to t = T 
* Piano-roll of each bar, each track, for the real and the generated data is represented as a fixed-size matrix

  ![](https://github.com/lucylow/Stochastic_SoundCloud/blob/master/images/piano%20rolls.png)
  *The green bars next to the piano represents the piano roll of the score sheet*

----


## Theory Musical Instrument Digital Interface (MIDI) Representation  

Musical Instrument Digital Interface (MIDI) maps musical note names to numbers making it easier for engineers to play, edit and record music. An example would be C4 key on piano == "60" MIDI. The data is then fed into the neural network as piano roll representation where
  * X axis = Time sequence
  * Y axis = Notes on a piano keyboard 
 
  ![](https://github.com/lucylow/Stochastic_SoundCloud/blob/master/images/Screen%20Shot%202020-09-07%20at%202.50.47%20AM.png)
  *Example of C scale with ten notes C4, D4, E4, E4, F4, D4, G4, E4, D4, and C4 with corresponding MIDI numbers 60, 62, 64, 64, 65, 62, 67, 64, 62, and 60.*

**One Hot Encoding of MIDI numbers**

How do the MIDI numbers fit as an input in our RNN-LSTM neural network? One Hot Encoding.One Hot Vectors are a categorical binary representation where each row has one feature with a value of 1 (music note is on) and the other features with value 0 (music note of off).

Example:
  * MIDI file #1 [Note 1, Note 2, Note3] ==> {[1,0,0], [0,1,0], [0,0,1] } One Hot Encoding
  * MIDI file #2 [Note 1, Note 2, Note3] ==> {[1,0,0], [0,1,0], [0,0,1]} One Hot Encoding
  
Each song is an ordered list of pseudo-notes where the final vector will have dimensions where the Number of samples (nb) x Length of sequence (timesteps) x One-Hot Encoding of pseudo-notes. The melody at each timestep gets transformed into a 38-dimensional one-hot vector. There are 38 total kinds of events with 36 note-on events, 1 note-off event, and 1 no-event.

  ![](https://github.com/lucylow/Stochastic_SoundCloud/blob/master/images/Screen%20Shot%202020-09-06%20at%2011.06.13%20PM.png)

  *Matrix of one-hot encoded MIDI data for first four piano bars. The default setting is 96 beats per beat but we set it to 4 ticks/ beat or resolution of 1/16th note per time step. Each row represents the quantization of the time dimension.* 




---

## Theory Hierarchical RNN LSTM Architecture 

**Why choose a RNN LSTM for Stochastic SoundCloud music generation?**
* Music is an art of time with a temporal structure 
* Music has hierarchical structure with higher-level building blocks (phrases) made up of smaller recurrent patterns (bars)
* Recurrent Neural Networks (RNNs) are able to capture time dependencies between inputs.
* Mozer Eck in 2002 found that for RNN composed music composed, the “local contours
made sense butthe pieces were not musically coherent” and suggested to use long short-term memory Recurrent Neural Networks (RNN LSTM) for music instead of just RNN 
* RNN LSTM designed to avoid the "rapid decay of backpropagated error",

![](https://github.com/lucylow/Stochastic_SoundCloud/blob/master/images/Screen%20Shot%202020-09-07%20at%202.47.15%20AM.png)


**Stochastic SoundCloud Machine Learning and Architecture**

Train the RNN LSTM Recurrent Neural Network to compose a melody. Lookback and Attention RNNs are proposed to tackle the problem of creating melody’s long-term structure. It needs to be fed with a chord sequence and will then output a Prediction Matrix, which can be transformed into a piano roll matrix and finally into a melody MIDI file.

The number of samples is given by the difference between the number of timesteps of the piano roll matrix and the sequence length: number of samples = number of timestepspiano roll − sequence length.
* 3-dimensional Input Matrix of size (number of samples, timesteps, input dimension)
* 2-dimensional Target Matrix of size (number of samples, output dimension)


Input to Target Matrix
* Network Input Matrix: One network input sample consists of a 2-dimensional
input matrix
* Prediction Target Matrix: One target sample consists of a 1-dimensional target vector

![](https://github.com/lucylow/Stochastic_SoundCloud/blob/master/images/Screen%20Shot%202020-09-06%20at%2011.13.14%20PM.png)
*The RNN LSTM consists of an input layer, an output layer and optionally hidden layers between the input and output layer. The chord sequences need to be within one octave and the belonging melodies within two octaves*

---

## Technical Music Dataset
* **Use pre-trained data:**
  * Mozart's Modern Classical composition Download and unzip classical music files from : https://github.com/lucylow/Stochastic_SoundCloud/blob/master/data/classical%20music%20dataset.zip
  * There are three possible .mag bundle files at https://github.com/lucylow/Stochastic_SoundCloud/tree/master/config%20files
    * Basic_rnn
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
  * MIDI libraries for Python
* [Magenta for Tensorflow](https://magenta.tensorflow.org/) with the 3 pre-trained RNN LSTM models:
  1) Basic RNN (basic one hot encoding)
  2) Lookback RNN
  3) Attention RNN (looks at bunch of previous steps)
* GarageBand for Mac
  
---
## Technical Installation 
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
## Technical RNN LSTM Machine Learning Model parameters
Stochastic SoundCloud RNN LSTM networks used in experiments had two hidden layers and each hidden layer had 256 hidden neurons with initial learning rate of 0.001. The minibatch size was 64 and to avoid over-fitting the dropout rate was set to a ratio of 0.5:

* config = (choose options between basic_rnn, mono_rnn, lookback_rnn, attention_rnn)
* bundle_file = (choose .mag file) 
* output_dir = output directory within Stochastic_Soundloud folder 
* num_outputs = 10 (number of music files you want to output)
* num_steps = 128 
* primer_melody = 60 (middle C on piano)


---
## Lucy’s New Mozart Mixtape Results 

**Basic RNN**
  * Note by note basis (monotonic)
  * One-hot encoding == melody
  * Pitch range [48 to 84]
  * Basic two-three notes
  
    ![](https://github.com/lucylow/Stochastic_SoundCloud/blob/master/images/Screen%20Shot%202020-09-06%20at%209.05.48%20PM.png)
  
**Lookback RNN**
  * Patterns that occur over one or two measures/bars in a song resulting in more "repetitive" beats 
  * Less basic than 1) and more musical structure with actual melodies since Lookback feature that makes the model repeat sequences easier
  * Generating Long-term Structure in Songs and Stories paper
  * Allows custom inputs and labels
  * Lookback RNN outperformed the Attention RNN 
  
**Attention RNN** 
  * Looks at bunch of previous steps to figure out what note to play next (more longer term dependencies)
  * Generating Long-term Structure in Songs and Stories paper
  * More mathematically complicated 
  * Notes more complex (polytonic)
  
    ![](https://github.com/lucylow/Stochastic_SoundCloud/blob/master/images/Screen%20Shot%202020-09-07%20at%202.54.12%20AM.png)
  
---
## What's next for Lucy’s New Mozart Mixtape?
* Preprocess the MIDI files - dataset was quite noisy (remove super high/low notes by discarding notes below C1 or above C8, decrease the ratio of empty bars)
* Apply the Music Turing Test to compare outputs to human generated music. Can the discriminator tell if the generated music is real or fake?
* Compare quantitative results of the three models usng binarization testing stratgies: Bernoulli sampling (BS) or hard thresholding (HT)

---
## Conclusion 

Stochastic SoundCloud presenting Lucy’s New Mozart Mixtape uses three novel reural neural networks (RNNs) used to generate symbolite melodies. This music generated using machine learning techniques using Magenta from Google's Tensorflow AI. Using a LSTM long-short-term-memory model, with three specific RNN examples: Basic RNN, Lookback RNN, and Attention RNN. Outputs ~10 randomly generated output.mid music files that can be opened up on Mac's Garageband.

---

## References
* Google Tensorflow Magenta: Melody RNN https://github.com/magenta/magenta / https://magenta.tensorflow.org/
* Generating Long-Term Structure in Songs and Stories https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn/
* Convolutional Generative Adversarial Networks with Binary Neurons for Polyphonic Music Generation
https://arxiv.org/abs/1804.09399
* Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment https://arxiv.org/abs/1709.06298
* Demonstration of GAN based model for generating multi-track piano rolls https://salu133445.github.io/musegan/pdf/musegan-ismir2017-lbd-paper.pdf
* Valerio Velardo : Collection of videos on LSTM RNN for melody generation https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz
* A Hierarchical Recurrent Neural Network for Symbolic Melody Generation https://arxiv.org/pdf/1712.05274.pdf
*  Konstantin Lackner. Bachelor’s thesis “Composing a melody with long-short term memory (LSTM) Recurrent Neural Networks”https://konstilackner.github.io/LSTM-RNN-Melody-Composer-Website/Thesis_final01.pdf


