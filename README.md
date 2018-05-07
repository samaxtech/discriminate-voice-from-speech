# Discriminating Voice from Speech

Implementation of a method for discriminating speech from music.

The algorithm is trained/learns from a given music data set that determine which thresholds are established to perform the differentiation (by extracting features from training data), so that it outputs 0 for music or 1 for speech.

Note: Output is a cell array (named 'music_speech') that contains the name of each file in the first column and its corresponding output (0 for music or 1 for speech) in the second.

INSTRUCTIONS:
------------------------------------------------------------------------------------------------------------------------------
***If algorithm training, feature extraction, accuracy evaluation and input audio file classification wants to be performed:***

1) Specify the path where input audio files to be tested are located, in "Read, process, and classify input audio files" section (**path is './test_files' by default**, a folder provided that contains some test audio files).
2) Locate 'audio' folder in the same directory as the main script ('music_vs_speech.m'). It contains 39 music and audio files where features are going to be extracted from.
3) Run the code.
------------------------------------------------------------------------------------------------------------------------------
***If only audio file classification wants to be performed:***

1) Specify the path where input audio files to be tested are located, in "Read, process, and classify input audio files" section (**path is './test_files' by default**, a folder provided that contains some test audio files).
2) Locate Matlab .mat files 'evaluation_var.mat' and 'plot_var.mat' in the same directory as the main script ('music_vs_speech.m'). They contain already extracted thresholds and variables to perform the classification and plot results.
3) **UNCOMMENT** 'Load Features' section at the beginning of the main script ('music_vs_speech.m').
4) **COMMENT** sections 'Training data','Extract Features from training data', and 'Evaluate accuracy of the algorithm, based on available data (files)'.
5) Run the code.
------------------------------------------------------------------------------------------------------------------------------
