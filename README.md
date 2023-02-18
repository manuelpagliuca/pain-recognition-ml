# Pain Recognition: dataset analysis and experimental validation
![Unity](https://img.shields.io/badge/build-passing-green)
![Unity](https://img.shields.io/badge/license-MIT-yellowgreen)
![Unity](https://img.shields.io/badge/language-Python-brightgreen)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](mailto:pagliuca.manuel@gmail.com) 
## About the project
This is an unified project for the courses Affective Computing and Natural Interaction at [PhuseLab](https://phuselab.di.unimi.it/), University of Milan, A.Y. 2021/2022.

The aim of this project is to test the accuracy of early and late fusion approaches on a multimodal dataset to classify the presence of pain in patients. Participants were subjected to an external heat-pain stimulus through a physical device.

Their facial expressions and biophysical signals were recorded through the use of cameras and the application of electrodes, then features were extracted. The descriptors came from two different modalities and will be combined by testing both fusion approaches. Finally, classifications and accuracy estimates were made, based on which it was possible to determine that early fusion is the most accurate approach for the dataset considered.
* For more information about the project download the [report](Pain_Detection_Manuel_Pagliuca_AC_NI_2022.pdf).

Spiegare il funzionamento generale del progetto, inserire il diagramma e spiegare i vettori.

### Video feature extraction

## Tools
* IntelliJ IDEA and Python for developing the project application.
* Microsoft Excel for working with the `.csv` files.
* [BioVid](https://ieeexplore.ieee.org/document/6617456) dataset.

## Dependencies
* OpenCV
* MediaPipe
* SkLearn
* 

## Computer vision techniques involved for feature extraction
Spiegare le tecniche di computer vision utilizzate nel progetto.

### Head pose estimation
spiegazione
Inserire la gif
### Land mark distances
spigezione
gif
### Gradient for face folds
spiegazione
gif

## Ideas for future extensions
- Calcolare il gradiente solo nel momento in cui lo stimolo del dolore viene attivato e non sull'intera finestra.
- ...
