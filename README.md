This repository contains the training script of our solution for winning 2nd place in the Arabicthon2022 in KSA.

## What is Arabicthon:

[Arabicthon](https://arabicthon.ksaa.gov.sa/) is a deep learning competition organized by [**The King Salman Global Academy for the Arabic Language**](https://ksaa.gov.sa/en/homepage/). It came with 3 tracks:

- The Arabic poetry challenge.
- The lexicon Challenge.
- The Arabic language games for kids.

## Our solution:

We choose to work on the Arabic poetry challenge. We build a web app that contains multiple tools that treat the Arabic poetry, such as:

- Poem generation based on the rhyme and prosody.
- Poem generation based on a picture.
- Verse completion given a rhyme.
- Meter classification without diacritics.
- Arabic poetry automatic diacritics.
- Aroud generation without the need of diacritics.
- And many other variants of these tools ...

This project has won us the 2nd place in the Arabicthon2022.

## Front:

React app that contains the frontend of the project. you can find it in here: [arabicthon_frontend](https://github.com/wadjih-bencheikh18/arabicthon-front)

## Backend:

Flask app that contains the backend of the project. you can find it in here: [arabicthon_backend](https://github.com/wadjih-bencheikh18/arabicthon-backend)

## Training:

### Poem generation:

The poem generated has been trained by finetuning [Aragpt2-medium](https://huggingface.co/aubmindlab/aragpt2-medium) on a dataset of ~1M Arabic poem verse. You can find the training file in here: [poem_generation_notebook.ipynb](https://github.com/TheSun00000/arabicthon_training/blob/main/training_final/poem_generation_notebook.ipynb "poem_generation_notebook.ipynb")

### Meter classification:

This classifier is based on 3 layers of Bi-directional LSTMs, trained on the same dataset in order to classify poem verses to 10 meters. You can find the training file in here: [meter-classification-lstm.ipynb](https://github.com/TheSun00000/arabicthon_training/blob/main/training_final/meter-classification-lstm.ipynb "meter-classification-lstm.ipynb")

### Aroud generation:

This is a seq2seq LSTM based model that takes a verse without diacritics, and outputs it's Aroud form. You can find the training file in here: [aroud-lstm.ipynb](<[https://github.com/TheSun00000/arabicthon_training/blob/main/training_final/aroud-lstm(1).ipynb](https://github.com/TheSun00000/arabicthon_training/blob/main/training_final/aroud-lstm(1).ipynb)> "aroud-lstm(1).ipynb")

### Image captioning:

This model is used in order to generate Arabic poetry based on an input image.
This model takes an image as input an output its captioning in English. You can find the training file in here:[image-captioning-with-attention.ipynb](https://github.com/TheSun00000/arabicthon_training/blob/main/training_final/image-captioning-with-attention.ipynb "image-captioning-with-attention.ipynb").

The caption is then translated into the Arabic language and then it gets fed to the GPT based generator as a first verse. The generator will then make sure to generate verses that are constrained by the rhyme and meter conditions while saving as much context as possible from the image caption.

### Arabic automatic diacritization:

This model is a seq2seq LSTM based model, that takes a verse as input, and predicts the most suitable diacritics. You can find the training file in here: [tachkil_notebook.ipynb](https://github.com/TheSun00000/arabicthon_training/blob/main/training_final/tachkil_notebook.ipynb "tachkil_notebook.ipynb").
