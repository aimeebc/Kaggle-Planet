# Planet: Understanding the Amazon from Space
## Use satellite data to track the human footprint in the Amazon rainforest

This repository contains the code produced for my entry to the Kaggle
competition Planet: Understanding the Amazon from Space. This was a really interesting
competition to develop algorithms capable of attributing both weather and land uses
labels to high resolution satellite images to help identify and understand deforestation
in the Amazon basin. Further details of the competition
can be found [at Kaggle.](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)

I joined kaggle and this competition 8 days before the deadline and chose my best
results as my final submission. This was a fine tuned version of the Inception v3
model with the weights loaded from Keras with tensorflow as the backend, using the
jpgs as input in their full size, 256 by 256, with 3 colour channels. I performed 10-fold cross-validation to evaluate my model and selected the highest performing fold as my submission.

I got started using the kernel recommended by the competition to [explore the data.](https://www.kaggle.com/robinkraft/getting-started-with-the-data-now-with-docs). I ran
this locally and used this code as a basis for loading the data. First I built a simple
convolutional neural network in Keras and trained it on the jpgs. I was
unable to load the full sample into memory, so first I tested with a subset of the data, then I wrote a generator to load the data in batches (utilities/LoadData.py).

While trying to select the optimum threshold value for class assignment based on the
predicted probabilities I found a solution by [anokas on the kaggle forums](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475) for optimising the thresholds based on f2 score, treating each class as independent. I implemented his
optimisation (utilities/Optimisations.py) and with a simple convolutional neural network I was able to achieve an f2-score of 0.69, equivalent to the score to beat of the example submission,
a reasonable starting point.

When I looked to more complex pre-trained models to fine tune I selected the Inception v3
model because I already had some familiarity with the model. To get started I followed the
[Keras guidelines on fine-tuning the Inception v3 model](https://keras.io/applications/) and
adapted them to this particular problem. The recommendation was to tune only the two top
inception modules, I tested tuning the top three and only the final module, but two indeed
gave the best performance. With this model I was able to achieve a public leaderboard score
of 0.90022 and 585th position out of 938. For the private leaderboard my final score was
0.89869 and 540th position out of 938.

I also wanted to implement data augmentation, which would have been my next step, and I wanted to fine-tune other models such as VGG16, VGG19 and ResNet50 which performed well for other
competition entrants, with a view to creating an ensemble of these models. From reading the
forums, had I had more time, I think this would have been the right approach to reach the much higher scores at the top of the leaderboard. I look forward to testing these out in my next competition!

In this repository you will find the main piece of code I used to train the model AmazonRainforestRunner.py and the code I used to load the best weights, perform k-fold cross-validation and produce the predictions and submission results for the test data, AmazonRainforestPredictions.py.

In the models I included the adaptation of the Inception v3 model I submitted.
Using a mixture of callbacks and writing intermediate information to numpy files I save information about the training and performance of my models and have some plotting functions to help me visualise this process (utilities/Plotting.py).
