# CS390-Lab0
CS390NIP Lab0

## Custom Net
To implement this section, I first started by looking at the slides. Working off of the skeleton code, I reviewed the slides 
that were over how to calculate backpropagation. From this, I implemented training, calculating the values needed for backpropagation
as I went. By using the weights, I can run a feed forward process and get the output of the individual layers. From there I calculate layer 2 
error, layer 2 delta, layer 1 error, layer 1 delta, and using these values, I was able to calculate the adjustments for each layer. 
I then added the adjustments to the respective weights and ran the training model again for each input. To calculate the layer adjustments, I 
had to reshape the matrices to the correct size as I wanted the adjustment matrices to be the same dimensions as the weight matrices. Finally,
I took the prediction vector and one hot encoded it so that each output would give a "real" prediction rather then a confidence value
for each individual number. 

## TensorFlow Net
To implement this section, I first started by looking at the Keras documentation and the slides from the class.
Using the code in the slides as an example to work from, I started to format the class that I would need to create
the neural net. Modeling it somewhat after the skeleton code provided for the custom neural net, I created an init, train,
run, and predict function. From implementing the net given the example slides I created the net. The only other  thing that I
needed to do was implement the one hot encoding of the prediction output. Before this, since the last layer used softmax, I 
was getting an output vector of floats that summed to one, but I wanted the prediction, or the value that was the max in the
list of prediction values. After one hot encoding, I was able to output which number the neural net thought the input was.  

