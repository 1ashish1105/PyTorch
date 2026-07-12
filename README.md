# PyTorch
Going to discuss everything about the Pytorch its concept and its practical implementation


1.Tensors:

 --> It is multidimentional array that is designed for the mathematical and computation efficiency.
 --> First we are going to discuss about the topic Tensors.
 --> So tensors are basically vectors like
    i.  Scaler or 0-D vectors like 1 or 2 any single number is example of Scaler.
    ii. 1-D tensor- Arrays is one of the best example of 1-D tensors. eg-[1,2,3,4,5], In NLP embedding is one of the best example of 1-D tensors.
    iii. 2-D tensor-It is in 2 direction array and best example is Grayscale image or eminist Dataset.
    iv. 3-D tensors- RGB is one of the exmple of 3-D tensor.
    v. 4-D tensors- Batches of RGB images.
    v1. 5-D tensors -Best example is Video. Video is basically represented as sequence of frames.
Uses:-i. Used in doing Mathematical Operations.
      ii. Used in storing real world data in like photo and videos.
      iii. Used in Deep learning for storing data and and perform calculations like WX+b.

2. Autograd:
   --> It is a tool which helps us to calculate the diffrentiation automaticaaly.
   --> We know that in depp Neural Network we have to work very hard to calculate difrentiation because it is very large nested neural networks so cal manully is not possible so use Autograd.

3. NN Modules :
   --> The torch.nn module in PyTorch is a core library that provide wode array of class and functions to develop and build our neural networks efficiently and effectively. It abstracts the compexity of creating and training neural network by offering pre-built layers,loss functions,acttivation functions, and other utilities, enable us to focus and design loss function.

 Key Component of torch.nn :-
  1.Modules(Layers):-
    i. It is the base class for all neural network module
    ii. Common layers like (fully connected layer),nn.Conv2D(convolutional layer), nn.LSTM(recurrent layer) and many others.
  2.Activation Functions:-
    i. We have all functions like relu,sigmoid and tanh to use in Neural Network.
  3.Loss Functions:-
   i. Provides us loss functions such as nn.CrossEntropy, nn.MSELoss and nn.NLLLoss to  quantifyu diff between models predictions.
  4.Container Module:-
   i. nn.Sequential- A sequential container to stack layers in order
  5.Regularization and Dropout:-
   i. Layers like nn.Dropout and nn.BatchNorm2d help prevent overfitting and imporve accuracy.
























   






    
