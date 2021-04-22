# DRL Project 1 - Banana Navigator
This repository contains the solution materials for a trained Deep Q-Learning Network (DQN) that navigates through a custom Banana Navigator Unity Environment. The materials include:
* jupyter notebook with executed cell output showing trained model performance
* notebook exported as an html report
* saved model weights of the successful agent
* model file class definitions for the RL Agent and the Q-Network  
`NOTE: model files and notebook adapted from the LunarLander example in the DRLND repository.`   

##### Dependencies
The ***Navigation.ipynb*** Jupyter Notebook ran with a Python 3 kernel in a Udacity-supported network workspace which came pre-installed with the custom Unity Environment as well as several python packages

___
### Environment/Agent Interaction
The agent traverses a 2D planar environment using an epsilon-greedy policy that selects from 4 actions (forward/backward/left/right) to collect yellow bananas (reward of +1) and avoid blue bananas (reward of -1) that are randomly distributed for each episode. The agent achieves success once it completes 100 consecutive episodes with an average score of 13 or more. The environment state contains 37 elements that quantify characteristics like object perspective and agent velocity.
___
### Model Architecture 
The deep neural network consists of three fully connected linear layers, chosen based on the representation of the input state for this environment - a simple low-dimensional vector. If the state input were a 2D pixel array, for example, then convolutional layers would be more appropriate.  
The first hidden layer transforms the input state from size 37 to size 64 before passing the output through a ReLU activation function. 64 was a large enough hidden layer size to represent sufficiently complex relationships for this task. The second hidden layer is identical to the first, and also links to a ReLU activation function. The final linear layer output logits provide the predicted Q-values for all actions, conditional on the state input.
___
### Training
The agent trained according to stochastic gradient descent with a mean squared error loss function and Adam optimizer, processing minibatches of size 64. The behavior policy epsilon decayed from 1.0 to 0.01 exponentially, where it remained fixed. Each episode ran at most 1000 timesteps, and the max number of episodes was 2000. The agent learned by exploiting key techniques, like a replay buffer, frame-skipping, and fixed-Q targets, that were originally demonstrated in the [original DQN paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf).
___
### Evaluation
Performance was graded based on average score over the most recent 100 episode window. This DQN agent managed to achieve a winning average score of `13.06` after only `534 episodes`.
