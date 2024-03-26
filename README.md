# deep-learning-challenge

### Overview of the analysis: 

The purpose of this analysis is to develop a binary classifier using deep learning neural network machine learning,that can predict whether applicants will be successful if funded by Alphabet Soup.

### Results: 

** Data Preprocessing:

   *Target variable:IS_SUCCESSFUL

   *Features variables: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE,ORGANIZATION, INCOME_AMT, SPECIAL_CONSIDERATIONS

   *Removed variables:EIN,NAME

** Compiling, Training, and Evaluating the Model


   The neural network model consists of three dense  layers:

  * First hidden layer:80 neurons, with a ReLU activation function.

  * Second hidden layer: 30 neurons with a ReLU activation function.

  * Output layer: 1 neuron with a sigmoid activation .

    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    dense (Dense)               (None, 80)                3520      
                                                                    
    dense_1 (Dense)             (None, 30)                2430      
                                                                    
    dense_2 (Dense)             (None, 1)                 31        
                                                                    
    =================================================================
    Total params: 5981 (23.36 KB)
    Trainable params: 5981 (23.36 KB)
    Non-trainable params: 0 (0.00 Byte)

  * I selected 80 neurons in the first layer based on the large dataset size (34000+) to allow the model to learn better. Additionally, I chose 30 neurons in the second layer to reduce complexity. The output layer with 1 neuron is suitable for binary classification tasks.

  * I selected the ReLU activation function because it is suitable for identifying relationships between the features. Similarly, I chose the sigmoid activation function for the output layer because it is suitable for binary classification tasks


  * The Model achieves  an accuracy of 72.76%.I was not able to achieve target model performance(>75%)

  output:
   215/215 - 1s - loss: 0.5650 - accuracy: 0.7277 - 1s/epoch - 5ms/step
   Loss: 0.564972996711731, Accuracy: 0.7276967763900757

  * Optimization 1:

    Added an additional hidden layer, resulting in three hidden layers with 15 neurons, while maintaining the number of neurons (80, 30) and epochs (100).The Model achieves  an accuracy of 72.63%

  * Optimization 2:

    Increased the number of neurons from 80 to 128 in the first hidden layer , from 30 to 64 in the second hidden layer and from 15 to 32 in the third hidden layer while keeping the epochs(100).The Model achieves  an accuracy of 72.81%
    

  * Optimization 3:

    Set the batch_size to 64  while keeping the number of layers(3) and neurons(128,64,32,1).The Model achieves  an accuracy of 72.62%

### Summary:


The deep learning model achieved an accuracy of 72.76% . Despite implementing three different optimization attempts, including increasing the number of neurons, adding an extra hidden layer, and increasing the batch_size, the model's performance did not meet the desired threshold.

For further analysis, we can consider ensemble techniques like Random Forest or Gradient Boosting, which are suitable for binary classification output.
