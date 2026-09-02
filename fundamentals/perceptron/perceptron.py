import numpy as np

class Perceptron:
    """
    Creating a single layer Perceptron

    Parameters:(These are configuration variables (hyperparameters) that are supplied by the 
    user before training begins. They control how the perceptron learns but are not learned 
        from the training data.)
    These the mathematical things like weights, biases which are used in the mathematical equation
    1) eta - learning rate-> a hyperparameter that determines how quickly the perceptron 
        can adjust its weights -> float -> between (0.0,1.0)

    2) n_iter -> number of epochs -> int
        ( 
        -Epochs are basically how many times did the ml model iterate over the dataset
        -It allows the algorithm to process every sample, calculate errors, and 
        update internal weights multiple times.
        -Too many Epochs can lead to overfitting - ie the model has memorized the data set thus
            hurting the real life performance of the model
        -Way too less epochs can lead to underfitting- ie the model failed to learn the patterns
        )
    3) random_state -> setting weights to random values that are very small
        --> Why can we not set the weights and bias to 0 or any constant number:
               1)   In a single layer perceptron it wont matter much 
                    But in a multi layer perceptron if we set all the weights to the same value then
                    the neurons will give similar outputs due to which the different layers of the neural
                    network wont be well used
               2)   It also breaks the symmetry between the neurons thus helps in the ml model
                    learning new featurs 
               3)   It positions the network in a region of weight space where gradients flow,
                    and makes training reproducible when paired with a fixed random seed
    In this we take random_state = 1 (This acts like a seed so that the same random states are generated)
    
    Attributes: ( Attributes are qualities or characteristics that describe an object, individual, or phenomenon)
    These are the object properties 

    1) w_ -> 1d Array
        Weights after fitting
    2) b_ -> Scalar
        Bias Unit after fitting

    errors_ : list 
        Number of misclassifications ( updates ) in each epoch  
        We couldnt use a numpy array here because numpy arrrays are hard to resize
        
    """

    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Parameters : (Input Values)
        X: this is the data set passed in
        {array like}, shape =[n_examples, n_features]
        Training vectors where n_examples is the number of examples and n_features is the 
        number of features 
        Here array-like means it accepts array like data such as numpy arrray , numpy list of lists

        y: This is the correct prediction that should have been made by the perceptron after being trained (basically y^)
         
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1]) # Draw random samples from a normal (Gaussian) distribution.
        """
        loc -> Mean (Simply the mid point of the data curve)
        scale -> Standard deviation ( How spread out the data is)
            Sets a very small standard deviation. 
            This ensures the random weights stay tiny (mostly between -0.03 and +0.03)
            so the model doesn't start with oversized values.
        size -> How many values do you want the cmoputer to give out and of what  shape 
            Pulls the total number of features (columns) from your data matrix X 
            so each feature gets exactly one weight.
        """

        self.b_ = 0.0  
        self.errors_ = [] 
        # In scikit-learn, the convention is: Attributes ending with _ are created or learned during fit().

        #The learning of the model actually starts
        for _ in range(self.n_iter):
            errors_ = 0 
            """
            zip(X, y): This functions like a physical zipper. 
            It takes the 1st element of X and pairs it with the 1st element of y into a tuple. 
            Then it does the same for the 2nd elements, the 3rd elements, and so on.
            """

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update

                errors_ += int(update != 0.0)
            self.errors_.append(errors_)
        return self

    def net_input(self, X):
        """Returns the net input"""
        ni = np.dot(X, self.w_) + self.b_
        return ni

    def predict(self, X):
        """Returns class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    # if self.net_input(x) >= 0.0, then return 1 else return 0
