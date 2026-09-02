import numpy as np

class Perceptron:
    def __init__(self, eta = 0.01, n_iter = 50, random_state =1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self,X, y):
        # s-1 : random weights are generated 
        rgen = np.random.RandomState(self.random_state)
        # s-2 : Update the random weights generated into the self.w_ by normalising them
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.b = 0.0

        errors = []

        # S-3 Start a loop for iterating for each epoch 
        for _ in range(self.n_iter):
            errors_ = 0 
        # S-4 Start a for loop where xi, target is mapped to each column of X, y and update both weight and bias
            for xi, target in zip(X,y):
                update += self.eta *(target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update

                errors += int(update !=0.0)
            self.errors_.append(errors)
        return self


    def net_input(self,X):
        net = np.dot(X, self.w_) + self.b_
        return net

    def predict(self, X,y):
        return np.where(self.net_input(X)>0.0,1,0)
        