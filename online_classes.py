from universal.algo import Algo
from universal import tools
import numpy as np

class OLRANDOM(Algo):

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLRANDOM, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps


    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b, history):
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window:])
        b = self.update(last_b, x_pred, self.eps)
        return b


    def predict(self, x, history):
        """ Predict returns on next day, selecting returns randomly among the last days in the window 
            THIS IS OUR MAIN MODIFICATION
            
        if self.each_share : 
            x_pred = np.ones(history.shape[1])
        
            for i in range(history.shape[1]): # for each share
                
                x_pred[i] =  np.asanyarray(history[[i]].sample(random_state = 123456))
                x_pred = x_pred.reshape((history.shape[1]))
                
        else :
            x_pred =  np.asanyarray(history.sample(random_state = 123456)).reshape((history.shape[1]))
        """
        x_pred =  np.asanyarray(history.sample(random_state = 123456)).reshape((history.shape[1]))
        return x_pred
    
    
    def update(self, b, x, eps):
        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean)**2)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x - x_mean)

        # project it onto simplex
        return tools.simplex_proj(b)


class OLGAUSS(Algo):

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10, mu = 1, sigma = 0.01 ):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLGAUSS, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps
        self.mu = mu
        self.sigma = sigma

    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b, history):
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window:])
        b = self.update(last_b, x_pred, self.eps)
        return b


    def predict(self, x, history):
        """ Predict the price relatives of the next day. 
            The prediction is a vector of gaussian centered in 1
            THIS IS OUR MAIN MODIFICATION
        """
        np.random.seed(123456)
        x_pred = np.random.normal(self.mu, self.sigma, history.shape[1]).reshape((history.shape[1]))
        return x_pred
    
    
    def update(self, b, x, eps):
        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean)**2)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x - x_mean)

        # project it onto simplex
        return tools.simplex_proj(b)

class OLEWM(Algo):

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10, alpha=0.6):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLEWM, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps
        self.alpha = alpha

    def init_weights(self, m):
        return np.ones(m) / m

    def step(self, x, last_b, history):
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window:])
        b = self.update(last_b, x_pred, self.eps)
        return b


    def predict(self, x, history):
        """ Predict returns on next day. 
            THIS IS OUR MAIN MODIFICATION
        """
        x_pred = np.asarray(history.ewm(alpha=self.alpha).mean().iloc()[-1]).reshape(history.shape[1])
        return x_pred / x

    def update(self, b, x, eps):
        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean)**2)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x - x_mean)

        # project it onto simplex
        return tools.simplex_proj(b)
    
class OLMEDIAN(Algo):

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLMEDIAN, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps


    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b, history):
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window:])
        b = self.update(last_b, x_pred, self.eps)
        return b


    def predict(self, x, history):
        """ Predict returns on next day. 
            THIS IS OUR MAIN MODIFICATION
        """
        
        return (history / x).median()

    def update(self, b, x, eps):
        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean)**2)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x - x_mean)

        # project it onto simplex
        return tools.simplex_proj(b)

class OLMAX(Algo):

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLMAX, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps


    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b, history):
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window:])
        b = self.update(last_b, x_pred, self.eps)
        return b


    def predict(self, x, history):
        """ Predict returns on next day. 
            THIS IS OUR MAIN MODIFICATION
        """
        
        return (history / x).max()

    def update(self, b, x, eps):
        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean)**2)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x - x_mean)

        # project it onto simplex
        return tools.simplex_proj(b)
    
class OLMIN(Algo):

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLMIN, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps


    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b, history):
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window:])
        b = self.update(last_b, x_pred, self.eps)
        return b


    def predict(self, x, history):
        """ Predict returns on next day. 
            THIS IS OUR MAIN MODIFICATION
        """
        
        return (history / x).min()

    def update(self, b, x, eps):
        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean)**2)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x - x_mean)

        # project it onto simplex
        return tools.simplex_proj(b)

class OLMAR_max_k(Algo):
    """ On-Line Portfolio Selection with Moving Average Reversion
    Reference:
        B. Li and S. C. H. Hoi.
        On-line portfolio selection with moving average reversion, 2012.
        http://icml.cc/2012/papers/168.pdf
    """

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10, k = 1):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLMAR_max_k, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps
        self.k = k

    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b, history):
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window:])
        b = self.update(last_b, x_pred, self.eps)
        return b


    def predict(self, x, history):
        """ Predict returns on next day. """
        return (history / x).mean()


    def update(self, b, x, eps):
        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        
        x = np.asarray(x)
        
        sorted_idx = np.argsort(x)[::-1]
        
        allocation = np.zeros(len(x))
        allocation[sorted_idx[:self.k]] = 1.0

        return (allocation + 0.0)/ np.sum(allocation)
