import numpy as np

class DataPreprocess():
    """
    A class for data preprocessing
    """
    
    def __init__(self):
        
        self.window_size = None
        self.X_data = None
        self.y_data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
    
    def get_data(self, stock_data, window_size, split_ratio=(0.6,0.2,0.2)):
        """Get data for the model

        Args:
            stock_data (pd.DataFrame): Stock data with date in its index and tickers in its columns
            window_size (int): Window size to look back in the model
            split_ratio (tuple, optional): Split ratio for train, validation and test. Defaults to (0.6,0.2,0.2).
        """
        # Normalize data as an input, X
        self._normalize_data(stock_data)
        
        # Get rank data as a target, y
        self._get_rank(stock_data)
        
        # Generate data with sliding window methods
        X, y = self._sliding_window(window_size) # X.shpae = (# of stocks, t, window)
        
        # Split the data into train, validation and test
        self._train_val_test_split(X, y, split_ratio)
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
        
    def _get_rank(self, stock_data):
        """Get rank data with given input data

        Args:
            stock_data (pd.DataFrame): Stock data with date in its index and tickers in its columns
        """
        self.y_data = stock_data.pct_change().rank(axis=1)
        
    def _normalize_data(self, stock_data):
        """Normalize data

        Args:
            stock_data (pd.DataFrame): Stock data with date in its index and tickers in its columns
        """
        self.X_data = stock_data.apply(lambda x: x / x.max())

    def _sliding_window(self, window_size):
        """Create dataset using sliding window

        Args:
            window_size (int): Window size to look back in the model

        Returns:
            X (np.array): An array of input data with sliding window
            y (np.array): An array of target output data with sliding window
        """
        self.window_size = window_size
        X = []
        y = []

        for i in range(len(self.X_data) - self.window_size -1): # -1 because it is the last index for y

            window_X_data = self.X_data[i:i+self.window_size].values
            window_y_data = self.y_data.iloc[i+self.window_size].values

            X.append(window_X_data)
            y.append(window_y_data)

        X = np.stack(X, 0).T # shape = (N, window, input_size)
        y = np.stack(y, 0).T # shape = (N, input_size)
        
        return X, y
    
    def _train_val_test_split(self, X, y, split_ratio):
        """Split the data into three parts: train, validation and test

        Args:
            X (np.array): An array of input data with sliding window
            y (np.array): An array of target output data with sliding window
            split_ratio (tuple, optional): Split ratio for train, validation and test. Defaults to (0.6,0.2,0.2).
        """
        
        # Get each ratio from the tuple
        train_pct, val_pct, test_pct = split_ratio
        
        # Generate two indices for split
        idx_1 = int(X.shape[-1]*train_pct)
        idx_2 = int(X.shape[-1]*(train_pct+val_pct))
        
        # Indexing for both X and y
        self.X_train = X[:,:,:idx_1]
        self.X_val = X[:,:,idx_1:idx_2]
        self.X_test = X[:,:,idx_2:]
        
        self.y_train = y[:,:idx_1]
        self.y_val = y[:,idx_1:idx_2]
        self.y_test = y[:,idx_2:]
        