from sklearn.preprocessing import StandardScaler
import numpy as np

class StandardTotalVarianceScaler(StandardScaler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def fit(self, X, y=None, sample_weight=None):
        T = super().fit(X, sample_weight=sample_weight)
        #Usually np.sqrt(var). Here, we artificially inflate
        #this by np.sqrt(the number of features) so the data 
        #will be further scaled down for a total variance of 1.
        #as implemented here, this still scales features individually.
        #could instead scale the whole vector by the total variance,
        #and thus preserve relative scaling within the vector.
        T.scale_ *= np.sqrt(T.n_features_in_)
        return T