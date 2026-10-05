"""Quick sanity check that validates imports and model shapes without CICIDS2017."""
import numpy as np
from src.models import seed_everything, train_rf, train_cnn, cnn_proba
seed_everything(42)
X=np.random.randn(400,20).astype('float32'); y=(X[:,0]+0.4*X[:,1]>0).astype('int64')
rf=train_rf(X,y,20,42); cnn,_,dev=train_cnn(X,y,20,epochs=1,batch_size=64)
print('RF OK:',rf.predict(X[:3])); print('CNN OK:',cnn_proba(cnn,X[:3],dev))
