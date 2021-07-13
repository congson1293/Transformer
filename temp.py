import joblib
import numpy as np

pretrain = joblib.load('models/embed_vocab.pkl')
pretrain_weight = np.zeros((len(pretrain), 768))
for i in range(len(pretrain)):
    weight = pretrain[i].detach().cpu().numpy()
    pretrain[i] = weight
joblib.dump(pretrain_weight, 'models/embed_weight_decoder.pkl')
