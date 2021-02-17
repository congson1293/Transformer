import numpy as np
from nltk.translate.bleu_score import sentence_bleu

references = np.array([[1,2,3,4,5]])
candidates = np.array([[1,2,3,5,4]])

print(sentence_bleu(references, candidates))