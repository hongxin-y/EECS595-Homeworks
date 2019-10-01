import pandas as pd
import numpy as np
import math
import pickle
import sys
import getopt

class HMMTagger():
    def __init__(self, A, B, pi):
        self.A = A
        self.B = B
        self.pi = pi
    
    def fit(self, O):
        return Viterbi(self.A, self.B, self.pi, O)

# A(N,N), B(N,M), pi(N), O(T), N is # states, M is the # observations, T is the # time intervals
# probs have been in log form
def Viterbi(A, B, pi, O):
    T = O.shape[0]
    N = A.shape[0]
    M = B.shape[1]
    v = np.array([[float('-inf') for _ in range(N)] for __ in range(T)], dtype = 'float64')
    for i in range(N):
        v[0][i] = pi[i] + B[i][O[0]]
    pre = range(N)
    for t in range(1,T):
        for j in range(N):
            for i in range(N):
                v[t][j] = max(v[t-1][i] + A[i][j], v[t][j])
            v[t][j] += B[j][O[t]]
    return v.argmax(1)

def load_observation(model, words_map, filename):
    state = []
    idx = len(words_map)
    for line in open(filename):
        obs = []
        words = line.split()
        for word in words:
            if(word not in words_map):
                obs.append(0)
            else:
                obs.append(words_map[word])
        obs = model.fit(np.array(obs))
        state = np.concatenate((state, obs))
    return np.array(state)

def test(model, words_map, labels_map, test_filename, validation_filename):
    test_state = load_observation(model, words_map, test_filename)
    state = []
    idx = len(labels_map)
    for line in open(validation_filename):
        words = line.split()
        for label in words[1::2]:
            state.append(labels_map[label])
    state = np.array(state)
    acc = 100*sum(test_state==state)/test_state.shape[0]
    return acc

def main(argv = None):
    if argv is None:
        argv = sys.argv
    test_file, validate_file = argv[1], argv[2]
    f = open('model.pyc', 'rb')
    hmm, labels_map, words_map = pickle.load(f)
    acc = test(hmm, words_map, labels_map, test_file, validate_file)
    print("The accuracy is %.2f%%." % (acc))
    return 0
   
if __name__ == "__main__":
    main()