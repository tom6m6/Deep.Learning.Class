import numpy as np
import pickle

def compute_acc(pred_file):
    with open('./data/test_labels', 'rb') as f:
        gold = np.asarray(pickle.load(f)).reshape(-1)

    with open(pred_file) as f:
        pred = f.readlines()
    pred = [int(sent.strip()) for sent in pred]
    correct_case = [i for i, _ in enumerate(gold) if gold[i] == pred[i]]

    acc = len(correct_case)*1./len(gold)
    print('The predicted accuracy is %s' %acc)

if __name__ == '__main__':
    pred_file = 'data/predict.txt'
    compute_acc(pred_file)