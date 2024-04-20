import logging
import sys
import json
import numpy as np

def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            good_idx = js['cwe'] + '/' + js['language'] + '/' + 'good' + js['cwe_id']
            bad_idx = good_idx.replace('good', 'bad')
            answers[good_idx] = 0

            if len(js['functions_before_patches']) > 0:
                answers[bad_idx] = 1
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            idx, label = line.split()
            predictions[str(idx)] = int(label)
    return predictions

def calculate_scores(answers,predictions):
    Acc=[]
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        Acc.append(answers[key]==predictions[key])

    scores={}
    scores['Acc']=np.mean(Acc)
    return scores

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument('--answers', '-a',help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p',help="filename of the leaderboard predictions, in txt format.")
    

    args = parser.parse_args()
    answers=read_answers(args.answers)
    predictions=read_predictions(args.predictions)
    scores=calculate_scores(answers,predictions)
    print(scores)

if __name__ == '__main__':
    main()
