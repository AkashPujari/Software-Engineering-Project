import logging, sys
from sklearn.metrics import recall_score,precision_score,f1_score

def parse_predictions(filename):
    predictions={}
    with open(filename) as input_:
        for x in input_:
            x = x.strip()
            idx1,idx2,label = x.split()
            predictions[(idx1,idx2)] = int(label)
    return predictions

def parse_truths(filename):
    truths={}
    with open(filename) as input_:
        for x in input_:
            x = x.strip()
            idx1,idx2,label = x.split()
            truths[(idx1,idx2)] = int(label)
    return truths

def calculate_results(predictions, truths):
    y_truths,y_preds=[],[]
    for key in truths:
        if key not in predictions:
            logging.error("Missing prediction for ({},{}) pair.".format(key[0],key[1]))
            sys.exit()
        y_truths.append(truths[key])
        y_preds.append(predictions[key])
    results={}
    results['Recall'] = recall_score(y_truths, y_preds)
    results['Prediction'] = precision_score(y_truths, y_preds)
    results['F1'] = f1_score(y_truths, y_preds)
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for BigCloneBench dataset.')
    parser.add_argument('--answers', '-a',help="filename of the truth labels, in txt format.")
    parser.add_argument('--predictions', '-p',help="filename of the predictions, in txt format.")
    

    args = parser.parse_args()
    predictions = parse_predictions(args.predictions)
    truths = parse_truths(args.answers)
    results = calculate_results(predictions, truths)
    print(results)

if __name__ == '__main__':
    main()
