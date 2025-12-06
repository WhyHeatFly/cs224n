# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    argp = argparse.ArgumentParser()
    argp.add_argument('--dev_birth_places')
    args = argp.parse_args()

    data = open(args.dev_birth_places, encoding='utf-8').readlines()
    predictions = ['London'] * len(data)
    total, correct = utils.evaluate_places(args.dev_birth_places, predictions)
    accuracy = correct / total * 100.0
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
