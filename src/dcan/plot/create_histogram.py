import ast
import sys

import matplotlib.pyplot as plt


def main(input_file, output_file):
    with open(input_file) as f:
        data = f.read()

        d = ast.literal_eval(data)

        score = 0
        while score in d:
            plt.hist(d[score],
                     alpha=0.5,
                     label=f'score of {score}')
            score += 1

        plt.legend(loc='upper right')
        plt.title('Actual QC motion score vs. frequency of prediction')

        plt.savefig(output_file)
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
