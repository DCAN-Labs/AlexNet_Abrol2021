import ast
import sys

import matplotlib.pyplot as plt


def main(input_file, output_file):
    with open(input_file) as f:
        data = f.read()

        d = ast.literal_eval(data)

        plt.hist(d[1],
                 alpha=0.5,
                 label='score of 1')
        plt.hist(d[2],
                 alpha=0.5,
                 label='score of 2')
        plt.hist(d[3],
                 alpha=0.5,
                 label='score of 3')
        plt.hist(d[4],
                 alpha=0.5,
                 label='score of 4')
        plt.hist(d[5],
                 alpha=0.5,
                 label='score of 5')

        plt.legend(loc='upper right')
        plt.title('Actual QC motion score vs. frequency of prediction')

        plt.savefig(output_file)
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
