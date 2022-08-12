import ast
import os.path
import sys

import matplotlib.pyplot as plt


def create_plot(run_number):
    formatted_run_str = "%02d" % (run_number,)
    project_path = '/home/miran045/reine097/projects/AlexNet_Abrol2021'
    with open(os.path.join(project_path, f'results/loes-scoring/model{formatted_run_str}/distributions.txt')) as f:
        data = f.read()

    d = ast.literal_eval(data)

    for i in range(len(d)):
        plt.hist(d[i][1],
                 alpha=0.5,
                 label=f'score of {d[i][0]}')

    plt.legend(loc='upper right')
    plt.title('Predicted Loes score vs. frequency of prediction')

    plt.savefig(
        os.path.join(project_path, f'doc/loes_scoring/training_runs/loes_score_prediction_run{formatted_run_str}.png'))
    plt.show()


if __name__ == "__main__":
    create_plot(int(sys.argv[1]))
