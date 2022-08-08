import ast
import matplotlib.pyplot as plt

with open('/home/miran045/reine097/projects/AlexNet_Abrol2021/results/loes-scoring/model01/distributions.txt') as f:
    data = f.read()

d = ast.literal_eval(data)

for i in range(len(d)):
    plt.hist(d[i][1],
             alpha=0.5,
             label=f'score of {d[i][0]}')

plt.legend(loc='upper right')
plt.title('Predicted Loes score vs. frequency of prediction')

plt.savefig('/home/miran045/reine097/projects/AlexNet_Abrol2021/doc/loes_scoring/training_runs/loes_score_prediction_run01.png')
plt.show()
