import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/miran045/reine097/projects/AlexNet_Abrol2021/data/BCP/qc_with_paths_standardized.csv')
print(df['rating'].min())
print(df['rating'].max())
df.hist(column='rating', bins=5)

plt.savefig('/home/miran045/reine097/projects/AlexNet_Abrol2021/doc/motion_qc_score/img/bcp_hist.png')

plt.show()
