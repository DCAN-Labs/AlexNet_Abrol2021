import os

import pandas as pd
from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt


parent_folder = '/home/feczk001/shared/data/loes_scoring/Loes_score/'
spreadsheet = os.path.join(parent_folder, 'loes_scores.csv')
df = pd.read_csv(spreadsheet, index_col=False)
row_count = len(df.index)
print(f'Subject/session count: {row_count}')
flair_count = df['flair'].sum()
print(f'flair file count: {flair_count}')
flair_sizes = df.flair_size.unique()
print(f'flair image sizes: {flair_sizes}')
mprage_count = df['mprage'].sum()
print(f'mprage file count: {mprage_count}')
mprage_sizes = df.mprage_size.unique()
print(f'mprage image sizes: {mprage_sizes}')
swi_count = df['swi'].sum()
print(f'swi file count: {swi_count}')
swi_sizes = df.swi_size.unique()
print(f'swi image sizes: {swi_sizes}')

ax = df.hist(column='Loes score', bins=25, grid=False, figsize=(12, 8), color='#86bf91', zorder=2, rwidth=0.9)

ax = ax[0]
for x in ax:
    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)

    # Switch off ticks
    x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = x.get_yticks()
    for tick in vals:
        x.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Remove title
    x.set_title("")

    # Set x-axis label
    x.set_xlabel("Loes score", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x.set_ylabel("Frequency", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    x.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

plt.savefig('/home/miran045/reine097/projects/AlexNet_Abrol2021/doc/loes_scoring/hist.png')
plt.show()
