import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker

df = pd.read_csv('./log/mychem.csv')

print("列名：", df.columns)

if 'Experiment Name' not in df.columns:
    raise ValueError("CSV文件中没有找到 'Experiment Name' 列，请检查列名是否正确。")

sns.set(style="darkgrid")

plt.figure(figsize=(12, 8), dpi=800)
fontsize = 30

df['Experiment Name'] = df['Experiment Name'].replace({
    'fedspec': 'FedCP',
    'feddbe': 'FedSSP',
    'fedavg': 'FedAvg',
    'fedprox': 'FedProx',
    'fedstar': 'FedStar',
    'fedsage': 'FedSage'
})

default_color = '#888888'
colors = {
    'FedCP': '#2ca02c',
    'FedSSP': '#AB1D0A',
    'FedAvg': '#1f77b4',
    'FedProx': '#C0B510',
    'FedStar': 'darkblue',
    'FedSage': '#9467bd',
    'GCFL': '#ff7f0e'
}

grouped_df = df.groupby('Experiment Name')

df['Round'] = pd.to_numeric(df['Round'])
df = df[df['Round'] >= 1]

for idx, (name, group) in enumerate(grouped_df):
    aggregated_df = group.groupby('Round').agg({'Mean Acc': ['mean', 'std']}).reset_index()
    aggregated_df.columns = ['Round', 'Mean Acc', 'Std Acc']

    window_size = 23
    aggregated_df['Mean Acc Smooth'] = aggregated_df['Mean Acc'].rolling(window=window_size, center=True).mean() * 100

    aggregated_df['Std Acc Smooth'] = aggregated_df['Std Acc'].rolling(window=9, center=True).mean() * 100

    line_color = colors.get(name, default_color)

    line_width = 4 if name == 'FedSSP' else 1.5

    sns.lineplot(data=aggregated_df, x='Round', y='Mean Acc Smooth', color=line_color, linewidth=line_width)

    plt.fill_between(aggregated_df['Round'], aggregated_df['Mean Acc Smooth'] - aggregated_df['Std Acc Smooth'],
                     aggregated_df['Mean Acc Smooth'] + aggregated_df['Std Acc Smooth'], color=line_color, alpha=0.2)

plt.gca().tick_params(axis='both', which='major', labelsize=fontsize)

plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(50))

plt.ylim(65, 85) 

plt.savefig('comparison_of_methods_smooth_std_dev.png', dpi=800)

plt.close()
