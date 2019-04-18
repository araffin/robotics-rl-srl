import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas

if __name__ == '__main__':
    rc('font', weight='bold')
    ae = [216, 240, 237, 236, 240, 224, 230, 214]
    raw_pixels = [89, 156, 8, 6, -2, -2, 0, 0]
    gt = [245, 243, 239, 241, 242, 246, 243, 230]
    srl_combination = [187, 191, 191, 206, 140, 185, 81, 69]
    srl_splits = [223, 183, 227, 194, 193, 185, 181, 184]
    supervised = [240, 243, 240, 237, 239, 239, 240, 237]
    random = [99, 50, 65, 116, 224, 211, 198, 50]

    perfs_replay_real = gt + supervised + srl_splits + ae + srl_combination + random + raw_pixels

    ae = [241, 228, 217, 245, 241, 235, 237, 242]
    raw_pixels = [194, 165, -2, 226, 235, 136, 124, 224]
    gt = [236, 240, 243, 246, 247, 237, 246, 240]
    srl_combination = [235, 224, 225, 231, 209, 228, 242, 235]
    srl_splits = [244, 239, 230, 206, 242, 238, 241, 240]
    supervised = [235, 241, 243, 246, 248, 235, 240, 245]
    random = [225, 177, 213, 174, 235, 207, 225, 229]

    perfs_replay_sim = gt + supervised + srl_splits + ae + srl_combination + random + raw_pixels
    tags_replay = ["Ground Truth" for i in range(8)] + ["Supervised" for i in range(8)] + \
                  ["SRL Splits" for i in range(8)] + ["Auto-encoder" for i in range(8)] + \
                  ["SRL Combination" for i in range(8)] + ["Random Feat." for i in range(8)] +\
                  ["Raw Pixels" for i in range(8)]

    df = pandas.DataFrame({'SRL Method': tags_replay, "Real": perfs_replay_real,
                           "Simulation": perfs_replay_sim})
    print(df["Real"])
    sns.set(style="ticks", palette="colorblind")

    # Draw a nested boxplot to show bills by day and time
    dd = pandas.melt(df, id_vars=['SRL Method'], value_vars=['Simulation', 'Real'], var_name='Setting')
    b = sns.boxplot(x='SRL Method', y='value', data=dd, hue='Setting')
    b.tick_params(labelsize=15)

    #b.set_xlabel(, fontsize=20)
    #b.set_ylabel("Rewards", fontsize=20)

    plt.xlabel("SRL Method", fontsize=20, fontweight='bold')
    plt.ylabel("Rewards", fontsize=20, fontweight='bold')
    plt.setp(b.get_legend().get_texts(), fontsize='22')  # for legend text
    plt.setp(b.get_legend().get_title(), fontsize='24')
    plt.show()
