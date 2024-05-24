import seaborn as sns
import matplotlib as mpl


def set_style():
    sns.set_theme(style="whitegrid", palette="Dark2", context="paper", font_scale=0.8)
    mpl.rcParams["figure.figsize"] = 3.5, 2.625
