
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from adjustText import adjust_text

def plot_umap(datasets, n_neighbors=5, min_dist=0.01, sample_fraction=1.0, point_size=1, alpha=0.1, title="UMAP Projection", ax=None):
    combined_data = []
    labels = []
    # Combine datasets and create labels, with sampling if specified
    for label, data in datasets.items():
        if sample_fraction < 1.0:
            indices = np.random.choice(data.shape[0], size=int(data.shape[0] * sample_fraction), replace=False)
            data = data[indices]
        combined_data.append(data)
        labels.extend([label] * data.shape[0])
    combined_data = np.vstack(combined_data)

    # Perform UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_jobs=-1)
    embedding = reducer.fit_transform(combined_data)

    # Plot the results
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Use the order of datasets.keys() for label order
    for label in datasets.keys():
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        ax.scatter(embedding[indices, 0], embedding[indices, 1], label=label, s=point_size, alpha=alpha)

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()

    return pd.DataFrame({'UMAP1': embedding[:, 0], 'UMAP2': embedding[:, 1], 'labels': labels})



# Adapted from https://hemtools.readthedocs.io/en/latest/content/Bioinformatics_Core_Competencies/Volcanoplot.html
def plot_volcano(df, x_col, y_col, x_threshold, y_threshold,*,labels=False,left_label="Markers",right_label="Drivers",title='Marker to driver',ax=plt):
    if ax==plt:
        ax.figure(figsize=(12, 12))
    ax.scatter(x=df[x_col],y=df[y_col].apply(lambda x:x),s=1,label="Low impact",color="grey")
    down = df[(df[x_col]<=-x_threshold)&(df[y_col]>=y_threshold)]
    up = df[(df[x_col]>=x_threshold)&(df[y_col]>=y_threshold)]
    ax.scatter(x=down[x_col],y=down[y_col].apply(lambda x:x),s=3,label=left_label,color="red")
    ax.scatter(x=up[x_col],y=up[y_col].apply(lambda x:x),s=3,label=right_label,color="green")
    if labels:
        texts=[]
        for i,r in up.iterrows():
            texts.append(ax.text(x=r[x_col],y=r[y_col],s=i))
        for i,r in down.iterrows():
            texts.append(ax.text(x=r[x_col],y=r[y_col],s=i))
        adjust_text(texts,arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

    ax.axvline(-x_threshold,color="grey",linestyle="--")
    ax.axvline(x_threshold,color="grey",linestyle="--")
    ax.axhline(y_threshold,color="grey",linestyle="--")
    if ax!=plt:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
    else:
        ax.xlabel(x_col)
        ax.ylabel(y_col)
        ax.title(title)
    ax.legend()

def plot_volcano_one_sided(df, x_col, y_col, x_threshold, y_threshold,*,labels=False,left_label="Markers",right_label="Drivers",title='Marker to driver',ax=plt):
    if ax==plt:
        ax.figure(figsize=(12, 12))
    ax.scatter(x=df[x_col],y=df[y_col].apply(lambda x:x),s=1,label="Low impact",color="grey")
    up = df[(df[x_col]>=x_threshold)&(df[y_col]>=y_threshold)]
    ax.scatter(x=up[x_col],y=up[y_col].apply(lambda x:x),s=3,label=right_label,color="green")
    if labels:
        texts=[]
        for i,r in up.iterrows():
            texts.append(ax.text(x=r[x_col],y=r[y_col],s=i))
        #for i,r in down.iterrows():
        #    texts.append(ax.text(x=r[x_col],y=r[y_col],s=i))
        adjust_text(texts,arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

    #ax.axvline(-x_threshold,color="grey",linestyle="--")
    ax.axvline(x_threshold,color="grey",linestyle="--")
    ax.axhline(y_threshold,color="grey",linestyle="--")
    if ax!=plt:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
    else:
        ax.xlabel(x_col)
        ax.ylabel(y_col)
        ax.title(title)
    ax.legend()


def plot_distances(category_labels, real_start_distances, fake_start_distances,
                   real_goal_distances, fake_goal_distances, main_path, outdir):
    for i, name in enumerate(category_labels):
        plt.figure(figsize=(15, 6))  # Set the overall figure size
        plt.subplot(1, 4, 1)
        sns.kdeplot(real_start_distances[i], bw_adjust=0.5, fill=True)
        plt.title("Real Start distance: " + name)
        plt.ylabel('Count')
        plt.xlabel('Distance')
        plt.xlim(0, 1)
        
        plt.subplot(1, 4, 2)
        sns.kdeplot(fake_start_distances[i], bw_adjust=0.5, fill=True)
        plt.title("Fake Start distance: " + name)
        plt.ylabel('Count')
        plt.xlabel('Distance')
        plt.xlim(0, 1)
        
        plt.subplot(1, 4, 3)
        sns.kdeplot(real_goal_distances[i], bw_adjust=0.5, fill=True)
        plt.title("Real Goal distance: " + name)
        plt.ylabel('Count')
        plt.xlabel('Distance')
        plt.xlim(0, 1)
        
        plt.subplot(1, 4, 4)
        sns.kdeplot(fake_goal_distances[i], bw_adjust=0.5, fill=True)
        plt.title("Fake Goal distance: " + name)
        plt.ylabel('Count')
        plt.xlabel('Distance')
        plt.xlim(0, 1)
        
        plt.tight_layout()  # Adjust layout for better spacing
        plt.savefig(main_path + outdir + "/" + name + "_FoolTheExternalModel.pdf")
        plt.show()



def plot_top_features(
    violins: np.ndarray,
    names:   list[str],
    top_n:   int       = 25,
    *,
    measure: str       = "median",   # "median" or "mean"
    error:   str       = "sem",      # "sem" (standard error) or "std"
    title:   str       = "Oracle score",
    location:str       = "feature_plot.pdf",
):
    """
    Plot the top_n features by either median or mean, with error bars.
      - measure="median"  uses np.median
      - measure="mean"    uses np.mean
      - error="sem"       is std/√N (or 1.253·std/√N if measure="median")
      - error="std"       is raw standard deviation
    """
    # 1) rank features by the chosen central tendency
    if measure == "median":
        central_all = np.median(violins, axis=1)
    else:
        central_all = np.mean(violins, axis=1)

    order    = np.argsort(central_all)[::-1][:top_n]
    block    = violins[order, :]
    labels   = np.array(names)[order]

    # 2) compute centers and errors
    if measure == "median":
        centers = np.median(block, axis=1)
    else:
        centers = np.mean(block, axis=1)

    stds    = np.std(block, axis=1, ddof=1)
    N       = block.shape[1]

    if error == "std":
        errs = stds
    else:
        if measure == "median":
            errs = 1.253 * stds / math.sqrt(N)
        else:
            errs = stds / math.sqrt(N)

    # 3) plot
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        labels, centers, errs,
        fmt="o", markersize=8, capsize=5, linestyle="None"
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Oracle score")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.savefig(location)
    plt.show()

    return central_all


def plot_perturbation(violins,names, top_n=10,title="temp",location="temp.pdf"):
    medians_all = np.median(violins, axis=1)
    positions = np.argsort(medians_all)[::-1]
    positions = positions[:top_n]
    subset_violins = violins[positions,:]
    gene_names = np.array(names[positions])
    # Calculate mean and standard deviation
    medians = np.median(subset_violins, axis=1)
    #print(min(means_all))
    e = (1.96 * np.std(subset_violins, axis=1) / math.sqrt(subset_violins.shape[1]))
    plt.figure(figsize=(30, 20))
    plt.errorbar(gene_names, medians,e, linestyle='None', marker='^')
    plt.title(title)
    plt.ylabel('Closeness to goal (larger is better)')
    plt.xlabel('Genes')
    plt.xticks(rotation=45, fontsize=15)
    plt.savefig(location)
    plt.show()
    return medians_all
