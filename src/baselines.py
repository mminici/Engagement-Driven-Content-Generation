#!/usr/bin/env python
# coding: utf-8

# In[5]:


from transformers import BertTokenizer, BertModel, EncoderDecoderModel, AutoTokenizer, pipeline
import torch
import nltk
import numpy as np
import os
import seaborn as sns
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# In[6]:


torch.manual_seed(42)
np.random.seed(42)

# In[7]:


saving_folder = "[PATH]"

if not os.path.exists(saving_folder):
    os.makedirs(saving_folder)

# In[16]:


topics = ["Cats", "Brexit"]
models = ["bert", "gpt2", "llama3.1", "chatgpt4o", "vanilla-gemma", "finetuned-gemma"]

# # Visualizing Rewards

# In[46]:


batch_size = 8

# In[60]:


strategy = "mean"

if strategy == "mean":
    func = np.mean
    index = 0
# In[54]:


images_path = os.path.join(saving_folder, "images", f"{strategy}_strategy")
if not os.path.exists(images_path):
    os.makedirs(images_path)

# In[55]:


dfs_path = os.path.join(saving_folder, "dataframes", f"{strategy}_strategy")
if not os.path.exists(dfs_path):
    os.makedirs(dfs_path)

# In[56]:


for saving_score in ["engagement"]:

    plot_legend = True
    saving_dict_path = os.path.join(saving_folder, f"{saving_score}s_dict.pkl")

    with open(saving_dict_path, "rb") as f:
        scores_dict = pickle.load(f)

    for TOPIC in ["Brexit"]:

        if TOPIC == "Cats":
            llm_positions = ["echo-high", "echo-low", "comm-largest", "comm-smallest", "central"]
            opinions = ["positive", "negative", "neutral", "uniform"]
            modularities = ["high", "low"]
            homophilies = ["high", "low"]
        else:
            llm_positions = ["high-centrality", "low-centrality"]
            opinions = ["positive", "negative"]
            modularities = [None]
            homophilies = [None]

        for NETWORK_OPINION in opinions:
            for MODULARITY in modularities:
                for HOMOPHILY in homophilies:

                    score_values = []
                    valid_positions = llm_positions.copy()

                    for LLM_pos in llm_positions:

                        for i, model in enumerate(models):

                            if "gemma" in model:

                                if TOPIC == "Cats":
                                    folder = f"initial-config-LLM_{LLM_pos}-MODULARITY_{MODULARITY}-HOMOPHILY_{HOMOPHILY}-NETWORK_OPINION_{NETWORK_OPINION}"
                                    subfolder = f"gemma2-completion-sentiment-propagation-readability-bcm-{TOPIC.lower()}"

                                else:
                                    # l = LLM_pos.replace("centrality", "degree")

                                    folder = f"initial-config-LLM_{LLM_pos}"
                                    subfolder = f"gemma2-completion-sentiment-propagation-readability-bcm-{TOPIC.lower()}-{NETWORK_OPINION}"

                                file_str = ""

                                if saving_score == "reward":
                                    score_type = "total"
                                else:
                                    score_type = "propagation"

                                file_str += f"{score_type}_rewards.npy"

                                saving_path = os.path.join(saving_folder.replace("baselines/", ""), folder, subfolder,
                                                           file_str)

                                try:
                                    scores = np.load(saving_path)
                                except Exception as e:
                                    print(e)
                                    print(f"Removing {LLM_pos} from {valid_positions}")
                                    valid_positions.remove(LLM_pos)
                                    del score_values[-i:]
                                    break

                                if "vanilla" in model:
                                    score = func(scores[:batch_size])
                                else:
                                    reshaped_scores = np.reshape(scores, (len(scores) // batch_size, batch_size))
                                    score = np.max(func(reshaped_scores, axis=1))

                            else:
                                key_str = f"model_{model}_opinion_{NETWORK_OPINION}_mod_{MODULARITY}_hom_{HOMOPHILY}_pos_{LLM_pos}_topic_{TOPIC}"

                                score = scores_dict[key_str][index]  # 0: taking the mean, 1: taking the maximum

                            score_values.append(score)

                    if TOPIC == "Cats":
                        fn = f"{saving_score}_opinion_{NETWORK_OPINION}_mod_{MODULARITY}_hom_{HOMOPHILY}_topic_{TOPIC}"
                    else:
                        fn = f"{saving_score}_opinion_{NETWORK_OPINION}_topic_{TOPIC}"

                    df = pd.DataFrame(list(zip(np.repeat(valid_positions, len(models)), score_values,
                                               np.tile(models, len(score_values) // len(models)))),
                                      columns=["LLM_pos", "score", "model"])

                    df.to_csv(os.path.join(dfs_path, fn + ".csv"), index=False)

                    ax = sns.barplot(x="LLM_pos", y="score", data=df, hue="model")  # ,legend=False) # , sharex=False)
                    sns.despine(bottom=True, left=True)
                    plt.xlabel("LLM Position")
                    plt.ylabel(f"{saving_score.capitalize()}")
                    plt.tight_layout()

                    if plot_legend:
                        handles, labels = ax.get_legend_handles_labels()
                        fig_legend, axi = plt.subplots(figsize=(3.3, .4))

                        fig_legend.legend(handles, labels, ncol=len(models))
                        axi.axis('off')
                        fig_legend.tight_layout()
                        fig_legend.savefig(os.path.join(images_path, "baseline_legend.png"), dpi=200)

                        plot_legend = False

                    ax.get_legend().remove()

                    plt.savefig(os.path.join(images_path, f"{fn}.png"), dpi=200)
                    plt.show()

# In[59]:
