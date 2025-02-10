import os
import pickle
import sys
from itertools import product

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, "SocialAIGym/src")

from data_component import DataComponent
from information_diffusion_component import BoundedConfidenceDiffusionComponent
from graph_utils import *


def get_real_data(dataset):
    data = DataComponent(real_data=f"{dataset.capitalize()}")

    data_folder = os.path.join(sys.path[0], "../data/processed/")

    topic = dataset.split("-")[0]

    # betweenness_path = os.path.join(data_folder, f"{topic}_betweenness_centrality.pkl")
    #
    # with open(betweenness_path, "rb") as f:
    #     betweenness_centrality = pickle.load(f)

    position_dict_path = os.path.join(data_folder, f"community_{topic}_position_dict.pkl")

    with open(position_dict_path, "rb") as f:
        position_dict = pickle.load(f)

    return data, position_dict # , betweenness_centrality


def get_network_parameters(NETWORK_OPINION, MODULARITY, HOMOPHILY):
    if NETWORK_OPINION == 'negative':
        alpha = 1
        beta = 10
    elif NETWORK_OPINION == 'positive':
        alpha = 10
        beta = 1
    elif NETWORK_OPINION == 'neutral':
        alpha = 10
        beta = 10
    elif NETWORK_OPINION == 'uniform':
        alpha = 1
        beta = 1
    else:
        raise Exception(f"network_opinion {NETWORK_OPINION} not in (positive, negative, neutral, uniform)")

    if MODULARITY == 'high':
        modularity = 0.75
    elif MODULARITY == 'low':
        modularity = 0.25
    elif MODULARITY == 'medium':
        modularity = 0.5
    else:
        raise Exception(f"modularity {MODULARITY} not in (high, low, medium)")

    if HOMOPHILY == 'high':
        homophily = 0.75
    elif HOMOPHILY == 'low':
        homophily = 0.25
    elif HOMOPHILY == 'medium':
        homophily = 0.5
    else:
        raise Exception(f"homophily {HOMOPHILY} not in (high, low, medium)")

    return alpha, beta, modularity, homophily


def compute_optimal_reward(bc_model, llm_node_id, search_granularity=1e-2):
    message_value_list = []
    reward_list = []

    search_space = np.arange(0., 1., search_granularity)
    for message_value in tqdm(search_space, total=len(search_space)):
        opinion_shift_tot, num_activated_users, _, _ = bc_model.propagate_message(message=message_value,
                                                                                  node_id=llm_node_id)
        message_value_list.append(message_value)
        reward_list.append(num_activated_users)
    return np.max(reward_list), {'message_opinion': message_value_list, 'reward': reward_list}


def get_optimal_rewards(TOPIC, create):
    df_path = os.path.join("[PATH]", f"{TOPIC}_optimal_rewards.csv")

    if not os.path.exists(df_path) or create:

        num_nodes = 100
        avg_deg = 5

        # bounded confidence model params
        epsilon = 0.2
        mu = 0.5

        optimal_rewards = []

        if TOPIC == "brexit":
            print("Computing optimal rewards for REAL BREXIT DATA")

            data, betweenness_centrality = get_brexit_data()

            data.pre_compute_neighboring()  # save neighbors for each node

            # 1. Init diffusion model ----------------------------------------
            information_diffusion_model = BoundedConfidenceDiffusionComponent(data_component=data,
                                                                              epsilon=epsilon,
                                                                              mu=mu)
            for LLM_pos in ("echo-high", "echo-low", "comm-largest", "comm-smallest", "central"):

                print("LLM_pos: ", LLM_pos)

                if LLM_pos == "echo-low":
                    llm_node_id = LLM_in_echochamber(data, "low")
                elif LLM_pos == "echo-high":
                    llm_node_id = LLM_in_echochamber(data, "high")
                elif LLM_pos == "comm-largest":
                    llm_node_id = LLM_in_comm(data, "largest")
                elif LLM_pos == "comm-smallest":
                    llm_node_id = LLM_in_comm(data, "smallest")
                elif LLM_pos == "central":
                    llm_node_id = LLM_central(data, betweenness_centrality=betweenness_centrality)

                optimal_reward, values = compute_optimal_reward(bc_model=information_diffusion_model,
                                                                llm_node_id=llm_node_id,
                                                                search_granularity=1e-5)

                print(
                    f"LLM_pos: {LLM_pos}")
                print("Optimal reward: ", optimal_reward)
                optimal_rewards.append([LLM_pos, optimal_reward])
                print(list(zip(values['message_opinion'], values['reward'])))

            df = pd.DataFrame(optimal_rewards,
                              columns=["LLM_pos", "Optimal_reward"])

        else:

            for LLM_pos in ("echo-high", "echo-low", "comm-largest", "comm-smallest", "central"):
                for MODULARITY in ("high", "low"):
                    for HOMOPHILY in ("high", "low"):
                        for NETWORK_OPINION in ('uniform', 'negative', 'neutral', 'positive'):

                            alpha, beta, modularity, homophily = get_network_parameters(NETWORK_OPINION=NETWORK_OPINION,
                                                                                        HOMOPHILY=HOMOPHILY,
                                                                                        MODULARITY=MODULARITY)

                            data = DataComponent(num_nodes, modularity, homophily, avg_deg, alpha=alpha, beta=beta)

                            if LLM_pos == "echo-low":
                                llm_node_id = LLM_in_echochamber(data, "low")
                            elif LLM_pos == "echo-high":
                                llm_node_id = LLM_in_echochamber(data, "high")
                            elif LLM_pos == "comm-largest":
                                llm_node_id = LLM_in_comm(data, "largest")
                            elif LLM_pos == "comm-smallest":
                                llm_node_id = LLM_in_comm(data, "smallest")
                            elif LLM_pos == "central":
                                llm_node_id = LLM_central(data)

                            data.pre_compute_neighboring()  # save neighbors for each node

                            # 1. Init diffusion model ----------------------------------------
                            information_diffusion_model = BoundedConfidenceDiffusionComponent(data_component=data,
                                                                                              epsilon=epsilon,
                                                                                              mu=mu)
                            optimal_reward, values = compute_optimal_reward(bc_model=information_diffusion_model,
                                                                            llm_node_id=llm_node_id,
                                                                            search_granularity=1e-5)

                            print(
                                f"LLM_pos: {LLM_pos}; MODULARITY: {MODULARITY}; HOMOPHILY: {HOMOPHILY}; NETWORK_OPINION: {NETWORK_OPINION}")
                            print("Optimal reward: ", optimal_reward)
                            optimal_rewards.append([LLM_pos, MODULARITY, HOMOPHILY, NETWORK_OPINION, optimal_reward])
                            print(list(zip(values['message_opinion'], values['reward'])))

            df = pd.DataFrame(optimal_rewards,
                              columns=["LLM_pos", "Modularity", "Homophily", "Network_opinion", "Optimal_reward"])

        df.to_csv(df_path, index=False)
    else:
        df = pd.read_csv(df_path)

    return df


def generate_df_scores(TOPIC):
    num_nodes = 100
    avg_deg = 5

    # bounded confidence model params
    epsilon = 0.2
    mu = 0.5

    TYPE = "completion"
    PROPAGATION = "bcm"

    batch_size = 8

    MODEL_NAME = "gemma2"

    path = f"[PATH]]"

    results = []

    if "brexit" in TOPIC or "referendum" in TOPIC:  # real_data
        results_path = os.path.join(path, f"{TOPIC}_results.csv")

        # low-degree positive
        # high-degree/high-centrality positive (sono uguali)
        # high-degree negative
        # high-centrality negative

        NETWORK_OPINIONS = ["positive", "negative"]
        LLM_POSITIONS = ["high-centrality", "low-centrality"]

        for NETWORK_OPINION in NETWORK_OPINIONS:
            data, position_dict = get_real_data(f"{TOPIC}-{NETWORK_OPINION}")

            data.pre_compute_neighboring()  # save neighbors for each node

            # 1. Init diffusion model ----------------------------------------
            information_diffusion_model = BoundedConfidenceDiffusionComponent(data_component=data,
                                                                              epsilon=epsilon,
                                                                              mu=mu)

            for LLM_pos in LLM_POSITIONS:

                if TOPIC == "brexit" and NETWORK_OPINION == "positive" and LLM_pos == "high-centrality":
                    LLM_pos = "high-degree"  # they are the same

                saving_fn = f"{MODEL_NAME}-{TYPE}-sentiment-propagation-readability-{PROPAGATION}-{TOPIC}-{NETWORK_OPINION}"

                config_path = f"initial-config-LLM_{LLM_pos}"

                utility_saving_path = os.path.join(path, config_path, saving_fn)

                print(utility_saving_path)

                # kls_path = os.path.join(utility_saving_path, "kl_values.npy")
                texts_path = os.path.join(utility_saving_path, "generated_texts.npy")
                sentiment_path = os.path.join(utility_saving_path, "sentiment_scores.npy")
                readability_path = os.path.join(utility_saving_path, "readability_scores.npy")
                propagation_path = os.path.join(utility_saving_path, "propagation_rewards.npy")
                rewards_path = os.path.join(utility_saving_path, "total_rewards.npy")

                try:
                    texts = np.load(texts_path)
                    sentiment_scores = np.load(sentiment_path)
                    # kls = np.load(kls_path)
                    readability_scores = np.load(readability_path)
                    propagation_scores = np.load(propagation_path)
                    rewards = np.load(rewards_path)

                    sentiment_scores = np.reshape(sentiment_scores,
                                                  (len(sentiment_scores) // batch_size, batch_size))
                    # readability_scores = np.reshape(readability_scores, (len(readability_scores), batch_size))
                    propagation_scores = np.reshape(propagation_scores,
                                                    (len(propagation_scores) // batch_size, batch_size))
                    rewards = np.reshape(rewards, (len(rewards) // batch_size, batch_size))

                    llm_node_id = position_dict[f"{NETWORK_OPINION}-{LLM_pos}"]

                    optimal_reward, values = compute_optimal_reward(bc_model=information_diffusion_model,
                                                                    llm_node_id=llm_node_id,
                                                                    search_granularity=1e-2)

                    optimal_indices = np.where(values["reward"] == optimal_reward)[0]
                    optimal_sentiments = np.array(values['message_opinion'])[optimal_indices]

                    results.append([LLM_pos,
                                    NETWORK_OPINION,
                                    np.mean(propagation_scores, axis=1),
                                    np.mean(sentiment_scores, axis=1),
                                    np.mean(readability_scores, axis=1),
                                    np.mean(rewards, axis=1),
                                    optimal_reward,
                                    optimal_sentiments])
                except:
                    print(f"FILE {utility_saving_path} NOT GENERATED YET.")

        results_df = pd.DataFrame(results, columns=["LLM_pos", "Network_opinion",
                                                    "Propagation", "Sentiment", "Readability", "Reward",
                                                    "Optimal_reward", "Optimal_sentiment"])

    else:

        prefix = ""
        VANILLA_RESULTS = True
        if VANILLA_RESULTS:
            prefix = "vanilla_"
        results_path = os.path.join(path, f"{prefix}{TOPIC}_results.csv")
        saving_fn = f"{MODEL_NAME}-{TYPE}-sentiment-propagation-readability-{PROPAGATION}-{TOPIC}"
        LLM_POSITIONS = ["echo-low", "echo-high", "comm-largest", "comm-smallest", "central"]

        print("Computing optimal rewards for SYNTHETIC DATA")

        OPINIONS = ["neutral", "positive", "negative", "uniform"]
        MODULARITIES = ["high", "low"]
        HOMOPHILIES = ["high", "low"]
        for LLM_pos in LLM_POSITIONS:
            for MODULARITY in MODULARITIES:
                for HOMOPHILY in HOMOPHILIES:
                    for NETWORK_OPINION in OPINIONS:

                        if (LLM_pos == "comm-largest" and MODULARITY == "high" and HOMOPHILY == "high") or \
                                (NETWORK_OPINION == "uniform" or NETWORK_OPINION == "neutral") and (
                                (LLM_pos == "comm-largest" and MODULARITY == "low" and HOMOPHILY == "high")) or \
                                ((NETWORK_OPINION == "uniform") and (
                                        LLM_pos == "comm-smallest" and MODULARITY == "high" and HOMOPHILY == "high")):
                            continue

                        # declare -a
                        config_path = f"initial-config-LLM_{LLM_pos}-MODULARITY_{MODULARITY}-HOMOPHILY_{HOMOPHILY}-NETWORK_OPINION_{NETWORK_OPINION}"

                        utility_saving_path = os.path.join(path, config_path, saving_fn)

                        print(utility_saving_path)

                        # kls_path = os.path.join(utility_saving_path, "kl_values.npy")
                        # texts_path = os.path.join(utility_saving_path, "generated_texts.npy")

                        sentiment_path = os.path.join(utility_saving_path, f"{prefix}sentiment_scores.npy")
                        readability_path = os.path.join(utility_saving_path, f"{prefix}readability_scores.npy")
                        propagation_path = os.path.join(utility_saving_path, f"{prefix}propagation_rewards.npy")
                        rewards_path = os.path.join(utility_saving_path, f"{prefix}total_rewards.npy")

                        try:
                            # texts = np.load(texts_path)
                            sentiment_scores = np.load(sentiment_path)
                            # kls = np.load(kls_path)
                            readability_scores = np.load(readability_path)
                            propagation_scores = np.load(propagation_path)
                            rewards = np.load(rewards_path)

                            if not VANILLA_RESULTS:

                                sentiment_scores = np.reshape(sentiment_scores,
                                                              (len(sentiment_scores) // batch_size, batch_size))

                                # readability_scores = np.reshape(readability_scores, (len(readability_scores), batch_size))
                                propagation_scores = np.reshape(propagation_scores,
                                                                (len(propagation_scores) // batch_size, batch_size))
                                rewards = np.reshape(rewards, (len(rewards) // batch_size, batch_size))

                                alpha, beta, modularity, homophily = get_network_parameters(
                                    NETWORK_OPINION=NETWORK_OPINION,
                                    HOMOPHILY=HOMOPHILY,
                                    MODULARITY=MODULARITY)

                                data = DataComponent(num_nodes, modularity, homophily, avg_deg, alpha=alpha, beta=beta)

                                if LLM_pos == "echo-low":
                                    llm_node_id = LLM_in_echochamber(data, "low")
                                elif LLM_pos == "echo-high":
                                    llm_node_id = LLM_in_echochamber(data, "high")
                                elif LLM_pos == "comm-largest":
                                    llm_node_id = LLM_in_comm(data, "largest")
                                elif LLM_pos == "comm-smallest":
                                    llm_node_id = LLM_in_comm(data, "smallest")
                                elif LLM_pos == "central":
                                    llm_node_id = LLM_central(data)

                                data.pre_compute_neighboring()  # save neighbors for each node

                                # 1. Init diffusion model ----------------------------------------
                                information_diffusion_model = BoundedConfidenceDiffusionComponent(data_component=data,
                                                                                                  epsilon=epsilon,
                                                                                                  mu=mu)
                                optimal_reward, values = compute_optimal_reward(bc_model=information_diffusion_model,
                                                                                llm_node_id=llm_node_id,
                                                                                search_granularity=1e-2)

                                optimal_indices = np.where(values["reward"] == optimal_reward)[0]
                                optimal_sentiments = np.array(values['message_opinion'])[optimal_indices]

                                results.append([LLM_pos, MODULARITY, HOMOPHILY, NETWORK_OPINION,
                                                np.mean(propagation_scores, axis=1),
                                                np.mean(sentiment_scores, axis=1),
                                                np.mean(readability_scores, axis=1),
                                                np.mean(rewards, axis=1),
                                                optimal_reward,
                                                optimal_sentiments])
                            else:
                                results.append([LLM_pos, MODULARITY, HOMOPHILY, NETWORK_OPINION,
                                                np.mean(propagation_scores),
                                                np.mean(sentiment_scores),
                                                np.mean(readability_scores),
                                                np.mean(rewards)])
                        except Exception as e:
                            print(e)

        if not VANILLA_RESULTS:
            results_df = pd.DataFrame(results, columns=["LLM_pos", "Modularity", "Homophily", "Network_opinion",
                                                        "Propagation", "Sentiment", "Readability", "Reward",
                                                        "Optimal_reward", "Optimal_sentiment"])
        else:
            results_df = pd.DataFrame(results, columns=["LLM_pos", "Modularity", "Homophily", "Network_opinion",
                                                        "Propagation", "Sentiment", "Readability", "Reward"])
    results_df.to_csv(results_path, index=False)


def plot_lineplot(x=None, y=None, xlabel=None, ylabel=None):
    plt.figure()

    sns.lineplot(x=x, y=y)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


def engagement_post_length_correlation(sentiment):
    def create_table(lengths, propagations, path):

        table = "| Length (no. characters) | Propagation |\n"
        table += "| ----- | ----- |\n"

        for i in range(len(lengths)):
            l = lengths[i]
            p = propagations[i]

            if p == 0.:
                continue

            table += "| {} | {} |\n".format(l, p)

        saving_table_file = os.path.join(path, "table.txt")
        with open(saving_table_file, "w") as f:
            f.write(table)

    topic = "brexit"
    LLM_pos = "high-degree"

    path = f"[PATH]" + f"initial-config-LLM_{LLM_pos}/"
    path += f"gemma2-completion-sentiment-propagation-readability-bcm-{topic}-{sentiment}"

    generated_posts = np.load(os.path.join(path, "generated_texts.npy"))
    generated_posts = np.reshape(generated_posts, generated_posts.shape[0] * 8)

    lengths = list(map(len, generated_posts))
    propagations = np.load(os.path.join(path, "propagation_rewards.npy"))

    non_zero_indices_propagations = np.where(propagations > 0)[0]

    cleaned_lengths = np.array(lengths)[non_zero_indices_propagations]
    cleaned_propagations = propagations[non_zero_indices_propagations]

    create_table(lengths, propagations, path)

    res = stats.pearsonr(lengths, propagations)
    pearson = res.statistic
    print("Person Correlation:", pearson)

    res = stats.pearsonr(cleaned_lengths, cleaned_propagations)
    cleaned_pearson = res.statistic
    print("Cleaned Person Correlation:", cleaned_pearson)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.regplot(x=lengths, y=propagations)

    plt.xlabel("Post Length (no. characters)", fontsize=20)
    plt.ylabel("Engagement", fontsize=20)

    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.grid(True, linestyle='--')

    ycoord = 0.6
    if sentiment == "positive":
        ycoord = 0.2

    plt.text(0.015, ycoord, "$\\rho$ =" + str(round(pearson, 3)), transform=plt.gca().transAxes, fontsize=25,
             bbox=dict(fill=False, edgecolor='gray', linewidth=2))

    plt.tight_layout()

    plt.savefig(os.path.join(path, f"engagement-post-length-correlation-{sentiment}.pdf"), dpi=200)
    plt.savefig(os.path.join(path, f"engagement-post-length-correlation-{sentiment}.png"), dpi=200)

    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.regplot(x=cleaned_lengths, y=cleaned_propagations)

    plt.xlabel("Post Length (no. characters)", fontsize=20)
    plt.ylabel("Engagement", fontsize=20)

    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.grid(True, linestyle='--')

    ycoord = 0.6
    if sentiment == "positive":
        ycoord = 0.2

    plt.text(0.015, ycoord, "$\\rho$ =" + str(round(cleaned_pearson, 3)), transform=plt.gca().transAxes, fontsize=25,
             bbox=dict(fill=False, edgecolor='gray', linewidth=2))

    plt.tight_layout()

    plt.savefig(os.path.join(path, f"cleaned_engagement-post-length-correlation-{sentiment}.pdf"), dpi=200)
    plt.savefig(os.path.join(path, f"cleaned_engagement-post-length-correlation-{sentiment}.png"), dpi=200)

    plt.show()


def main():
    TOPIC = "referendum"  # "brexit"  # cats are synthetic data - brexit/referendum are real data
    # df = get_optimal_rewards(TOPIC, create=False)
    # print(df)

    generate_df_scores(TOPIC=TOPIC)
    # sentiments = ["positive", "negative"]
    # for sentiment in sentiments:
    #     engagement_post_length_correlation(sentiment)


if __name__ == '__main__':
    main()
