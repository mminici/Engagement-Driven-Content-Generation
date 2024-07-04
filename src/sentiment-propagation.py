#!/usr/bin/env python
# coding: utf-8
from statistics import harmonic_mean

# In[1]:


hugging_token = "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"

# In[2]:


from huggingface_hub import login

login(hugging_token)

# In[3]:


import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

import torch
import networkx as nx

from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    pipeline
)
import pandas as pd
import numpy as np
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

import matplotlib.pyplot as plt

# In[4]:

import readability

from transformers import set_seed

set_seed(42)

# In[5]:

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="google/gemma-2b-it",
                                      metadata={"help": "the model name"})  # "NousResearch/Llama-2-7b-chat-hf"
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="lvwerra/distilbert-imdb",
                                             metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for input"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=1, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=True, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="/mnt/nas/coppolillo/LLMs/ppo_checkpoints/",
                                      metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})


# parser = HfArgumentParser(ScriptArguments)
# script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
parser = HfArgumentParser((ScriptArguments,))
script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
set_seed(script_args.seed)

# In[6]:


# Propagation Model
from pathlib import Path
import os, sys

sys.path.insert(0, "SocialAIGym/src")

from data_component import DataComponent
from information_diffusion_component import BoundedConfidenceDiffusionComponent
from opinion_diffusion_component import FJDiffusionComponent
from graph_utils import *


def main(TOPIC, TYPE, PROPAGATION, NETWORK_OPINION, 
         MODULARITY='medium', HOMOPHILY='medium', LLM_pos=0
    ):
    # synthetic data generator params

    num_nodes = 100
    avg_deg = 10

    # bounded confidence model params
    epsilon = 0.2
    mu = 0.5

         
    if NETWORK_OPINION == 'negative':
        alpha = 1
        beta = 10
    elif NETWORK_OPINION == 'positive':
        alpha = 10
        beta = 1
    elif NETWORK_OPINION == 'neutral':
        alpha = 1
        beta = 1
    else:
        raise Exception(f"network_opinion {NETWORK_OPINION} not in (positive, negative, neutral)")   


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
 
           

    data = DataComponent(num_nodes, modularity, homophily, avg_deg, alpha=alpha, beta=beta)

    if LLM_pos == 0:
        llm_node_id = 0
    elif LLM_pos == "echo-low":
        llm_node_id = LLM_in_echochamber(data, "low")
    elif LLM_pos == "echo-high":
        llm_node_id = LLM_in_echochamber(data, "high")
    elif LLM_pos == "comm-largest":
        llm_node_id = LLM_in_comm(data, "largest")
    elif LLM_pos == "comm-smallest":
        llm_node_id = LLM_in_comm(data, "smallest")
    elif LLM_pos == "central":
        llm_node_id = LLM_central(data)


    # In[7]:

    if PROPAGATION == "bcm":
        data.pre_compute_neighboring()  # save neighbors for each node
        information_diffusion_model = BoundedConfidenceDiffusionComponent(data_component=data, epsilon=epsilon, mu=mu)
        llm_node_id = 0

    elif PROPAGATION == "fj":
        information_diffusion_model = FJDiffusionComponent(data_component=data)
        betweenness_centrality = nx.betweenness_centrality(data.get_graph())
        sorted_by_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
        llm_node_id = sorted_by_betweenness[0][0]

        # compute minimum
        zoomed_fj = np.linspace(-5, 5, 500)

        zoomed_results = []

        for l in tqdm(zoomed_fj, total=len(zoomed_fj)):
            zoomed_results.append(
                information_diffusion_model.polarization_plus_disagreement_at_equilibrium(l, llm_node_id))

        fj_minimum = round(min(zoomed_results))

    opinions = information_diffusion_model.get_opinions()

    # Statistics of Initial Configuration
    ## 1. opinions
    print(
        f'Opinions stats \nmean: {opinions.mean()}\nstd: {opinions.std()}\nmin: {opinions.min()}\nmax: {opinions.max()}')

    _ = plt.hist(opinions, bins=25, range=[0, 1])
    plt.xlabel('opinion value')
    plt.ylabel('occurrences')
    plt.show()

    ## 2. graph
    intial_config_path = Path("../data/intial_config")
    os.makedirs(intial_config_path, exist_ok=True)
    filename = f"initial-config-LLM_{LLM_pos}-MODULARITY_{MODULARITY}-HOMOPHILY_{HOMOPHILY}-alpha_{alpha}-beta_{beta}-avgdeg_{avg_deg}.pdf"
    plot_graph_config(data, llm_node_id, intial_config_path / filename)

    # In[8]:

    attrs = {}
    for n in data.G.nodes():
        attrs.update({n: {"y": n, "x": [opinions[n]]}})
    nx.set_node_attributes(data.G, attrs)

    # In[9]:

    # Below is an example function to build the dataset. In our case, we use the IMDB dataset
    # from the `datasets` library. One should customize this function to train the model on
    # its own dataset.
    def build_dataset(
            tokenizer, dataset_name, input_min_text_length=2, input_max_text_length=8
    ):
        """
        Build dataset for training. This builds the dataset from `load_dataset`, one should
        customize this function to train the model on its own dataset.

        Args:
            dataset_name (`str`):
                The name of the dataset to be loaded.

        Returns:
            dataloader (`torch.utils.data.DataLoader`):
                The dataloader for the dataset.
        """

        # train_dataset = load_dataset(dataset_name, split="train")

        if TYPE == "completion":
            verb = "are"
            if TOPIC == "Obamacare" or TOPIC == "Brexit":
                verb = "is"
            prompt = f"{TOPIC.capitalize()} {verb} the most"
        elif TYPE == "generation":
            prompt = f"Generate a post about {TOPIC}."

        size = config.batch_size
        df = pd.DataFrame(np.repeat(prompt, size), columns=["0"])
        ds = Dataset.from_dict(df)

        train_dataset = ds.rename_columns({"0": "question"})

        original_columns = train_dataset.column_names
        num_proc = 24

        def preprocess_function(examples):
            new_examples = {
                "query": [],
                "input_ids": [],
            }
            for question in examples["question"]:
                #  query = "Question: " + question + "\n\nAnswer: \n"
                query = question
                tokenized_question = tokenizer(query, truncation=True, max_length=script_args.max_length)
                new_examples["query"].append(query)
                new_examples["input_ids"].append(tokenized_question["input_ids"])

            return new_examples

        ds = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=original_columns,
        )
        ds = ds.filter(lambda x: len(x["input_ids"]) <= script_args.max_length, batched=False)

        ds.set_format(type="torch")
        return ds

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    reward_model_name = script_args.reward_model_name

    model_name = script_args.model_name
    # model_name = script_args.output_dir + f"gemma2-{TYPE}-readability-{TOPIC}-{NETWORK_OPINION}"

    config = PPOConfig(
        early_stopping=True,
        target_kl=50,
        # init_kl_coef=1,
        model_name=model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        batch_size=script_args.batch_size,
        mini_batch_size=script_args.mini_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        ppo_epochs=script_args.ppo_epochs,
        seed=script_args.seed
    )

    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    rw_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": script_args.batch_size,
        "truncation": True
    }

    if "decapoda" in script_args.model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
        # required for llama
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN,
            }
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(tokenizer, script_args.dataset_name)

    print(dataset["query"][0])

    if "Llama" in config.model_name:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

    if "gemma" in config.model_name:
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear"
        )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        load_in_8bit=False,
        device_map="auto",
        peft_config=lora_config
    )

    optimizer = None
    if script_args.adafactor:
        optimizer = Adafactor(
            filter(lambda p: p.requires_grad, model.parameters()),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=config.learning_rate,
        )

    # In[14]:

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer
    )

    def generate_response(question, model, tokenizer):
        inputs = tokenizer(question, return_tensors="pt").to(device)

        # correct_inputs = tokenizer(correct_answer, return_tensors="pt").to(inputs_device)

        outputs = model.generate(**inputs, max_new_tokens=150)  # =correct_inputs["input_ids"].shape[1])
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # In[17]:

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug

    TRAIN = True

    output_min_length = 32
    output_max_length = script_args.output_max_length
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    if TYPE == "completion":
        N_EPOCHS = 80
    elif TYPE == "generation":
        N_EPOCHS = 500

    sentiment_model = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb")

    kl_values = []
    generated_texts = []
    sentiment_scores = []
    total_rewards = []
    propagation_rewards = []
    readability_scores = []

    # # TODO
    # ### - FJ non funziona per come vogliamo noi (e.g., sballa totalmente al variare della posizione dell LLM)
    # ### - Modificare la distribuzione delle opinioni sul grafo
    # ### - Capire se funziona questo raffinamento della soglia -> il valore del messaggio non deve essere centrato su 0.5
    # ### - Continuare a cercare cose per la fluency (e.g., classificatore basato su BERT)

    bcm_threshold = 0.2

    # In[ ]:

    if TRAIN:

        # We then build the sentiment analysis pipeline, passing the model name and the
        # sentiment analysis pipeline arguments. Let's also make sure to set the device
        # to the same device as the PPOTrainer.

        # reward_model = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb")

        # We then define the arguments to pass to the `generate` function. These arguments
        # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
        # the `generate` function of the trained model.
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": 100_000
        }

        for epoch in tqdm(range(N_EPOCHS), total=N_EPOCHS):
            for batch in tqdm(ppo_trainer.dataloader, total=len(ppo_trainer.dataloader)):
                question_tensors = batch["input_ids"]

                response_tensors = ppo_trainer.generate(
                    question_tensors,
                    return_prompt=False,
                    length_sampler=output_length_sampler,
                    **generation_kwargs,
                )
                batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                print(batch["response"])

                # Compute sentiment score
                texts = [q + r for q, r in zip(batch["query"], batch["response"])]

                generated_texts.append(texts)
                readabilities = [readability.getmeasures(x, lang='en')['readability grades']["Kincaid"]
                                 for x in texts]

                readability_scores.append(readabilities)

                # if PROPAGATION == "bcm":
                sentiment_outputs = sentiment_model(texts)
                messages_values = [output["score"] if output["label"] == "POSITIVE" else 1 - output["score"]
                                   for output in sentiment_outputs]

                # elif PROPAGATION == "fj":
                #    sentiment_outputs = sentiment_model(texts, **rw_kwargs)
                #    messages_values = [output[1]["score"] for output in sentiment_outputs]

                sentiment_scores.extend(messages_values)

                rewards = []

                for i, message_value in enumerate(messages_values):
                    if PROPAGATION == "bcm":

                        # discouraging neutral posts
                        # if abs(message_value - 0.5) < bcm_threshold:
                        #    reward = 0
                        # else:
                        opinion_shift_tot, num_activated_users, _ = information_diffusion_model.propagate_message(
                            message=message_value,
                            node_id=llm_node_id)

                        reward = num_activated_users

                    elif PROPAGATION == "fj":
                        pol_dis = information_diffusion_model.polarization_plus_disagreement_at_equilibrium(
                            message_value,
                            llm_node_id)

                        reward = -(pol_dis - fj_minimum) ** 2

                    propagation_rewards.append(reward)

                    # COMBINING PROPAGATION REWARD AND POST READABILITY

                    # HARMONIC MEAN -> DOES NOT WORK
                    # reward = harmonic_mean([reward, readabilities[i]])

                    # ARITHMETIC MEAN -> COULD WORK (SHOULD BE WEIGHTED)
                    # reward = (reward + readabilities[i])/2

                    # GEOMETRIC MEAN
                    reward = np.sqrt(reward * readabilities[i])

                    total_rewards.append(reward)

                    rewards.append(torch.tensor(reward, dtype=torch.float))

                # torch_readability_scores = [torch.tensor(x, dtype=torch.float) for x in readability_scores[-1]]
                # rewards.extend(torch_readability_scores)

                # Run PPO step
                stats = ppo_trainer.step(question_tensors, response_tensors, rewards)

                stats["env/readability"] = np.mean(readability_scores[-1])
                stats["env/sentiment"] = np.mean(messages_values)
                stats["env/propagation"] = np.mean(propagation_rewards[-len(batch):])

                ppo_trainer.log_stats(stats, batch, rewards)

                kls = (stats["objective/logprobs"] - stats["objective/ref_logprobs"]).mean(axis=1)
                kl_values.append(kls)

            kl = stats["objective/kl"]
            early_stop = ppo_trainer._early_stop(kl)

            if early_stop:
                print(f"Early stopping... KL is equal {kl}")
                break

    # saving_fn = f"gemma2-{TYPE}-sentiment-propagation-{PROPAGATION}-{TOPIC}-SOFTMAX-{NETWORK_OPINION}"
    saving_fn = f"gemma2-{TYPE}-sentiment-propagation-readability-{PROPAGATION}-{TOPIC}-{NETWORK_OPINION}"

    saving_path = script_args.output_dir + saving_fn

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    model.save_pretrained(saving_path)
    tokenizer.save_pretrained(saving_path)

    utility_saving_path = f"/mnt/nas/coppolillo/LLMs/{saving_fn}"

    if not os.path.exists(utility_saving_path):
        os.makedirs(utility_saving_path)

    kls_path = os.path.join(utility_saving_path, "kl_values.npy")
    texts_path = os.path.join(utility_saving_path, "generated_texts.npy")
    sentiment_path = os.path.join(utility_saving_path, "sentiment_scores.npy")
    readability_path = os.path.join(utility_saving_path, "readability_scores.npy")
    propagation_path = os.path.join(utility_saving_path, "propagation_rewards.npy")
    rewards_path = os.path.join(utility_saving_path, "total_rewards.npy")

    np.save(kls_path, kl_values)
    np.save(texts_path, generated_texts)
    np.save(sentiment_path, sentiment_scores)
    np.save(readability_path, readability_scores)
    np.save(propagation_path, propagation_rewards)
    np.save(rewards_path, total_rewards)

    # ## Possibile estensione:
    #
    # ### - Plottare in 2D (UMAP) i prompt generati una volta finito il fine-tuning
    # ### - Cambiare il BCM per avere lo spostamento non in valore assoluto ma che shifti sull'asse (in modo da de-/polarizzare)
    # ### - Modificare il task e fare esempi minimi

    # In[ ]:

    device_ref_model = "cuda:0"
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(script_args.model_name, device_map=device_ref_model)

    # In[37]:

    LOAD = not TRAIN

    if LOAD:
        device_model = "auto"

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            saving_path,
            device_map=device_model
        )

        tokenizer = AutoTokenizer.from_pretrained(saving_path)

        tokenizer.pad_token = tokenizer.eos_token

        ppo_trainer = PPOTrainer(config, model, None, tokenizer)

    # In[40]:

    #### get a batch from the dataset
    bs = 8
    game_data = dict()
    dataset.set_format("pandas")
    df_batch = dataset[:].sample(bs)
    game_data["query"] = df_batch["query"].tolist()
    query_tensors = df_batch["input_ids"].tolist()

    # In[41]:

    response_tensors_ref = []

    gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
                  "pad_token_id": tokenizer.eos_token_id}

    for i in range(bs):
        gen_len = output_length_sampler()
        output = ref_model.generate(
            torch.tensor(query_tensors[i], device=device_ref_model).unsqueeze(dim=0), max_new_tokens=gen_len,
            **gen_kwargs
        ).squeeze()[-gen_len:]
        response_tensors_ref.append(output)

    # In[43]:

    response_tensors = []

    for i in range(bs):
        output = ppo_trainer.generate(
            torch.tensor(query_tensors[i], device=device), max_new_tokens=gen_len, **gen_kwargs
        ).squeeze()[-gen_len:]
        response_tensors.append(output)

    # In[44]:

    #### decode responses
    game_data["response (before)"] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
    game_data["response (after)"] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

    #### sentiment analysis of query/response pairs before/after
    texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]

    game_data_sentiments = [output["score"] if output["label"] == "POSITIVE" else 1 - output["score"] for output in
                            sentiment_model(texts)]
    game_data["rewards (before)"] = [information_diffusion_model.propagate_message(message=x,
                                                                                   node_id=llm_node_id)[1] for x in
                                     game_data_sentiments]

    texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]

    game_data_sentiments = [output["score"] if output["label"] == "POSITIVE" else 1 - output["score"] for output in
                            sentiment_model(texts)]
    game_data["rewards (after)"] = [information_diffusion_model.propagate_message(message=x,
                                                                                  node_id=llm_node_id)[1] for x in
                                    game_data_sentiments]

    # store results in a dataframe
    df_results = pd.DataFrame(game_data)

    df_path = utility_saving_path + "_df.csv"

    df_results.to_csv(df_path, index=False)
    print(df_results)

    print("mean:")
    print(df_results[["rewards (before)", "rewards (after)"]].mean())
    print()
    print("median:")
    print(df_results[["rewards (before)", "rewards (after)"]].median())


if __name__ == '__main__':
    TYPE = "completion"  # "generation"  #
    PROPAGATION = "bcm"  # "fj" #
    NETWORK_OPINION = "neutral"  # "positive"  # negative

    TOPICS = ["cats"]  # , "Obamacare", "Brexit", "vaccines", "gender rights"]
    for TOPIC in TOPICS:
        main(TOPIC, TYPE, PROPAGATION, NETWORK_OPINION)
