import gc
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"

import torch
import networkx as nx

from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser,
    pipeline
)
import pathlib
import pickle
import pandas as pd
import numpy as np
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

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
    model_name: Optional[str] = field(default="NousResearch/Llama-2-7b-chat-hf", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="lvwerra/distilbert-imdb",
                                             metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
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
    output_dir: Optional[str] = field(default="/mnt/nas/minici/SocialAIGym/data/interim/ppo_checkpoints/",
                                      metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})


parser = HfArgumentParser((ScriptArguments,))
script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]
set_seed(script_args.seed)


## Utility functions:
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

    prompt = "Generate a post about cats."

    size = 10000
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
            query = "Question: " + question + "\n\nAnswer: \n"
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


# Propagation Model
import os, sys

sys.path.insert(0, "../src/")

from data_component import DataComponent
from information_diffusion_component import BoundedConfidenceDiffusionComponent

## HYPER PARAMS ------------------------------------
# synthetic data generator params
num_nodes = 100
modularity = 0.5
homophily = 0.5

# bounded confidence model params
epsilon = 0.2
mu = 0.5

TRAIN = True
LOAD = False  # TODO ??
output_min_length = 32

# ---------------------------------------------------


reward_model_name = script_args.reward_model_name
config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
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

# GRID EXPERIMENTS
modularity_grid = [.8, ]  # [.2, .8]
homophily_grid = [.8, ]  # [.2, .8]
llm_positions = [0, 1, -2, -1]

RESULTS = {}

for modularity in modularity_grid:
    for homophily in homophily_grid:
        # 0. Init data component ----------------------------------------
        data = DataComponent(num_nodes, modularity, homophily)
        data.pre_compute_neighboring()  # save neighbors for each node

        # 0.1 Calculate betweenness centrality for all nodes
        betweenness_centrality = nx.betweenness_centrality(data.get_graph())
        sorted_by_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)

        for position in llm_positions:
            # 0.1 Set node position
            llm_node_id = sorted_by_betweenness[position][0]

            # 1. Init diffusion model ----------------------------------------
            information_diffusion_model = BoundedConfidenceDiffusionComponent(data_component=data, epsilon=epsilon,
                                                                              mu=mu)

            opinions = information_diffusion_model.get_opinions()
            print(
                f'Opinions stats \nmean: {opinions.mean()}\nstd: {opinions.std()}\nmin: {opinions.min()}\nmax: {opinions.max()}')
            attrs = {}
            for n in data.G.nodes():
                attrs.update({n: {"y": n, "x": [opinions[n]]}})
            nx.set_node_attributes(data.G, attrs)
            # ----------------------------------------------------------------------

            # 2. Build data component - We retrieve the dataloader by calling the `build_dataset` function.
            dataset = build_dataset(tokenizer, script_args.dataset_name)

            # ----------------------------------------------------------------------

            # 3. Build model components --------------------------------------------
            # a. AutoModelForCausalLMWithValueHead,
            # b. optimizer,
            # c. PPOTrainer
            # Now let's build the model, the reference model, and the tokenizer.

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
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

            # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
            ppo_trainer = PPOTrainer(
                config,
                model,
                ref_model=None,
                tokenizer=tokenizer,
                dataset=dataset,
                data_collator=collator,
                optimizer=optimizer
            )
            device = ppo_trainer.accelerator.device
            if ppo_trainer.accelerator.num_processes == 1:
                device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug <------- TODO: What is this bug? Why device 0?

            # 4. Utility components --------------------------------------------
            # 4.1 LengthSampler
            output_max_length = script_args.output_max_length
            output_length_sampler = LengthSampler(output_min_length, output_max_length)

            # 4.2 Sentiment model
            sentiment_model = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb")

            # ---------------------------------------------------------------------

            # 5. Training step --------------------------------------------

            if TRAIN:
                # We then build the sentiment analysis pipeline, passing the model name and the
                # sentiment analysis pipeline arguments. Let's also make sure to set the device
                # to the same device as the PPOTrainer.

                # reward_model = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb")

                # We then define the arguments to pass to the `generate` function. These arguments
                # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
                # the `generate` function of the trained model.
                generation_kwargs = {
                    # "min_length": -1,
                    "top_k": 0.0,
                    "top_p": 1.0,
                    "do_sample": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": 100_000
                }

                for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
                    question_tensors = batch["input_ids"]

                    response_tensors = ppo_trainer.generate(
                        question_tensors,
                        return_prompt=False,
                        length_sampler=output_length_sampler,
                        **generation_kwargs,
                    )
                    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

                    # Compute sentiment score
                    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
                    sentiment_outputs = sentiment_model(texts, **rw_kwargs)
                    # rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in
                    # reward_outputs]

                    messages_values = [output[1]["score"] for output in sentiment_outputs]

                    rewards = []
                    for message_value in messages_values:
                        opinion_shift_tot, num_activated_users, _ = information_diffusion_model.propagate_message(
                            message=message_value,
                            node_id=llm_node_id)
                        rewards.append(torch.tensor(num_activated_users, dtype=torch.float))

                    # Run PPO step
                    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
                    ppo_trainer.log_stats(stats, batch, rewards)

            # ---------------------------------------------------------------------

            # 6. Save step -------------------------------------------------------
            saving_path = script_args.output_dir + "llama2-sentiment-propagation"

            if not os.path.exists(saving_path):
                os.makedirs(saving_path)

            model.save_pretrained(saving_path)
            tokenizer.save_pretrained(saving_path)
            # ---------------------------------------------------------------------

            # 7. Reference model for results -------------------------------------
            device_ref_model = "cuda:0"
            ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(script_args.model_name,
                                                                          device_map=device_ref_model)

            if LOAD:
                device_model = "auto"

                model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    saving_path,
                    device_map=device_model
                )

                tokenizer = AutoTokenizer.from_pretrained(saving_path)

                tokenizer.pad_token = tokenizer.eos_token

                ppo_trainer = PPOTrainer(config, model, None, tokenizer)

            assert list(ref_model.parameters()) == list(model.parameters())

            # 8. Results dataset -------------------------------------------------------
            #### get a batch from the dataset

            #### 8.1 param setting
            bs = 16
            game_data = dict()
            dataset.set_format("pandas")
            df_batch = dataset[:].sample(bs)
            game_data["query"] = df_batch["query"].tolist()
            query_tensors = df_batch["input_ids"].tolist()
            gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
                          "pad_token_id": tokenizer.eos_token_id}

            #### 8.2 fill results tensors: ref, trained model
            response_tensors_ref = []
            for i in range(bs):
                gen_len = output_length_sampler()
                output = ref_model.generate(
                    torch.tensor(query_tensors[i], device=device_ref_model).unsqueeze(dim=0), max_new_tokens=gen_len,
                    **gen_kwargs
                ).squeeze()[-gen_len:]
                response_tensors_ref.append(output)

            response_tensors = []
            for i in range(bs):
                output = ppo_trainer.generate(
                    torch.tensor(query_tensors[i]), max_new_tokens=gen_len, **gen_kwargs
                ).squeeze()[-gen_len:]
                response_tensors.append(output)

            # ---------------------------------------------------------------------

            # 9. Results dataset -------------------------------------------------
            #### decode responses
            game_data["response (before)"] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
            game_data["response (after)"] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

            #### sentiment analysis of query/response pairs before/after
            texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
            #  game_data["rewards (before)"] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)] # [output for output in calc_rewards(texts)] #
            game_data_sentiments = [output[1]["score"] for output in sentiment_model(texts,
                                                                                     **rw_kwargs)]  # [output for output in calc_rewards(texts)] #
            game_data["rewards (before)"] = [information_diffusion_model.propagate_message(message=x,
                                                                                           node_id=llm_node_id)[1] for x
                                             in game_data_sentiments]  # [output for output in calc_rewards(texts)] #

            texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
            # game_data["rewards (after)"] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)] # [output for output in calc_rewards(texts)] #
            game_data_sentiments = [output[1]["score"] for output in sentiment_model(texts,
                                                                                     **rw_kwargs)]  # [output for output in calc_rewards(texts)] #
            game_data["rewards (after)"] = [information_diffusion_model.propagate_message(message=x,
                                                                                          node_id=llm_node_id)[1] for x
                                            in game_data_sentiments]

            # store results in a dataframe
            df_results = pd.DataFrame(game_data)

            print("mean:")
            # display(df_results[["rewards (before)", "rewards (after)"]].mean())
            print(df_results[["rewards (before)", "rewards (after)"]].mean())
            print()
            print("median:")
            # display(df_results[["rewards (before)", "rewards (after)"]].median())
            print(df_results[["rewards (before)", "rewards (after)"]].median())

            RESULTS[(modularity, homophily, position, llm_node_id)] = df_results.copy()
            del model
            del ref_model
            torch.cuda.empty_cache()
            gc.collect()

base_dir = pathlib.Path.cwd().parent
data_dir = base_dir / 'data' / 'output'
data_dir.mkdir(parents=True, exist_ok=True)
with open(data_dir / 'no_graph_results.pkl', 'wb') as file:
    pickle.dump(RESULTS, file)
