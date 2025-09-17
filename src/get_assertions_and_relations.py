import os
import json
import argparse
import textdistance

import pandas as pd

from unsloth import FastLanguageModel
from transformers import TextStreamer

from typing import List

# os.environ["GRAZIE_JWT_TOKEN"] = ""

from grazie.api.client.endpoints import GrazieApiGatewayUrls
from grazie.api.client.gateway import GrazieApiGatewayClient, GrazieAgent, AuthType
from grazie.api.client.profiles import Profile
from grazie.api.client.v8.chat.prompt import ChatPrompt
from grazie.api.client.v8.llm_parameters import LLMParameters
from grazie.api.client.v8.parameters import Parameters

from utils.relation_prompts import relation2demo

ASSERTION_PROMPT_FOR_GRAZIE_API_MODELS = "Your task is to segment the text into assertions, each standing as a separate claim."
RELATION_PROMPT_FOR_GRAZIE_API_MODELS = "Your task is to annotate claims with the relations from the Rhetorical Structure Theory."

model2profile = {"openai-o1": Profile.OPENAI_O_1, "openai-gpt-4o": Profile.OPENAI_GPT_4_O, "openai-chat-gpt": Profile.OPENAI_CHAT_GPT}

def generate_with_unsloth(prompt_text: str, model, tokenizer, hyperparams):
    messages = [
        {"role": "user", "content": prompt_text}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Must add for generation
        enable_thinking=False,  # Disable thinking
    )

    generated = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        **hyperparams,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )

    generated_text = tokenizer.batch_decode(generated)[0]
    if "<|im_start|>assistant" in generated_text:
        generated_text = generated_text.split("<|im_start|>assistant")[1].strip()
        generated_text = generated_text.split("<|im_end|>")[0].strip()
    if "</think>" in generated_text:
        generated_text = generated_text.split("</think>")[1]
    print(generated_text, type(generated_text))

    return generated_text.strip()


def generate_with_openai(prompt_text: str, prompt_for_grazie_api_models: str, model, profile):
    if profile.name == "openai-o1":
        parameters = {}
    else:
        parameters = {LLMParameters.Temperature: Parameters.FloatValue(0.0)}
    generated_text = model.v8.chat(
        ChatPrompt()
        .add_system(prompt_for_grazie_api_models)  # TODO: double-check this!
        .add_user(prompt_text),
        profile=profile,
        parameters=parameters
    ).content

    return generated_text.strip()

def extract_relations(relations, assertions):
    output_relations = []
    for relation in relations:
        arg1 = None
        arg2 = None
        for assertion_idx, assertion in enumerate(assertions):
            assertion = assertion.strip()
            if len(assertion)==0: # TODO: double check if this happens!
                continue
            if arg1 and arg2:
                output_relations.append({"relation": relation["relation"], "argument1": arg1, "argument2": arg2, "argument1_text": assertions[arg1], "argument2_text": assertions[arg2]})
                break
            if assertion == relation["argument1"]: #if textdistance.overlap(assertion, relation["argument1"])>0.99:
                arg1 = assertion_idx
            if assertion == relation["argument2"]: #textdistance.overlap(assertion, relation["argument2"])>0.99:
                arg2 = assertion_idx

    return output_relations


def generate_assertions_and_relations(task: str, input_data_path: str, output_data_path: str, model_name_or_path: str, debug: bool):
    output_annotations = []
    ids = []

    # Initializing the model and tokenizer (if used)
    if "unsloth" in model_name_or_path.lower():
        profile = None
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=4000,  # 4000, default 2048
            load_in_4bit=True,  # 4bit uses much less memory
            load_in_8bit=False,  # more accurate, uses 2x memory
            full_finetuning=False,  # inference-only mode
            # token = "hf_...",      # in case of gated models
        )
        hyperparams = {"max_new_tokens": 512, "temperature": 0.7, "top_p": 0.8,
                       "top_k": 20}  # non-thinking mode

    else:
        profile = model2profile[model_name_or_path]
        model = GrazieApiGatewayClient(
                    grazie_agent=GrazieAgent(name="grazie-ml-dev", version="dev"),
                    url=GrazieApiGatewayUrls.PRODUCTION, # PRODUCTION
                    auth_type=AuthType.USER, # USER
                )
        tokenizer = None
        hyperparams = dict()

    if task == "assertions":
        prompt_for_grazie_api_models = ASSERTION_PROMPT_FOR_GRAZIE_API_MODELS
    elif task == "relations":
        prompt_for_grazie_api_models = ASSERTION_PROMPT_FOR_GRAZIE_API_MODELS
    else:
        raise ValueError("Invalid task")

    # Reading the data
    df = pd.read_json(input_data_path, lines=True)
    if debug:
        df = df[:3]
    for record in df.iterrows():
        record = record[1]
        if task == "assertions":
            raw_inputs = record["abstracts"]
        elif task == "relations":
            raw_inputs = "\n".join(record["assertions"])
        else:
            raise ValueError("Invalid task")

        idx = record["ids"]

        if task == "assertions":
            prompt_text = f"Given the following text your task is to output all claims (assertions) that appear in this text. Each claim must be on a new line: {raw_inputs}\n"
        elif task == "relations":
            relation_definitions = ""
            for rel in relation2demo:
                relation_definitions += "Relation: " + rel + " " + relation2demo[rel] + "\n"
            prompt_text = f"Given the following claims your task is to output all possible relations between the claims. Relations can be: Evidence, Cause, Contrast, Condition, Background. Consider the following definitions: {relation_definitions}. Output JSON like in the example with one relation per line, you should identify all the relations present in the text: {{'relation': 'Relation', 'argument1':Argument1, 'argument2':Argument2}}\n {raw_inputs}"
        else:
            raise ValueError("Invalid task")

        if "unsloth" in model_name_or_path.lower():
            generated_text = generate_with_unsloth(prompt_text, model, tokenizer, hyperparams)
        else:
            generated_text = generate_with_openai(prompt_text, prompt_for_grazie_api_models, model, profile)

        ids.append(idx)

        if task == "assertions":
            assertions = generated_text.split("\n")
            processed_assertions = []
            for assertion in assertions:
                if assertion[0].isdigit() or assertion.startswith("-"):
                    if " " in assertion:
                        assertion = assertion[assertion.index(" ")+1:]
                processed_assertions.append(assertion.strip())
            output_annotations.append(processed_assertions)

        elif task == "relations":
            try:
                # OpenAI sometimes generates extra characters or misses brackets which results in failed parsing
                generated_text = generated_text.replace("```json", "")
                generated_text = generated_text.replace("```", "")
                if "[" in generated_text:
                    generated_text = generated_text[generated_text.index("["):]
                if "]" in generated_text:
                    generated_text = generated_text[:generated_text.rindex("]") + 1]
                generated_text = generated_text.replace("'", "\"")
                generated_text = generated_text.replace("}\n{", "},\n{")
                if not generated_text.startswith("["):
                    generated_text = "[ " + generated_text
                if not generated_text.endswith("]"):
                    generated_text = generated_text + "]"

                generated_text_as_json = json.loads(generated_text)
                extracted_relations = extract_relations(generated_text_as_json, record["assertions"])
                output_annotations.append(extracted_relations)
            except Exception as e:
                print(e)
                print(f"Could not parse to JSON: {generated_text}")

    if task == "assertions":
        assertions_with_ids = []
        for assertions, idx in zip(output_annotations, ids):
            assertions_with_ids.append({"ids": idx, "assertions": assertions})
        output_df = pd.DataFrame(assertions_with_ids)
        output_df.to_json(output_data_path, orient="records", lines=True) # orient="index", indent=2)
    elif task == "relations":
        relations_with_ids = []
        for relation, idx in zip(output_annotations, ids):
            relations_with_ids.append({"ids": idx, "relations": relation})
        output_df = pd.DataFrame(relations_with_ids)
        output_df.to_json(output_data_path, orient="records", lines=True) # orient="index", indent=2)

    print(f"Finished! The outputs are stored in {output_data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["assertions", "relations"], default="assertions")
    parser.add_argument("--model_name_or_path", type=str, default="openai-gpt-4o") # "unsloth/Qwen3-14B-unsloth-bnb-4bit"
    parser.add_argument("--input_data_path", type=str, default="data/selected_paper_abstracts.jsonl")
    parser.add_argument("--output_data_path", type=str, default="data/generated_assertions.jsonl")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    generate_assertions_and_relations(args.task, args.input_data_path, args.output_data_path, args.model_name_or_path, args.debug)
