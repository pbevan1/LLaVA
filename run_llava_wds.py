# Required libraries
import argparse
import torch
import glob
import random
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import requests
from io import BytesIO
import os
import webdataset as wds
from webdataset import multi
from tqdm import tqdm
import uuid
import itertools
import numpy as np
import time
import wandb

# Function to load image from a given path or URL
def load_image(image_file):
    # Check if the image file is a URL
    if image_file.startswith('http') or image_file.startswith('https'):
        # Fetch image from URL
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        # Load image from local path
        image = Image.open(image_file).convert('RGB')
    return image

# Function to generate a caption for a given image
def get_caption(raw_image, caption_instruction, i):
    # Construct the input prompt using default tokens and instruction
    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + caption_instruction

    # Prepare the conversation format
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = raw_image

    # Preprocess the image to get its tensor representation
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    # Convert the prompt to input IDs and tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # Define the stopping criteria for the model
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # Generate caption using the model
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.3,
            max_new_tokens=200,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    # Check if there's any difference between input and output IDs
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    # Clean up the caption and return
    caption = outputs.replace("This image showcases ", "")
    return caption

# Function to get the dataset from specified tar files
def get_dataset():
    # Construct URLs to the tar files
    urls = [f'/fsx/home-peterbevan/LLaVA/pop/{i:05d}.tar' for i in [x for x in list(range(end_tar)) if x >= start_from_tar]]

    # Load dataset from these URLs using WebDataset
    dataset = wds.WebDataset(urls, handler=wds.ignore_and_continue)
    print(urls)
    return dataset


if __name__ == "__main__":

    wandb.init(project="LLaVA_1.5")
    
    # Disabling any initializations for Torch
    disable_torch_init()
    
    # Loading a pretrained model named "LLaVar"
    # model_name = "LLaVa"
    model_name = "llava-v1.5-13b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_name, None, model_name)

    number = 1

    for i in [1]:

        # Setting up environment for GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{number}"

        # Dividing the data into 8 segments, this is to decide the range of data we're processing in the loop
        # stepsize = int(520 / 8)

        # Define start and end points based on the current loop index and stepsize
        # start_from_tar = stepsize * number
        # end_tar = stepsize * (number+1)
        
        start_from_tar = 0
        end_tar = 1

        # Fetch dataset
        ds = get_dataset()
        print("dataset loaded")

        print(start_from_tar)
        print(end_tar)

        # Use multi-threading to load the dataset
        loader = multi.MultiLoader(ds, workers=8)

        # Define parameters for saving processed data into shard files
        image_key = "img"
        compression = False
        maxsize = 1e9
        maxcount = 100000
        shards = "./shards/"
        shard_prefix = f'laion-pop_{number}_'
        pattern = os.path.join(shards, shard_prefix + f"%05d.tar" + (".gz" if compression else ''))

        # Ensure directory exists
        os.makedirs(shards, exist_ok=True)

        # Process and save data into shards
        with wds.ShardWriter(pattern, maxsize=int(maxsize), maxcount=int(maxcount)) as sink:

            print(shard_prefix)
            i = 0
            for e in loader:
                
                s = time.time()

                # Convert image bytes into a PIL Image
                raw_image = Image.open(BytesIO(e['jpg'])).convert("RGB")
                
                # Instruction for captioning
                # caption_instruction = ('write a very detailed, factual caption, that describes the contents & composition of this image in details while avoiding any interpretations and staying to the facts. Describe which parts of the image are in focus and - if there are blurry parts - which are blurry. The whole text should have up to maximally 40 words. Start with the words "This image showcases":')
                # New instruction from cristoph
                caption_instruction = ('Can you please describe this image in up to two paragraphs? Please specify any objects within the image, backgrounds, scenery, interactions, and gestures or poses. If they are multiple of any object, please specify how many. Is there text in the image, and if so, what does it say? If there is any lighting in the image, can you identify where it is and what it looks like? What style is the image? If there are people or characters in the image, what emotions are they conveying? Please keep your descriptions factual and terse but complete. DO NOT add any unnecessary speculation about the things that are not part of the image such as "the image is inspiring to viewers" or "seeing this makes you feel joy". DO NOT add things such as "creates a unique and entertaining visual", as these descriptions are interpretations and not a part of the image itself. Start with the words "This image showcases":')

                # Get caption for the current image
                cap = get_caption(raw_image, caption_instruction, i)

                print(cap)

                # Form the sample data structure
                ds_key = "%09d" % i
                sample = {
                    "__key__": ds_key,
                    image_key: e['jpg'],
                    "alt": e['json'],
                    "original_txt": e["txt"]
                }

                # Write the sample into the shard
                sink.write(sample)

                i += 1
                print(time.time() - s)