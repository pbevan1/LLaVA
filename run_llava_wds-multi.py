import argparse
import torch
import glob
import random
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import time
from PIL import Image

import requests
from PIL import Image
from io import BytesIO

import time
import os
import io
from PIL import Image
import webdataset as wds
from webdataset import multi
from tqdm import tqdm 
from PIL import Image
import time
import uuid
import itertools
import numpy as np

import multiprocessing

import wandb
import GPUtil
import json
import signal


def get_gpu_status():
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print(f"GPU {gpu.id}: {gpu.load*100:.1f}% load, {gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f}MB used")


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def get_caption(raw_image ,caption_instruction, i, tokenizer, model, image_processor):
    try:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + caption_instruction
        
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        image = raw_image

        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.3, #- random.random()/10,
                max_new_tokens=200,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        #print("Caption:", outputs)
        caption = outputs.replace("This image showcases ","")
    except Exception as e:
        print(e)

    return caption


def worker(number):
  print(f"Worker {number} started.")  # Indicates the start of the worker.

  # Load model
  print(f"Worker {number}: Loading model...")
  disable_torch_init()

  model_name = "llava-v1.5-13b"
  tokenizer, model, image_processor, context_len = load_pretrained_model(model_name, None, model_name)
  print(f"Worker {number}: Model loaded.")  # Indicates successful model loading.

#   stepsize = int( 520000 / 8  )
  total_files = 521
  num_processes = 8
  stepsize = total_files // num_processes  # This will give 65

  start_from_tar = stepsize * number
  end_tar = stepsize * (number+1)
  print(f"Worker {number}: Processing files from {start_from_tar} to {end_tar}.")  # Indicates the range of files being processed.

  def get_dataset():
    print(f"Worker {number}: Generating dataset URLs...")  # Before generating URLs
    urls = [f'/fsx/llaver/LLaVA/pop/{i:05d}.tar' for i in [x for x in list(range(end_tar)) if x >= start_from_tar]] # 231350
    print(f"Worker {number}: Loading dataset...")  # Before loading the dataset
    #urls2 = [f'pipe:aws s3 cp {url} -' for url in urls]
    dataset = wds.WebDataset(urls, handler=wds.ignore_and_continue)
    # print(urls)
    return dataset

  ds = get_dataset()
  print(f"Worker {number}: Dataset loaded.")  # Indicates successful dataset loading.

  print("dataset loaded")
  loader = multi.MultiLoader(ds, workers=1)  # Workers=1 seems to solve deadlock issue and no decrease in throughput.
  print(f"Worker {number}: Data loader initialized.")  # Indicates successful loader initialization.

  image_key = "img"
  compression = False
  maxsize=1e9
  maxcount=100000
  shards = "./shards/" 
  shard_prefix = f'laion-pop_w{number}_'
  start_index = read_checkpoint(number)
  start_index_padded = f"n{start_index:05}_"
  pattern = os.path.join(shards, shard_prefix + start_index_padded + f"%05d.tar" + (".gz" if compression else ''))

  os.makedirs(shards, exist_ok=True)

  with wds.ShardWriter(pattern, maxsize=int(maxsize), maxcount=int(maxcount)) as sink:
    print(f"Worker {number}: Entering processing loop...")  # Before entering the loop for image processing

    # i=0
    # Loading image number from checkpoint
    exception_count = 0
    while exception_count < 200:
        for i, e in enumerate(tqdm(loader, position=number, desc=f"GPU{number}")):
            if i < start_index:
                continue  # Skip images that have already been processed

            s = time.time()
            
            try:
                #print(type(e))
                #print(e['jpg'])
                #print(e["txt"])
                # Set an alarm for 30 seconds from now
                signal.alarm(30)
                raw_image = Image.open(io.BytesIO(e['jpg']) ).convert("RGB") 
                #print(type(raw_image))
                caption_instruction = ('Can you please describe this image in up to two paragraphs? Please specify any objects within the image, backgrounds, scenery, interactions, and gestures or poses. If they are multiple of any object, please specify how many. Is there text in the image, and if so, what does it say? If there is any lighting in the image, can you identify where it is and what it looks like? What style is the image? If there are people or characters in the image, what emotions are they conveying? Please keep your descriptions factual and terse but complete. DO NOT add any unnecessary speculation about the things that are not part of the image such as "the image is inspiring to viewers" or "seeing this makes you feel joy". DO NOT add things such as "creates a unique and entertaining visual", as these descriptions are interpretations and not a part of the image itself. The description should be purely factual, with no subjective speculation. Make sure to include the style of the image, for example cartoon, photograph, 3d render etc. Start with the words "This image showcases":')

                cap = get_caption(raw_image, caption_instruction, i, tokenizer, model, image_processor)
                # print(f"Worker {number} caption generated: {cap}")

                # Reset the alarm
                signal.alarm(0)

                ds_key = "%09d" % i

                sample = {
                            "__key__": ds_key,
                            image_key: e['jpg'],
                            #"img_url": e['url'],
                            "alt": e['json'],
                            "original_txt": e["txt"],
                            "llava_1_5_caption": cap
                        }
                sink.write(sample)
            
                # print(f"Worker {number}: Processed image {i} in {time.time()-s} seconds.")  # Indicates each image processing time

            except Exception as e:
                exception_count+=1
                print("Error creating caption")
                print(e)

            update_checkpoint(number, i)


def start_worker_on_specific_gpu(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    worker(gpu_id)


def read_checkpoint(worker_number):
    checkpoint_path = f"captioning_checkpoints/checkpoint_{worker_number}.json"
    try:
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        return checkpoint.get("index", 0)
    except FileNotFoundError:
        return 0


def update_checkpoint(worker_number, index):
    checkpoint_path = f"captioning_checkpoints/checkpoint_{worker_number}.json"
    os.makedirs("captioning_checkpoints", exist_ok=True)

    checkpoint = {"index": index}
    
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f)


# Function to read API key from a file
def read_api_key(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('WANDB_API_KEY='):
                return line.strip().split('=')[1]
    raise ValueError(f'No API key found in {file_path}')


if __name__ == "__main__":

    torch.cuda.device_count()
    torch.cuda.empty_cache()

    # Initialise wanb project
    file_path = 'wandb.env'
    api_key = read_api_key(file_path)
    # Log in to wandb
    wandb.login(host='https://stability.wandb.io', relogin=True, key=api_key)
    wandb.init(project="LLaVA_1.5")

    multiprocessing.set_start_method('spawn', force=True)

    # Anzahl der Prozesse
    num_processes = 8

    processes = []

    # Start the processes
    for n in range(num_processes):
            process = multiprocessing.Process(target=start_worker_on_specific_gpu, args=(n,))
            processes.append(process)
            process.start()

    # Wait for all processes to finish
    for process in processes:
            get_gpu_status()
            process.join()

    print("All processes have finished")