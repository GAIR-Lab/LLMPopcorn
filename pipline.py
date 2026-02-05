import argparse
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import gc
import openai
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ==============================
# Part 1: Generate RAG Prompt and Save to File
# ==============================
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def generate_rag_prompt(user_prompt, 
                        output_dir, 
                        total_videos=1974, 
                        selected_videos_num=50, 
                        num_tags=1, 
                        ratio=0.1,  # ratio represents the proportion of negative samples
                        seed=42):
    # Data Loading
    set_random_seed(seed)

    # 2. Load various data files
    views_data = pd.read_csv('Microlens/MicroLens-100k_likes_and_views.txt',  
                             sep='\t', header=None, names=['video_id','likes','views'])
    title_data = pd.read_csv('Microlens/MicroLens-100k_title_en.csv', 
                             sep=',', header=None, names=['video_id','title_en'])
    cover_data = pd.read_csv('Microlens/llava-v1.5_caption.txt', 
                             sep=',', header=None, names=['video_id','cover_desc'])
    desc_data = pd.read_csv('Microlens/Microlens100K_captions_en.csv', 
                            sep='\t', header=None, names=['video_id','caption_en'])
    tags_data = pd.read_csv('Microlens/tags_to_summary.csv',
                            sep=',', header=None, names=['video_id','partition'])
    
    # 3. Load comment data and count comments for each video_id
    comments_data = pd.read_csv('Microlens/MicroLens-100k_comment_en.txt',
                                sep='\t', header=None, names=['user_id','video_id','comment_text'])
    comments_data = comments_data[['video_id','comment_text']]

    # Group by video_id and count comments
    comment_count_df = (
        comments_data
        .groupby('video_id')['comment_text']
        .count()
        .reset_index(name='comment_count')
    )

    # Merge all data
    merged = (
        views_data
        .merge(title_data, on='video_id', how='left')
        .merge(cover_data, on='video_id', how='left')
        .merge(desc_data, on='video_id', how='left')
        .merge(tags_data, on='video_id', how='left')
        .merge(comment_count_df, on='video_id', how='left')
    )

    # Load test set IDs
    test_id_data = pd.read_csv('/MicroLens/test_id.csv', 
                               sep=',', header=None, names=['video_id'])

    # Perform inner join with the merged dataframe on 'video_id'
    merged = merged.merge(test_id_data, on='video_id', how='inner')
    
    # Drop rows with missing values in key fields
    merged.dropna(subset=['title_en', 'cover_desc', 'caption_en', 'partition', 'comment_count'], inplace=True)

    # 4. Use SentenceTransformer to create embeddings for 'partition' and user prompts
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    unique_partitions = merged['partition'].unique().tolist()
    partition_embeddings = model.encode(unique_partitions)
    prompt_embedding = model.encode([user_prompt])[0]

    # Calculate similarities and get top partitions
    similarities = [
        np.dot(prompt_embedding, pe) 
        / (np.linalg.norm(prompt_embedding) * np.linalg.norm(pe)) 
        for pe in partition_embeddings
    ]
    top_k_indices = np.argsort(similarities)[::-1][:num_tags]
    top_k_partitions = [unique_partitions[i] for i in top_k_indices]

    # Filter data by top partitions
    filtered_data = merged[merged['partition'].isin(top_k_partitions)]

    # Sort by comment count
    filtered_data = filtered_data.sort_values('comment_count', ascending=False)

    # Split dataset based on ratio (proportion of negative samples)
    n_negative = int(len(filtered_data) * ratio)
    n_positive = len(filtered_data) - n_negative

    # Select popular videos (high comment count)
    positive_videos = filtered_data.head(n_positive)

    # Select unpopular videos from remaining data
    remaining_data = filtered_data.iloc[n_positive:]
    negative_videos = remaining_data.tail(n_negative)

    # Remove duplicates to prevent overlap
    positive_videos.drop_duplicates(subset=['video_id'], inplace=True)
    negative_videos.drop_duplicates(subset=['video_id'], inplace=True)

    # Merge positive and negative samples for retrieval
    combined_videos = pd.concat([positive_videos, negative_videos])
    combined_videos.drop_duplicates(subset=['video_id'], keep='first', inplace=True)

    # Create combined text for embedding
    combined_texts = (
        combined_videos['title_en'] + " " 
        + combined_videos['cover_desc'] + " " 
        + combined_videos['caption_en']
    ).tolist()
    combined_embeddings = model.encode(combined_texts)

    # Perform similarity search
    index = faiss.IndexFlatL2(combined_embeddings.shape[1])
    index.add(np.array(combined_embeddings).astype('float32'))
    query_embedding = model.encode([user_prompt]).astype('float32')
    _, I = index.search(query_embedding, len(combined_videos))
    retrieved_videos = combined_videos.iloc[I[0]]

    # Calculate final sample sizes
    final_n_negative = int(selected_videos_num * ratio)
    final_n_positive = selected_videos_num - final_n_negative

    # Find positive and negative samples from retrieval results
    positive_ids = set(positive_videos['video_id'].tolist())
    negative_ids = set(negative_videos['video_id'].tolist())

    retrieved_positive = retrieved_videos[retrieved_videos['video_id'].isin(positive_ids)]
    retrieved_negative = retrieved_videos[retrieved_videos['video_id'].isin(negative_ids)]

    # Select final samples
    final_selected_positive = retrieved_positive.head(final_n_positive)
    final_selected_negative = retrieved_negative.head(final_n_negative)

    # Maintain retrieval order
    final_selected_videos = pd.concat([final_selected_positive, final_selected_negative])
    final_selected_videos = final_selected_videos.loc[
        retrieved_videos.index.intersection(final_selected_videos.index)
    ]
    # Build output text
    rag_positive_context = "\n".join([
        f"Reference Video {i+1} (Positive Sample - Popular):\n"
        f"Title: {row['title_en']}\n"
        f"Cover Desc: {row['cover_desc']}\n"
        f"Desc: {row['caption_en']}\n"
        f"Comment Count: {row['comment_count']}\n"
        for i, (idx, row) in enumerate(final_selected_positive.iterrows())
    ])

    rag_negative_context = "\n".join([
        f"Reference Video {i+1} (Negative Sample - Unpopular):\n"
        f"Title: {row['title_en']}\n"
        f"Cover Desc: {row['cover_desc']}\n"
        f"Desc: {row['caption_en']}\n"
        f"Comment Count: {row['comment_count']}\n"
        for i, (idx, row) in enumerate(final_selected_negative.iterrows())
    ])

    rag_context = rag_positive_context + "\n" + rag_negative_context
    
    cot_prompt = f"""
    Now that you're a talented video creator with a wealth of ideas, you need to think from the user's perspective and after that generate the most popular video title, an AI-generated cover prompt, and a 3-second AI-generated video prompt.
    
    Below is the user query:
    
    {user_prompt}
    
    Below is the reasoning chain (Chain of Thought) that you should follow step-by-step.
    
    Reasoning Chain:
    1. Analyze both popular and unpopular videos as references using the provided context:
       {rag_context}   
    2. Based on the analyzed videos, brainstorm unique and creative ideas for a new video topic. Ensure that the idea is original and does not replicate existing content, as direct copying is strictly forbidden.
    3. Write or conceptualize a logical and original script or content based on a Step2 topic
    4. Double-check:
       1. Whether the theme and content are accurately conveyed
       2. Whether the theme and content are strongly related and complete in fulfilling the user's needs
    5. Start generating based on confirmed topics and content
    6. Re-evaluate:
       1. Whether the generated prompt is logically correct.
       2. Check if the final suggestions match popular trends
    7. If it doesn't meet expectations, refine and finalize the output.  
    
    Explicitly generate a chain of thought during the reasoning process for each candidate. The chain of thought should detail the steps, considerations, and rationale behind the candidate generation, ensuring transparency and clarity in the decision-making process.
    
    Final Answer Requirements:
    - A single line for the final generated Title (MAX_length = 50).
    - A single paragraph for the Cover Prompt.
    - A single paragraph for the Video Prompt (3-second).
    
    
    Now, based on the above reasoning, generate the response in JSON format， here is an example:
    {{
      "title": "Unveiling the Legacy of Ancient Rome: Rise, Glory, and Downfall.",
      "cover_prompt": "Generate an image of a Roman Emperor standing proudly in front of the Colosseum, with a subtle sunset backdrop, highlighting the contrast between the ancient structure.",
      "video_prompt": "Open with a 3-second aerial shot of the Roman Forum, showcasing the sprawling ancient ruins against a clear blue sky, before zooming in on a singular, imposing structure like the Colosseum."
    }}
    Please provide your answer following this exact JSON template for the response.
    """
    
    # Save prompt to file
    output_file = os.path.join(output_dir, f"{user_prompt}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cot_prompt)
    
    print(f"RAG prompt saved to {output_file}")



# ==============================
# Part 2: Load RAG Prompts and Perform Inference
# ==============================
def inference_from_prompts(input_dir, output_dir,seed=42):
    set_random_seed(seed)
    llama_model_name = "meta-llama/Llama-3.3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model_llama = AutoModelForCausalLM.from_pretrained(
        llama_model_name, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config
    )

    llama_pipeline = pipeline(
        "text-generation",
        model=model_llama,
        tokenizer=tokenizer,
        max_new_tokens=5000,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    for file_name in tqdm(os.listdir(input_dir), desc="inferencing"):
        output_path = os.path.join(output_dir, file_name)
        if os.path.exists(output_path):
            continue
            
        if file_name.endswith(".txt"):
            # 读取输入文件内容
            input_path = os.path.join(input_dir, file_name)
            with open(input_path, "r", encoding="utf-8") as f:
                cot_prompt = f.read()
    
            # 定义 system 和 user 的内容
            messages = [
                {"role": "system", "content": "Now that you're a talented video creator with a wealth of ideas, you need to think from the user's perspective and after that generate the most popular video title, an AI-generated cover prompt, and a 3-second AI-generated video prompt."},
                {"role": "user", "content": cot_prompt},
            ]
            
    
            # 调用模型推理
            response = llama_pipeline(messages, num_return_sequences=1)
            full_output = response[0]['generated_text']
    
            # 保存输出到指定路径
            output_path = os.path.join(output_dir, file_name)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_output[2]['content'])
    
            print(f"Inference result saved to {output_path}")

def inference_from_prompts_qwen(input_dir, output_dir, seed=42):
    # Set random seed for reproducibility
    set_random_seed(seed)
    
    # Qwen model name
    qwen_model_name = "Qwen/Qwen2.5-72B-Instruct"
    
    # Load tokenizer with trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
    )
    
    # Load Qwen model with auto device mapping and quantization config
    model_qwen = AutoModelForCausalLM.from_pretrained(
        qwen_model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    # Create text generation pipeline with consistent parameters
    qwen_pipeline = pipeline(
        "text-generation",
        model=model_qwen,
        tokenizer=tokenizer,
        max_new_tokens=5000,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    # Process all txt files in input directory
    for file_name in tqdm(os.listdir(input_dir), desc="inferencing"):
        if file_name.endswith(".txt"):
            # Read input file content
            input_path = os.path.join(input_dir, file_name)
            with open(input_path, "r", encoding="utf-8") as f:
                cot_prompt = f.read()
            
            # Define system and user messages
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "Now that you're a talented video creator with a wealth of ideas, "
                        "you need to think from the user's perspective and after that generate "
                        "the most popular video title, an AI-generated cover prompt, and a 3-second "
                        "AI-generated video prompt."
                    )
                },
                {"role": "user", "content": cot_prompt},
            ]
            
            # Call model for inference
            response = qwen_pipeline(messages, num_return_sequences=1)
            full_output = response[0]['generated_text']
            
            # Save output to specified path
            output_path = os.path.join(output_dir, file_name)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_output[2]['content'])
            
            print(f"Inference result saved to {output_path}")

def inference_from_prompts_mistral(input_dir, output_dir, seed=42):
    # Set random seed for reproducibility
    set_random_seed(seed)
    
    # Mistral model name
    mistral_model_name = "mistralai/Mistral-Large-Instruct-2411"
    
    # Load tokenizer with trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(mistral_model_name, trust_remote_code=True)
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
    )
    
    # Load Mistral model with auto device mapping and quantization config
    model_mistral = AutoModelForCausalLM.from_pretrained(
        mistral_model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    # Create text generation pipeline with consistent parameters
    mistral_pipeline = pipeline(
        "text-generation",
        model=model_mistral,
        tokenizer=tokenizer,
        max_new_tokens=5000,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    # Process all txt files in input directory
    for file_name in tqdm(os.listdir(input_dir), desc="inferencing"):
        output_path = os.path.join(output_dir, file_name)
        if os.path.exists(output_path):
            continue
            
        if file_name.endswith(".txt"):
            # Read input file content
            input_path = os.path.join(input_dir, file_name)
            with open(input_path, "r", encoding="utf-8") as f:
                cot_prompt = f.read()
            
            # Define system and user messages
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "Now that you're a talented video creator with a wealth of ideas, "
                        "you need to think from the user's perspective and after that generate "
                        "the most popular video title, an AI-generated cover prompt, and a 3-second "
                        "AI-generated video prompt."
                    )
                },
                {"role": "user", "content": cot_prompt},
            ]
            
            # Call model for inference
            response = mistral_pipeline(messages, num_return_sequences=1)
            full_output = response[0]['generated_text']
            
            # Save output to specified path
            output_path = os.path.join(output_dir, file_name)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_output[2]['content'])
            
            print(f"Inference result saved to {output_path}")

import os
import time
from tqdm import tqdm

def inference_from_prompts_api(input_dir, output_dir, api_key, model="deepseek-reasoner", base_url="https://api.deepseek.com"):
    # Recommended to use 'import openai' directly. Adjust based on your actual library name
    from openai import OpenAI 
    print(model)
    
    # Initialize DeepSeek client
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Iterate through files in input directory
    for file_name in tqdm(os.listdir(input_dir), desc="Inferencing with DeepSeek"):
        output_path = os.path.join(output_dir, file_name)
        if file_name.endswith(".txt") and not os.path.exists(output_path):
            # Read input file content
            input_path = os.path.join(input_dir, file_name)
            with open(input_path, "r", encoding="utf-8") as f:
                cot_prompt = f.read()

            # Define system and user content
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "Now that you're a talented video creator with a wealth of ideas, "
                        "you need to think from the user's perspective and after that generate "
                        "the most popular video title, an AI-generated cover prompt, and a 3-second "
                        "AI-generated video prompt."
                    )
                },
                {"role": "user", "content": cot_prompt},
            ]
            
            # Retry mechanism
            while True:
                try:
                    # Call DeepSeek API for inference
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages
                    )
                    # Break loop if successful
                    break
                except Exception as e:
                    # Print error message and wait 60 seconds before retrying
                    print(f"Error occurred: {e}")
                    print("Waiting 60 seconds before retrying...")
                    time.sleep(120)

            full_output = response.choices[0].message.content

            output_path = os.path.join(output_dir, file_name)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_output)

            print(f"Inference result saved to {output_path}")



def inference_from_prompts_gpt(input_dir, output_dir, api_key, model="gpt-4o"):
    """
    Use OpenAI GPT API to process input prompts and generate outputs.

    Parameters:
        input_dir (str): Directory containing input text files with prompts.
        output_dir (str): Directory to save the generated outputs.
        api_key (str): OpenAI API key for authentication.
        model (str): The GPT model to use (default: "gpt-4").
    """
    # Set the OpenAI API key
    openai.api_key = api_key

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through files in the input directory
    for file_name in tqdm(os.listdir(input_dir), desc="Inferencing with GPT"):
    
        if file_name.endswith(".txt"):
            # Read input file content
            input_path = os.path.join(input_dir, file_name)
            with open(input_path, "r", encoding="utf-8") as f:
                cot_prompt = f.read()

            # Define the system and user prompts
            messages = [
                {"role": "system", "content": "You are a talented video creator with a wealth of ideas. Think from the user's perspective and generate the most popular video title, an AI-generated cover prompt, and a 3-second AI-generated video prompt."},
                {"role": "user", "content": cot_prompt},
            ]

            # Call the GPT API for inference
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    stream=False
                )
                # Extract the generated content
                full_output = response['choices'][0]['message']['content']

                # Save the output to the specified path
                output_path = os.path.join(output_dir, file_name)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(full_output)

                print(f"Inference result saved to {output_path}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--USER_PROMPT", type=str, help="The user prompt to process.")
    parser.add_argument("--MODE", type=str, choices=["generate", "infer","infer_api","infer_gpt","infer_qwen","infer_mistral"], required=True, help="Mode: generate or infer.")
    parser.add_argument("--INPUT_DIR", type=str, help="Input directory for inference.")
    parser.add_argument("--OUTPUT_DIR", type=str, required=True, help="Output directory.")
    parser.add_argument("--MODEL", type=str, required=True, help="Model.")
    parser.add_argument("--VID_NUM", type=int, default=10,help="number of selected videos")
    parser.add_argument("--TAGS_NUM", type=int, default=1,help="number of selected videos")
    args = parser.parse_args()

    if args.MODE == "generate":
        os.makedirs(args.OUTPUT_DIR, exist_ok=True)
        generate_rag_prompt(args.USER_PROMPT, args.OUTPUT_DIR, selected_videos_num=args.VID_NUM,num_tags=args.TAGS_NUM)
    elif args.MODE == "infer":
        os.makedirs(args.OUTPUT_DIR, exist_ok=True)
        inference_from_prompts(args.INPUT_DIR, args.OUTPUT_DIR)
    elif args.MODE == "infer_qwen":
        os.makedirs(args.OUTPUT_DIR, exist_ok=True)
        inference_from_prompts_qwen(args.INPUT_DIR, args.OUTPUT_DIR)
    elif args.MODE == "infer_mistral":
        os.makedirs(args.OUTPUT_DIR, exist_ok=True)
        inference_from_prompts_mistral(args.INPUT_DIR, args.OUTPUT_DIR)
    elif args.MODE == "infer_api":
        os.makedirs(args.OUTPUT_DIR, exist_ok=True)
        inference_from_prompts_api(args.INPUT_DIR, args.OUTPUT_DIR,"API_KEY",model=args.MODEL)
    elif args.MODE == "infer_gpt":
        api_key = "API_KEY"
        os.makedirs(args.OUTPUT_DIR, exist_ok=True)
        inference_from_prompts_gpt(args.INPUT_DIR, args.OUTPUT_DIR,api_key,model="gpt-4o")
    import torch
    
    gc.collect()
    torch.cuda.empty_cache()