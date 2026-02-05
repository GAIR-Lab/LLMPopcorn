import os
import csv
import re

def find_last_occurrence(lines, pattern):
    """
    Search backwards in the given list of lines for the first line matching the pattern.
    Returns the captured group text if found, otherwise returns an empty string.
    """
    for line in reversed(lines):
        match = re.match(pattern, line.strip())
        if match:
            return match.group(1)
    return ""  # Return an empty string if not found

def extract_prompts_from_file(file_path):
    """
    Search from the end of file for "title": "xxx", "cover_prompt": "xxx", "video_prompt": "xxx".
    - Case insensitive
    - Can contain multiple underscores between characters
    
    If the extracted title equals 
    "Unveiling the Legacy of Ancient Rome: Rise, Glory, and Downfall."
    then set all three values to empty strings.
    """
    # Use (?i) in regex for case insensitivity
    # For example, title -> t(?:_+)?i(?:_+)?t(?:_+)?l(?:_+)?e
    # This means there can be 0~n underscores between t and i (?:_+)? and similarly for others
    title_pattern = (
        r'^(?i)\s*"t(?:_+)?i(?:_+)?t(?:_+)?l(?:_+)?e"\s*:\s*"(.*)",?\s*$'
    )
    cover_pattern = (
        r'^(?i)\s*"c(?:_+)?o(?:_+)?v(?:_+)?e(?:_+)?r(?:_+)?p(?:_+)?r(?:_+)?o(?:_+)?m(?:_+)?p(?:_+)?t"\s*:\s*"(.*)",?\s*$'
    )
    video_pattern = (
        r'^(?i)\s*"v(?:_+)?i(?:_+)?d(?:_+)?e(?:_+)?o(?:_+)?p(?:_+)?r(?:_+)?o(?:_+)?m(?:_+)?p(?:_+)?t"\s*:\s*"(.*)",?\s*$'
    )

    # Read the text
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    # Search for the three items from the end
    title = find_last_occurrence(lines, title_pattern)
    cover_prompt = find_last_occurrence(lines, cover_pattern)
    video_prompt = find_last_occurrence(lines, video_pattern)

    # If the found title is the specified text, set all three to empty
    if title.strip() == "Unveiling the Legacy of Ancient Rome: Rise, Glory, and Downfall.":
        title = ""
        cover_prompt = ""
        video_prompt = ""

    return title, cover_prompt, video_prompt


def process_txt_files(input_folder, output_csv):
    """
    1. Traverse all .txt files in input_folder
    2. For each file, search backwards for the last occurrence of "title": "...", "cover_prompt": "...", "video_prompt": "..."
    3. Output CSV: user prompt, title, cover prompt, video prompt
    """
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["user prompt", "title", "cover prompt", "video prompt"])

        # Traverse all .txt files in the folder
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(".txt"):
                full_path = os.path.join(input_folder, filename)
                
                # Extract text corresponding to JSON keys
                title, cover_prompt, video_prompt = extract_prompts_from_file(full_path)

                # User prompt is the filename without the extension
                user_prompt = os.path.splitext(filename)[0]

                # Write a row
                writer.writerow([user_prompt, title, cover_prompt, video_prompt])


if __name__ == "__main__":
    # 1) Change to your txt folder path
    #creation_outputs_ai_concrete_rag_50_testset
    #baseline_concrete_outputs_2
    #creation_outputs_ai_concrete_rag_50_tags_4_testset
    #creation_rag_cot_prompt_ai_abstract_rag_50_testset_deepseek
    #creation_rag_cot_prompt_ai_concrete_rag_50_testset_deepseekr1
    #baseline_concrete_outputs_deepseekr1
    typs = ["concrete"] #"concrete",
    rags = [50] # 0,20,40,60,80,120,140
    for typ in typs:
        for rag in rags:
            input_folder_path = f"creation_rag_cot_prompt_ai_{typ}_rag_{rag}_tags_1_testset"
            
            # 2) Change output CSV path
            # output_prompt_baseline/prompt_baseline_abstract_2.csv
            # output_prompt_rag_more
            # output_prompt_baseline/prompt_baseline_concrete_gpt4o.csv
            # output_prompt_rag_more/prompt_ai_concrete_rag_50_testset_deepseekr1.csv
            output_csv_file = f"output_prompt_rag_more/prompt_ai_{typ}_rag_{rag}_tags_1_testset.csv"
        
            process_txt_files(input_folder_path, output_csv_file)
            print("Processing complete! Results written to:", output_csv_file)
