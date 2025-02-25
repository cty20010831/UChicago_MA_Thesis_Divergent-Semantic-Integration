"""

All code in this file is licensed to John D. Patterson from The Pennsylvania State University, 04-04-2022, under the Creative Commons Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Link to License Deed https://creativecommons.org/licenses/by-nc-sa/4.0/

Link to Legal Code https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Please cite Johnson, D. R., Kaufman, J. C., Baker, B. S., Barbot, B., Green, A., van Hell, J., … Beaty, R. (2021, December 1). Extracting Creativity from Narratives using Distributional Semantic Modeling. Retrieved from psyarxiv.com/fmwgy in any publication or presentation

"""

# IMPORTS
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
import pandas as pd
import string
import time
import torch
import os
from pathlib import Path

# Define input and output paths
input_path = "../../data/processed"
output_path = "output"
os.makedirs(output_path, exist_ok=True)

# Define a helper function to process text with error handling for sentence segmentation
def process_text(text, segmenter):
    """Process text with error handling for sentence segmentation."""
    try:
        segmenter.train(text)
        return segmenter.tokenize(text)
    except (ValueError, ZeroDivisionError):
        # If training fails, use the default pretrained segmenter
        return segmenter.tokenize(text)

# INITIALIZE BERT AND TOKENIZERS
model = BertModel.from_pretrained("bert-large-uncased", output_hidden_states = True) # initialize BERT model instance
model.eval()
segmenter = PunktSentenceTokenizer() # initialize segmenter: does sentence segmentation, returns list
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased') # initialize BERT tokenizer
cos = torch.nn.CosineSimilarity(dim = 0)

# LOAD AND COMBINE DATA FROM ALL BATCHES
all_data = []
for batch_folder in Path(input_path).glob("batch_*"):
    batch_file = os.path.join(batch_folder, "processed_data.csv")

    # Read batch data
    batch_data = pd.read_csv(batch_file)
    all_data.append(batch_data)

# Combine all batches
d = pd.concat(all_data, ignore_index=True)
print("Finished concatenating dataframes of all batches.")

# CREATE STORAGE DICTIONARY
s = {}

# CREATE POST-EMBEDDING FILTERING LIST
filter_list = np.array(['[CLS]', '[PAD]', '[SEP]', '.', ',', '!', '?'])

# SEGMENT DATA INTO SENTENCES
start_time = time.time()
for index, row in d.iterrows():
    ID = row["worker_id"]
    text = row["narrative"]
    drawing_group = row["drawing_group"]
    
    # Create unique identifier for each narrative
    narrative_id = f"{ID}_{drawing_group}"
    
    # Skip if narrative is missing or empty
    if pd.isna(text) or text.strip() == "":
        s[narrative_id] = {"DSI": 0}  # Assign 0 for empty narratives
        continue
        
    s[narrative_id] = {}  # Initialize dictionary for this narrative

    try:
        # Clean the text first
        text = text.strip()
        if len(text) == 0:
            s[narrative_id]["DSI"] = 0
            continue
            
        # Get sentences using the helper function
        sentences = process_text(text, segmenter)
        
        if len(sentences) == 0:
            s[narrative_id]["DSI"] = 0
            continue
            
        # LOOP OVER SENTENCES AND GET BERT FEATURES (LAYERS 6 & 7)
        features = []  # initialize list to store dcos values, one for each sentence
        words = []
        for i in range(len(sentences)):  # loop over sentences
            sentence = sentences[i].translate(str.maketrans('','',string.punctuation))
            sent_tokens = tokenizer(sentence, max_length = 50, truncation = True, padding = 'max_length', return_tensors="pt")
            sent_words = [tokenizer.decode([k]) for k in sent_tokens['input_ids'][0]]
            sent_indices = np.where(np.in1d(sent_words, filter_list, invert = True))[0]  # we'll use this to filter out special tokens and punctuation
            with torch.no_grad():
                sent_output = model(**sent_tokens) # feed model the sentence tokens and get outputs
                hids = sent_output.hidden_states # isolate hidden layer activations
            layer6 = hids[6] # isolate layer 6 hidden activations
            layer7 = hids[7] # do the same for layer 7

            for j in sent_indices:  # loop over words and create list of all hidden vectors from layers 6 & 7; additionally store number of words (doubled, to account for layer 6 and 7 duplicates)
                words.append(sent_words[j])
                words.append(sent_words[j])
                features.append(layer6[0,j,:])  # layer 6 features
                features.append(layer7[0,j,:])  # layer 7 features

        # GET DCOS VALUES FOR STORY
        num_words = len(words) # number of words, in terms of hidden activation vectors (2*words)
        lower_triangle_indices = np.tril_indices_from(np.random.rand(num_words, num_words), k = -1)  # creates a matrix that represents words*2 (i.e., from word representations from both layer 6+7) and gets the indices of the lower triangle, omitting diagonal (k = -1)A
        story_dcos_vals = []  # intialize storage for dcos of current sentence
        for k in range(len(lower_triangle_indices[0])): # loop over lower triangle indices
            features1 = features[lower_triangle_indices[0][k]]
            features2 = features[lower_triangle_indices[1][k]]
            dcos = (1-cos(features1, features2))  # compute dcos
            story_dcos_vals.append(dcos) # store dcos value in list

        mean_story_dcos = torch.mean(torch.stack(story_dcos_vals)).item()
        s[narrative_id]["DSI"] = mean_story_dcos  # Store single DSI score for this narrative

        # Track processing status
        if (index + 1) % 50 == 0: 
            print(f'Finished calculating DSI for {index + 1} rows.')

    except Exception as e:
        print(f"Error processing narrative {narrative_id}: {str(e)}")
        s[narrative_id]["DSI"] = 0
        continue

# Save output
dsi_df = pd.DataFrame.from_dict(s, orient="index")

# Split the index and add the drawing_group prefix back
dsi_df["worker_id"] = dsi_df.index.str.split("_Incomplete_Group_").str[0]
dsi_df["drawing_group"] = "Incomplete_Group_" + dsi_df.index.str.split("_Incomplete_Group_").str[1]
dsi_df = dsi_df.reset_index(drop=True)[["worker_id", "drawing_group", "DSI"]]

# Save to CSV
output_file = os.path.join(output_path, "DSI_output.csv")
dsi_df.to_csv(output_file, index=False)

elapsed_time = time.time()-start_time
print('elapsed time: ' + str(elapsed_time))
print(f'Processed {len(s)} narratives')
print(f'Output saved to {output_file}')