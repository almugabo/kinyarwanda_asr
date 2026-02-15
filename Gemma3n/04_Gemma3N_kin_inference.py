#!/usr/bin/env python
# coding: utf-8

# ## Inference for Gemma3N_Kin

# In[1]:


import os
import numpy as np
from vllm import LLM, SamplingParams
from vllm.multimodal.audio import AudioResampler
import soundfile as sf
import csv


# In[ ]:





# In[2]:


# Load Model 

MODEL_PATH = "/home/mike/unsloth_work/gemma_3n_kin_10000_epochs_3"

llm = LLM(
    model=MODEL_PATH,
    max_model_len=4096,
    max_num_seqs=1,
)


# In[ ]:


## Helper functions 
def load_audio_file(filepath: str):
    """ Load and return (audio_array, sampling_rate) """
    audio, sr = sf.read(filepath, dtype='float32')
    if len(audio.shape) > 1:
        # Mono conversion
        audio = np.mean(audio, axis=1)
    return audio, sr

def transcribe_audio(xaudio_array: np.ndarray, xsr: int):
    '''
    given an audio np array and sampling rate
    output a transcription 
    '''
    #first resample 
    # Gemma3N works with sampling rate of 1600
    TARGET_SR = 16000
    if xsr != TARGET_SR:
        xaudio_array = resampler.resample(xaudio_array, orig_sr=sr)
        
    # Sampling params for transcription
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
    )        

    prompt = (
        "<start_of_turn>user\n"
        "<audio_soft_token>"
        "transcribe this Kinyarwanda audio into text:\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            # Must be a tuple (audio_waveform, sample_rate)
            "audio": (xaudio_array.astype(np.float32), xsr)
        }
    }



    outputs = llm.generate(inputs, sampling_params)

    text = outputs[0].outputs[0].text
    
    return text 


# In[27]:


## test 

#xfile = '/media/mike/SSD4T/__staging/AI_Training_dset/audio_qwenasr/wavs/0yb6EzFR1y7aKNsQogml.wav'
xfile = '/media/mike/SSD4T/__staging/AI_Training_dset/audio_qwenasr/wavs/0yl9ZizCKe1P83rP11uR.wav'
q1, q2 = load_audio_file(xfile)

transcribe_audio(q1,q2)


# In[23]:





# ## test with 100 examples from validation dataset 

# In[61]:


import json 
import random 
import pandas as pd 


# In[59]:


xjson_file = '/media/mike/SSD4T/__staging/AI_Training_dset/audio_qwenasr/train.jsonl'

with open(xjson_file , 'r') as xff:
    xdata = [json.loads(line) for line in xff.readlines()]
    
xlist =  random.sample(xdata, 250)



# In[60]:


xlist_results = []

for q1 in xlist:
    xpath = q1['audio']
    xid = xpath.split('/')[-1:][0].replace('.wav', '')
    xtext_ref = q1['text'].split('<asr_text>')[1]    
    
    q1, q2 = load_audio_file(xpath)
    q3 = transcribe_audio(q1,q2)
    #result_dict 
    xdict = {}
    xdict['id'] = xid 
    xdict['text_ref'] = xtext_ref
    xdict['text_tra'] = q3  
    xlist_results.append(xdict)
    

t1 = pd.DataFrame(xlist_results)
    
print(t1.head())   


# In[62]:





# In[64]:


t1.to_excel('test_transcription.xlsx', index=False)
print('OK')


# In[ ]:




