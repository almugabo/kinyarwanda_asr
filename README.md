# Kinyarwanda ASR
a repository with experimental scripts to finetune ASR models in Kinyarwanda


Automatic Speech Recognition (ASR)  or Speech To Text (STT) is the task of transcribing spoken language automatically.

ASR models for a given language are either

(a) trained from scratch

(b) created by finetuning existing ASR model in other languages


Here an overview of different efforts to create Kinyarwanda ASR models

### (a) from scratch

(1) NVIDIA ASR 

Kinyarwanda ASR using Mozilla Common Voice

https://docs.nvidia.com/nemo-framework/user-guide/25.09/nemotoolkit/asr/examples/kinyarwanda_asr.html

(2) KINSPEAK 

Antoine Nzeyimana. 2023. Kinspeak: Improving speech recognition for kinyarwanda via semi-supervised learning methods. arXiv preprint
([https://arxiv.org/html/2510.01145v1](https://arxiv.org/abs/2308.11863))


(3) META - Omnilingual ASR 

a suite of models providing aASR capabilities for more than 1,600 languages

([https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/])

https://github.com/facebookresearch/omnilingual-asr

paper: 
https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/




### (b) fine-tuned from the recent Digital Umuganda Leaderboard 

#### 1. w2v-bert-2.0-kinyarwanda-asr (first place Track A) 

Finetuned from: facebook/w2v-bert-2.0

[badrex/w2v-bert-2.0-kinyarwanda-asr](https://huggingface.co/badrex/w2v-bert-2.0-kinyarwanda-asr)

#### 2. Finetuned Whisper  (first place Track B&C)

used openai/whisper-large-v3 as a base model

https://github.com/SunbirdAI/kinyarwanda-asr-hackathon/blob/main/whisper_finetuning_kinyarwanda_hackathon.ipynb   

!!! data correction : *we noticed that many of the examples seemed to have the wrong label, i.e. the text transcription seemed to be for a different audio file.*



P.S:
- NVIDIA has several tools to inspect the data (not clear if maintained)

Speech data explorer

https://docs.nvidia.com/nemo-framework/user-guide/25.09/nemotoolkit/tools/speech_data_explorer.html

Comparison tool for ASR Models

https://docs.nvidia.com/nemo-framework/user-guide/25.09/nemotoolkit/tools/comparison_tool.html


See also a review 

Automatic Speech Recognition (ASR) for African Low-Resource Languages: A Systematic Literature Review
[ArXiv](https://arxiv.org/html/2510.01145v1)

