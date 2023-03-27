# Extractive Question Answering
> A project by Ridhiman Dhindsa

### Description: 
This is an NLP model based on transformers, in which only the Encoder part is used,  
since it is required to process questions asked in natural language, and not generate new answers. 
Answers are generated based on documents from the website of the Election Commision of India. 

## Salient Features:
a.	EQA: Using transformer based encoder model. Compared various models. Found Roberta and TinyRoberta efficient for the task at hand.
TinyRoberta 50% smaller in size. Lesser compute. Comparable performance to Roberta. Better throughput.  
b.	Collected and formatted dataset of 3700 question answer pairs in SQUAD 2.0 format (Stanford Question Answering Dataset). 
Json file : having keys : context, question, answer  
c.	Took pre-trained tinyroberta model (it’s a pytorch model) from Huggingface. Fine-tuned for my dataset. 
Obtained Exact Match: 55.50 F1 Score: 75.54  
d.	Future plan: Improve accuracy by tuning the hyper-parameters. Improve throughput  & latency using tensorflow-tensorRT 
(TF-TRT: built on the NVIDIA CUDA® parallel programming model, enables you to optimize inference using techniques such as quantization, layer and tensor fusion, 
kernel tuning, and others on NVIDIA GPUs). It provides a simple API that delivers substantial performance gains on NVIDIA GPUs with minimal effort.  
e.	Deployment in triton server: Triton Inference Server, part of the NVIDIA AI platform, streamlines and standardizes AI inference 
by enabling teams to deploy, run, and scale trained AI models from any framework on any GPU- or CPU-based infrastructure.
