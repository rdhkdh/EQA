# Extractive Question Answering
> A project by Ridhiman Dhindsa

### Description:
This is an NLP model based on transformers for extractive question answering.
The model has been fine-tuned on dataset gathered form PDFs available on various public websites.

## Salient Features:
1) I have used a transformer based encoder model. I selected the model after comparing various
models available for NLP. Following criteria was used for selection of model:    
  * Model file size should be small so that it can be accommodated on local machine for fine-tuning.    
  * The number of parameters should not be too large for fine-tuning the model on a local machine with or
without GPU.  
  * It should not require too much compute.  
  * The accuracy should be comparable to the alternative models available in the range.  
  * Model should have good throughput.  
2) Collected and formatted dataset of 3700 question answer pairs in SQUAD 2.0 format (Stanford
Question Answering Dataset).  
3) I have taken pre-trained model from Hugging Face. The model was fine-tuned for custom
dataset.
Results obtained till now are: Exact Match: 55.50 F1 Score: 75.54
4) Future plan:  
I intend to Improve accuracy by tuning the hyper-parameters. I am also aiming at improvement of
throughput and latency using tensorflow-tensorRT.  
5) Subsequent to above optimizations, I will deploy the model in Triton Inference Server.
