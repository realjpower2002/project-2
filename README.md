### TO RUN THESE MODELS ###
The models are configured to run using the checkpoint "model.pt" saved in each
directory. To run these, simply enter the directory, and run the python files
on a machine which has CUDA installed. 

They will produce responses to two prompts, "Which do you prefer? Dogs or cats?" 
and "How was your day this morning?", and then calculate perplexity and BLEU
score. To train these models, set the constant USE_CHECKPOINT in the code
file to "False" and the constant TRAIN to "True", and then run the file on
your own custom data, provided in the form of prompt/completion in jsonl format.

