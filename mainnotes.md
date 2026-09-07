# Division of experiments for the paper


The following repo contains the implementation of the obfuscated activation experiments 

Order of execution of the experiment 

1. Ordinary Attacks (§3.2) – Detecting standard jailbreaks with latent monitors

File(s): inference_time_experiments/train_harmful.py

(The script trains the probes and evaluates them on the jailbreak datasets; the obf_reps package contains the evaluation logic.)


Code execution flow for the experiment 

- execution starts from the main() inside the train_harmful.py 




main()

- it relies on the following 

* pickle for working with pkl files 
* time for time management 
* pathlib for os path related stuff
* hydra for sort of config manupulation and saving, running the command line arguments 
* usual stuff like numpy, torch, matplotlib, tqdm 
* using the dataloader from pytorch as well 

It relies on the following files and functions 

- we are getting the functions 


* evaluate_metric
* evaluate_model 
* evaluate_model_and_metric_harmful 

functions from evaluate 

evaluate model takes model, loss function, logger and generation length  and both the types of data modules as inputs and 








Flow:

Load datasets: benign (UltraChat, XSTest) and harmful 

(Circuit Breakers) – split into train/val/test.

Train monitors on benign vs. non‑jailbroken harmful 

examples:

Logistic Regression, MLP, Mean‑Difference probes, Gaussian OOD detector.

Both input‑based (last token) and generation‑based (all generated tokens).

Generate jailbreak datasets: For each harmful query, apply each jailbreak method (GCG, PAIR, …) to elicit a response. Keep only successful jailbreaks (StrongREJECT >0.5).


Evaluate monitors on the jailbreak test sets: compute recall at 1% FPR (threshold derived from benign scores). 


Save results 

Log metrics to the output directory.







