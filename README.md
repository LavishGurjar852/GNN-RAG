This github consist of GNN folder and the remaining are the parts of another folder LLM .
This is readme file for LLM.
The other file README_main.md will be the guiding file.

## Evaluation
We provide the results of GNN retrieval in `results/gnn`. To evaluate GNN-RAG performance, run `scripts/rag-reasoning.sh`. 
You can also compute perfromance on multi-hop question by `scripts/evaluate_multi_hop.sh`. 


## Resutls

We append all the results for Table 2: See `results/KGQA-GNN-RAG-RA`. You can look at the actual LLM generations, as well as the KG information retrieved ("input" key) in `predictions.jsonl`.
