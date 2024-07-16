
## Evaluation
We provide the results of GNN retrieval in `results/gnn`. To evaluate GNN-RAG performance, run `scripts/rag-reasoning.sh`. 

You can also compute perfromance on multi-hop question by `scripts/evaluate_multi_hop.sh`. 

To test different LLMs for KGQA (ChatGPT, LLaMA2), see `scripts/plug-and-play.sh`. 

## Resutls

We append all the results for Table 2: See `results/KGQA-GNN-RAG-RA`. You can look at the actual LLM generations, as well as the KG information retrieved ("input" key) in `predictions.jsonl`.
