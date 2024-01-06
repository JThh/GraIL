# Hyperbolic Regularization: GraIL framework fused with translational KG embeddings

We will only introduce experiments in the transductive settings. For inductive experiments, please refer to the [GraIL repository](https://github.com/kkteru/grail/tree/master). 

The experiments can be accomplished in two steps: 

1. Pretrain translational KG Embeddings: use TransE, RotatE, ComplEx, RefH, and AttH.

Run commands:

```
chmod +x ./kge/run.sh && ./kge/run.sh train AttH 1710 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
```

2. Load KG model weights and run GraIL training steps.

Run commands:

```
python train.py -d 1710 -e grail_1710_v1 -fm 'AttH' --load_model False --gpu 0 # Optional: --use_kge_embeddings True --kge_model 'AttH'
```

To test GraIL run the following commands.

```
python test_auc.py -d 1710_transd -e grail_wn_v1
python test_ranking.py -d 1710_transd -e grail_wn_v1
```

The trained model and the logs are stored in experiments folder. Note that to ensure a fair comparison, we test all models on the same negative triplets. In order to do that in the current setup, we store the sampled negative triplets while evaluating GraIL and use these later to evaluate other baseline models (_consistent with the original GraIL setup_).

## Experiments

Below are the performance metrics on the test set of different model variations:

| Model          | MRR   | H@1   | H@3   | H@10  |
| -------------- | ----- | ----- | ----- | ----- |
| GraIL          | 0.261 | 0.186 | 0.288 | 0.399 |
| GraIL-TransE   | 0.256 | 0.185 | 0.277 | 0.397 |
| GraIL-ComplEx  | 0.262 | 0.195 | 0.284 | 0.394 |
| GraIL-RotatE   | 0.258 | 0.187 | 0.281 | 0.400 |
| GraIL-RefH     | 0.263 | 0.190 | 0.285 | 0.406 |
| **GraIL-AttH** | **0.271** | **0.197** | **0.293** | **0.413** |

## References

- For GraIL's approach and framework details, visit the [GraIL repository](https://github.com/kkteru/grail/tree/master).
- For knowledge graph embedding techniques, refer to [KGEmb by HazyResearch](https://github.com/HazyResearch/KGEmb).
- To understand the underlying theories, consult the papers "Inductive Relation Prediction by Subgraph Reasoning" ([arXiv link](https://arxiv.org/abs/1911.06962)) and "Low-Dimensional Hyperbolic Knowledge Graph Embeddings" ([arXiv link](https://arxiv.org/abs/2005.00545)).
