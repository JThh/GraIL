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