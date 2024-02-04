# I-JEPA + POS

```bash
python main_pos.py \
  --fname configs/dev.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 cuda:8
```

<!-- ## TODO -->
<!-- - write a scheduler for the pos loss. it drops quickly. we might want to find a good balance.
  - we may need to set a high pos lambda at the beginning, and gradually decay?
- write the formula on overleaf.
- double check the probe acc, and the pos loss part. -->
<!-- - update the logging part (auto-configure the write tag, and the w&b). -->
<!-- - update the linear probing part. -->
<!-- - update the auto port finding. -->
<!-- Just a note: instead of predicting the positions for only the dropped ones, let the model to predict them all can help smooth the learning. -->

## Current Design
For the context, we partially drop some of its positional embeddings (replaced by learnable mask tokens), then feed it into the encoder. 

The encoded feature will be fed into two different predictors. One is the original I-JEPA predictor, and the other is the position predictor (where we only care about correctly predict the position for patches whose position embeddings are dropped).

Notice that in this way we make the ijepa objective per se harder as well, due to the additional positional embedding removal on the context.