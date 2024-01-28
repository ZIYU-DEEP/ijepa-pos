# I-JEPA + POS

```bash
python main_pos.py \
  --fname configs/dev.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 cuda:8
```

TODO
- write a scheduler for the pos loss. it drops quickly. we might want to find a good balance.