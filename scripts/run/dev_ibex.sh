srun --pty --time=4:00:00 --gres=gpu:a100:4 --cpus-per-gpu=4 --mem=80G bash -l

python main_pos.py \
  --fname configs/dev_ibex.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3

python main_pos.py \
  --fname configs/dev_ibex_pos.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3

python main_pos.py \
  --fname configs/test.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3

python main_pos.py \
  --fname configs/test_new.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3

python main_pos.py \
  --fname configs/test_new_new.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3

python main_pos.py \
  --fname configs/test_twisted.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3