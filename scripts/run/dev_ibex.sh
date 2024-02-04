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
  --fname configs/test_twisted_stronger.yaml \
  --devices cuda:0 cuda:1 cuda:2 cuda:3

srun --job-name=ijeapa_base --time=12:00:00 --gres=gpu:a100:4 --cpus-per-gpu=4 --mem=128G python main_pos.py --fname configs/ijepa_baseline.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3

srun --job-name=ijeapa_base --time=12:00:00 --gres=gpu:a100:4 --cpus-per-gpu=4 --mem=128G python main_lin.py --fname configs/ijepa_baseline.yaml --devices cuda:0 cuda:1 cuda:2 cuda:3