arg1=$1
data=${2:-UrbanWinds2DDataModule}
arg2=$(echo "$arg1" | tr '[:upper:]' '[:lower:]')
echo "python src/main.py fit --model $arg1  --data $data --data.data_dir 'data' --trainer.logger TensorBoardLogger --trainer.logger.save_dir './logs/$(echo $arg2)_logs' --trainer.check_val_every_n_epoch 1 --trainer.max_epochs 600 --trainer.log_every_n_steps 5"
python src/main.py fit --model $arg1  --data $data --data.data_dir 'data' --trainer.logger TensorBoardLogger --trainer.logger.save_dir "./logs/$(echo $arg2)_logs" --trainer.check_val_every_n_epoch 1 --trainer.max_epochs 600 --trainer.log_every_n_steps 5
