arg1=$1
arg2=$(echo "$arg1" | tr '[:upper:]' '[:lower:]')
echo "python src/main.py fit --model $arg1  --data UrbanWinds2DLidarDataModule --data.data_dir 'data' --trainer.logger TensorBoardLogger --trainer.logger.save_dir './logs/$(echo $arg2)_logs' --trainer.check_val_every_n_epoch 10 --trainer.max_epochs 10000 --trainer.log_every_n_steps 5"
python src/main.py fit --model $arg1  --data UrbanWinds2DLidarDataModule --data.data_dir 'data' --trainer.logger TensorBoardLogger --trainer.logger.save_dir "./logs/$(echo $arg2)_logs" --trainer.check_val_every_n_epoch 10 --trainer.max_epochs 10000 --trainer.log_every_n_steps 5