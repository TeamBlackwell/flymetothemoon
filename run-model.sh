arg1=$1
arg2=$(echo "$arg1" | tr '[:upper:]' '[:lower:]')
<<<<<<< HEAD
echo "python src/main.py fit --model $arg1  --data UrbanWinds2DDataModule --data.data_dir 'data' --trainer.logger TensorBoardLogger --trainer.logger.save_dir './logs/$(echo $arg2)_logs' --trainer.check_val_every_n_epoch 1 --trainer.max_epochs 100000000 --trainer.log_every_n_steps 1"
python src/main.py fit --model $arg1  --data UrbanWinds2DDataModule --data.data_dir 'data' --trainer.logger TensorBoardLogger --trainer.logger.save_dir "./logs/$(echo $arg2)_logs" --trainer.check_val_every_n_epoch 1 --trainer.max_epochs 100000000 --trainer.log_every_n_steps 1
=======
echo "python src/main.py fit --model $arg1  --data UrbanWinds2DLidarDataModule --data.data_dir 'data' --trainer.logger TensorBoardLogger --trainer.logger.save_dir './logs/$(echo $arg2)_logs' --trainer.check_val_every_n_epoch 10 --trainer.max_epochs 10000 --trainer.log_every_n_steps 5"
python src/main.py fit --model $arg1  --data UrbanWinds2DLidarDataModule --data.data_dir 'data' --trainer.logger TensorBoardLogger --trainer.logger.save_dir "./logs/$(echo $arg2)_logs" --trainer.check_val_every_n_epoch 10 --trainer.max_epochs 10000 --trainer.log_every_n_steps 5
>>>>>>> 08e9acf4294be5c656084138191eadf431e3dc29
