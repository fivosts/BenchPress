echo "GPU 1 on CC3"
EXPORT CUDA_VISIBLE_DEVICES='1'
for i in {1..10};
do
	steps=$((20000*$i))
	./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=$steps --workspace_dir $HOME/workspace --config ./workbench/hole2/hole.pbtxt --notify_me="fivos_ts@hotmail.com"
	./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=$steps --workspace_dir $HOME/workspace --config ./workbench/hole2/hole.pbtxt --notify_me="fivos_ts@hotmail.com"
	./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=$steps --workspace_dir $HOME/workspace --config ./workbench/hole2/hole.pbtxt --notify_me="fivos_ts@hotmail.com"
	./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=$steps --workspace_dir $HOME/workspace --config ./workbench/hole2/hole.pbtxt --notify_me="fivos_ts@hotmail.com"
	./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=$steps --workspace_dir $HOME/workspace --config ./workbench/hole2/hole.pbtxt --notify_me="fivos_ts@hotmail.com"
	./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=$steps --workspace_dir $HOME/workspace --config ./workbench/hole2/hole.pbtxt --notify_me="fivos_ts@hotmail.com"
done
