echo "GPU 0 on Swift"
for i in {1..10};
do
	steps=$((20000*$i))
	./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=$steps --workspace_dir $HOME/mount/ft/workspace --config ./workbench/hole4/hole.pbtxt --notify_me="fivos_ts@hotmail.com"
	./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=$steps --workspace_dir $HOME/mount/ft/workspace --config ./workbench/hole4/hole_dupe.pbtxt --notify_me="fivos_ts@hotmail.com"
	./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=$steps --workspace_dir $HOME/mount/ft/workspace --config ./workbench/hole4/hole_few_preds.pbtxt --notify_me="fivos_ts@hotmail.com"
	./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=$steps --workspace_dir $HOME/mount/ft/workspace --config ./workbench/hole4/hole_many_preds.pbtxt --notify_me="fivos_ts@hotmail.com"
	./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=$steps --workspace_dir $HOME/mount/ft/workspace --config ./workbench/hole4/hole_normal.pbtxt --notify_me="fivos_ts@hotmail.com"
	./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=$steps --workspace_dir $HOME/mount/ft/workspace --config ./workbench/hole4/hole_seq.pbtxt --notify_me="fivos_ts@hotmail.com"
done
