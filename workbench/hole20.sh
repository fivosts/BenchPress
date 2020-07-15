echo "GPU 1 on CC3"
for i in {1..10};
do
	steps=$((100000*$i))
	./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=$steps --workspace_dir $HOME/mount/ft/workspace --config ./workbench/hole20/hole.pbtxt --notify_me="fivos_ts@hotmail.com"
done
