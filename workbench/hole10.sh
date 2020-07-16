echo "GPU 0 on Swift"

./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=150000 --workspace_dir $HOME/workspace --config ./workbench/hole10/hole.pbtxt --notify_me="fivos_ts@hotmail.com"
./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=150000 --workspace_dir $HOME/workspace --config ./workbench/hole10/hole_dupe.pbtxt --notify_me="fivos_ts@hotmail.com"
./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=150000 --workspace_dir $HOME/workspace --config ./workbench/hole10/hole_few_preds.pbtxt --notify_me="fivos_ts@hotmail.com"
./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=150000 --workspace_dir $HOME/workspace --config ./workbench/hole10/hole_many_preds.pbtxt --notify_me="fivos_ts@hotmail.com"
./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=150000 --workspace_dir $HOME/workspace --config ./workbench/hole10/hole_normal.pbtxt --notify_me="fivos_ts@hotmail.com"
./clgen  --min_samples 2 --sample_per_epoch=0 --num_train_steps=150000 --workspace_dir $HOME/workspace --config ./workbench/hole10/hole_seq.pbtxt --notify_me="fivos_ts@hotmail.com"
