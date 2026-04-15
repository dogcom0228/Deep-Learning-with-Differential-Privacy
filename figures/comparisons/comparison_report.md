# Results Comparison Report

Total runs: 8
Experiments: 3
Datasets: 2

## Globally varying config fields
- dataset.name
- dataset.augment
- model.name
- optimizer.lr
- optimizer.weight_decay
- optimizer.nesterov
- scheduler.warmup_epochs
- scheduler.min_lr
- privacy.enabled
- privacy.noise_multiplier
- privacy.poisson_sampling
- privacy.grad_sample_mode
- runtime.channels_last
- runtime.amp
- training.batch_size

## Dataset-level takeaways
- cifar10: best run is cifar10-dp-sgd/20260415-024451 with best_eval_accuracy=0.4771
- cifar10: best private run is cifar10-dp-sgd/20260415-024451 with epsilon=6.794
- mnist: best run is mnist-sgd/20260415-143213 with best_eval_accuracy=0.9837
- mnist: best private run is mnist-dp-sgd/20260415-015205 with epsilon=21.24
- mnist: private vs non-private best accuracy gap = -0.0352

## Experiment-level sweeps
- cifar10-dp-sgd: 1 run(s)
- cifar10-dp-sgd: varying parameters -> none
- cifar10-dp-sgd: lowest epsilon run is 20260415-024451 with epsilon=6.7936
- mnist-dp-sgd: 6 run(s)
- mnist-dp-sgd: varying parameters -> privacy.noise_multiplier
- mnist-dp-sgd: lowest epsilon run is 20260415-023553 with epsilon=7.0212
- mnist-sgd: 1 run(s)
- mnist-sgd: varying parameters -> none
