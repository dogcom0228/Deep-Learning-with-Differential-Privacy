python -m dp_sgd train --config configs/mnist-dp.yaml
python -m dp_sgd train --config configs/mnist-dp.yaml --override privacy.noise_multiplier=1.2
python -m dp_sgd train --config configs/mnist-dp.yaml --override privacy.noise_multiplier=1.4
python -m dp_sgd train --config configs/mnist-dp.yaml --override privacy.noise_multiplier=1.6
python -m dp_sgd train --config configs/mnist-dp.yaml --override privacy.noise_multiplier=1.8
python -m dp_sgd train --config configs/mnist-dp.yaml --override privacy.noise_multiplier=2.0

python -m dp_sgd train --config configs/cifar10-dp.yaml
python -m dp_sgd train --config configs/cifar10-dp.yaml --override privacy.noise_multiplier=1.2
python -m dp_sgd train --config configs/cifar10-dp.yaml --override privacy.noise_multiplier=1.4
python -m dp_sgd train --config configs/cifar10-dp.yaml --override privacy.noise_multiplier=1.6
python -m dp_sgd train --config configs/cifar10-dp.yaml --override privacy.noise_multiplier=1.8
python -m dp_sgd train --config configs/cifar10-dp.yaml --override privacy.noise_multiplier=2.0

python -m dp_sgd train --config configs/mnist-sgd.yaml
python -m dp_sgd train --config configs/cifar10-sgd.yaml
