# Variational Linearized Laplace Approximation for Bayesian Deep Learning

## Requirements

Python 3.10.10
Torch version 1.10 (required by BackPack)

1. Create environment as `python -m venv .venv`
2. Activate environment as `source .venv/bin/activate`
3. Update pip as `pip install --upgrade pip`
4. Install requirements as `pip install -r requirements.txt`

## Regression

Dataset options: Year, Airline and Taxi

```
python ./scripts/regression/map.py --verbose --dataset Year --net_structure 200 200 200 --MAP_iterations 20000 --iterations 40000 --split 0 --weight_decay 0.01 --bb_alpha 1 --seed 0 --dtype float64 --MAP_lr 0.01
```

```
python ./scripts/regression/lla.py --subset last_layer --hessian full --verbose --dataset Year --net_structure 200 200 200 --MAP_iterations 20000 --iterations 40000 --split 0 --weight_decay 0.01 --bb_alpha 1 --seed 0 --dtype float64 --MAP_lr 0.01
```


```
srun python ./scripts/regression/ella.py --verbose --dataset Year --net_structure 200 200 200 --MAP_iterations 20000 --iterations 40000 --split 0 --num_inducing 2000 --weight_decay 0.01 --bb_alpha 1 --prior_std 1 --ll_log_va -5 --seed 0 --MAP_lr 0.01
```

```
srun python ./scripts/regression/valla.py --verbose --dataset Year --net_structure 200 200 200 --MAP_iterations 20000 --iterations 40000 --split 0 --num_inducing 100 --weight_decay 0.01 --bb_alpha 1 --seed 0 --dtype float64 --MAP_lr 0.01
```

## MNIST/FMNIST

Dataset options: MNIST and FMNIST

```
python ./scripts/multiclass/map.py --verbose --dataset MNIST --MAP_iterations 20000 --iterations 40000 --test_ood --test_corruptions --split 0 --weight_decay 0.001 --bb_alpha 1 --seed 0 --MAP_lr 0.001
```

```
python ./scripts/multiclass/lla.py --hessian last_layer --subset full --verbose --dataset MNIST --MAP_iterations 20000 --iterations 40000 --test_ood --test_corruptions --split 0 --weight_decay 0.001 --bb_alpha 1 --seed 0 --MAP_lr 0.001
```

```
python ./scripts/multiclass/ella.py --verbose --dataset MNIST --MAP_iterations 20000 --iterations 40000 --test_ood --test_corruptions --split 0 --num_inducing 2000 --weight_decay 0.001 --bb_alpha 1 --prior_std 1 --seed 0 --MAP_lr 0.001
```

```
python ./scripts/multiclass/valla.py --verbose --dataset MNIST --MAP_iterations 20000 --iterations 40000 --test_ood --test_corruptions --split 0 --num_inducing 100 --weight_decay 0.001 --bb_alpha 1 --seed 0 --MAP_lr 0.001
```

## ResNet

ResNet options: Resnet18, ResNet32, ResNet44 and ResNet56

```
python ./scripts/multiclass_resnet/valla_backpack.py --verbose --batch_size 100 --dataset CIFAR10 --MAP_iterations 50000 --iterations 40000 --split 0 --resnet resnet18 --num_inducing 100 --weight_decay 0.001 --bb_alpha 1 --seed 0 --test_ood --test_corruptions --MAP_lr 0.01 --device gpu
```
