from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import torchvision
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torchvision.transforms as trn
from torchvision.transforms.functional import rotate
from torchvision.transforms import GaussianBlur
import torch

class Training_Dataset(Dataset):
    def __init__(
        self,
        inputs,
        targets,
        output_dim,
        verbose=True,
        normalize_inputs=True,
        normalize_targets=True,
    ):
        self.inputs = inputs
        if normalize_targets:
            self.targets_mean = np.mean(targets, axis=0, keepdims=True)
            self.targets_std = np.std(targets, axis=0, keepdims=True)
        else:
            self.targets_mean = 0
            self.targets_std = 1
        self.targets = (targets - self.targets_mean) / self.targets_std

        self.n_samples = inputs.shape[0]
        self.input_dim = inputs.shape[1]
        self.output_dim = output_dim

        # Normalize inputs
        if normalize_inputs:
            self.inputs_std = np.std(self.inputs, axis=0, keepdims=True) + 1e-6
            self.inputs_mean = np.mean(self.inputs, axis=0, keepdims=True)
        else:
            self.inputs_mean = 0
            self.inputs_std = 1

        self.inputs = (self.inputs - self.inputs_mean) / self.inputs_std
        if verbose:
            print("Number of samples: ", self.n_samples)
            print("Input dimension: ", self.input_dim)
            print("Label dimension: ", self.output_dim)
            print("Labels mean value: ", self.targets_mean)
            print("Labels standard deviation: ", self.targets_std)


    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


class Test_Dataset(Dataset):
    def __init__(self, inputs, output_dim, targets=None, inputs_mean=0.0, inputs_std=1.0):
        self.inputs = (inputs - inputs_mean) / inputs_std
        self.targets = targets
        self.n_samples = inputs.shape[0]
        self.input_dim = inputs.shape[1]
        if self.targets is not None:
            self.output_dim = output_dim

    def __getitem__(self, index):
        if self.targets is None:
            return self.inputs[index]
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


data_base = "./data"


class Synthetic_Dataset(Dataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        rng = np.random.default_rng(seed=0)

        def f(x):
            return np.sin(4 * x) 

        # inputs = rng.standard_normal(300)
        inputs = np.linspace(-2.0,2.0, 300)
        targets = f(inputs) + rng.standard_normal(inputs.shape)*0.1
        
        mask = (((inputs > -1) * (inputs < -0.5)) | ((inputs < 1) * (inputs > 0.5))).flatten()
        mask2 = ((inputs > -0.5) * (inputs < 0.5)).flatten()

        #mask = ...
        
        self.train = Training_Dataset(
            inputs[mask, np.newaxis],
            targets[mask, np.newaxis],
            self.output_dim,
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.val = Test_Dataset(
            inputs[mask2, np.newaxis],
            self.output_dim,
            targets[mask2, np.newaxis],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            inputs[..., np.newaxis],
            self.output_dim,
            targets[..., np.newaxis],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 300

    def get_split(self, *args):
        return self.train, self.val, self.test

    def len_train(self):
        return 200 



class Synthetic2_Dataset(Dataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        data = np.load("../data/data.npy")
        inputs, targets = data[:, 0], data[:, 1]

        test_inputs = np.linspace(np.min(inputs)-5, np.max(inputs)+5, 200)

        self.train = Training_Dataset(
            inputs[..., np.newaxis],
            targets[..., np.newaxis],
            output_dim=1,
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.full = Test_Dataset(
            test_inputs[..., np.newaxis],
            1,
            None,
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_inputs[..., np.newaxis],
            1,
            None,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return 400

    def get_split(self, *args):
        return self.train, self.full, self.test

    def len_train(self):
        return 400 

    
class MNIST_Dataset(Dataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 10
        self.output_dim = 10
        self.corruption_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]

        train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        test = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        n_val = 5000
        train_data = train.data.reshape(60000, -1) / 255.0
        self.input_dim = train_data.shape[1]
        test_data = test.data.reshape(10000, -1) / 255.0
        train_targets = train.targets.reshape(-1, 1)
        test_targets = test.targets.reshape(-1, 1)

        self.train = Training_Dataset(
            train_data.numpy()[:-n_val],
            train_targets.numpy()[:-n_val],
            self.output_dim,
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.val = Test_Dataset(
            train_data.numpy()[-n_val:],
            self.output_dim,
            train_targets.numpy()[-n_val:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data.numpy(),
            self.output_dim,
            test_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

        test2 = datasets.FashionMNIST(
                    root="./data",
                    train=False,
                    download=True,
                    transform=transforms.ToTensor(),
                )

        test2_data = test2.data.reshape(10000, -1) / 255.0
        
        ood_test_data = np.concatenate([test_data, test2_data])
        ood_test_targets = np.concatenate(
            [
                np.zeros(test_data.shape[0]),
                np.ones(test2_data.shape[0])
            ]).reshape(-1, 1)

        self.ood_test = Test_Dataset(
            ood_test_data,
            self.output_dim,
            ood_test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )


    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)

    def get_split(self, *args):
        return self.train, self.val, self.test
    
    def get_corrupted_split(self, angle):
        test = datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        rotated_data = rotate(test.data, angle)
        
        test_data = rotated_data.reshape(10000, -1) / 255.0
        test_targets = test.targets.reshape(-1, 1)

        return Test_Dataset(
            test_data.numpy(),
            self.output_dim,
            test_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def len_train(self):
        return len(self.train)
    
    def get_ood_datasets(self):
        return self.ood_test
    

class FMNIST_Dataset(Dataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 10
        self.output_dim = 10
        self.corruption_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
        train = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        test = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.input_shape = [1, 28, 28]

        n_val = 5000
        train_data = train.data.reshape(60000, -1) / 255.0

        self.input_dim = train_data.shape[1]
        test_data = test.data.reshape(10000, -1) / 255.0
        train_targets = train.targets.reshape(-1, 1)
        test_targets = test.targets.reshape(-1, 1)

        self.train = Training_Dataset(
            train_data.numpy()[:-n_val],
            train_targets.numpy()[:-n_val],
            self.output_dim,
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.val = Test_Dataset(
            train_data.numpy()[-n_val:],
            self.output_dim,
            train_targets.numpy()[-n_val:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data.numpy(),
            self.output_dim,
            test_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

        test2 = datasets.MNIST(
                    root="./data",
                    train=False,
                    download=True,
                    transform=transforms.ToTensor(),
                )

        test2_data = test2.data.reshape(10000, -1) / 255.0
        
        ood_test_data = np.concatenate([test_data, test2_data])
        ood_test_targets = np.concatenate(
            [
                np.zeros(test_data.shape[0]),
                np.ones(test2_data.shape[0])
            ]).reshape(-1, 1)

        self.ood_test = Test_Dataset(
            ood_test_data,
            self.output_dim,
            ood_test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)

    def get_split(self, *args):
        return self.train, self.val, self.test

    def get_corrupted_split(self, angle):
        test = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        rotated_data = rotate(test.data, angle)
        
        test_data = rotated_data.reshape(10000, -1) / 255.0
        test_targets = test.targets.reshape(-1, 1)

        return Test_Dataset(
            test_data.numpy(),
            self.output_dim,
            test_targets.numpy(),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def len_train(self):
        return len(self.train)
    
    def get_ood_datasets(self):
        return self.ood_test


class CIFAR10_Dataset(Dataset):
    def __init__(self):
        self.type = "multiclass"
        self.classes = 10
        self.output_dim = 10
        self.corruption_types = 19
        self.corruption_values = 5
        train = datasets.CIFAR10(
            root="./data", train=True, download=True
        )
        test = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
        )
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        train_data = (train.data/255.0 - mean)/std
        test_data = (test.data/255.0 - mean)/std

        self.input_shape = train_data.shape[1:]
        n_val = 5000
        train_data = np.transpose(train_data, (0, 3, 1, 2))
        test_data = np.transpose(test_data, (0, 3, 1, 2))

        self.input_dim = train_data.shape[1]
    
        train_targets = np.array(train.targets).reshape(-1, 1)
        test_targets = np.array(test.targets).reshape(-1, 1)
        

        self.train = Training_Dataset(
            train_data[:-n_val],
            train_targets[:-n_val],
            self.output_dim,
            normalize_targets=False,
            normalize_inputs=False,
        )


        torch.manual_seed(0)

        valid_dataset = datasets.CIFAR10(root="./data",
                                        train=True,
                                        transform=transforms.Compose([
                                            transforms.RandomResizedCrop(size=32, scale=(0.5, 1)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean,
                                                                std=std)
		                                ]), download=True)
        num_train = len(train_data)
        indices = list(range(num_train))
        valid_idx = indices[-n_val:]
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

        val_loader = torch.utils.data.DataLoader(
			valid_dataset, batch_size=100, sampler=valid_sampler, shuffle = False)


		# remove the randomness
        xs, ys = [], []
        for x, y in val_loader:
            xs.append(x); ys.append(y)
        xs = torch.cat(xs); ys = torch.cat(ys)

        self.val = Test_Dataset(
            xs,
            self.output_dim,
            ys,
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            test_data,
            self.output_dim,
            test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

        test2 = datasets.SVHN(
            root="./data",
            split="test",
            download=True,
            transform=transforms.ToTensor(),
        )
        mean = np.expand_dims(np.expand_dims(mean, -1),-1)
        std = np.expand_dims(np.expand_dims(std, -1),-1)

        test2_data = (test2.data/255.0 - mean)/std
        ood_test_data = np.concatenate([test_data, test2_data])
        ood_test_targets = np.concatenate(
            [
                np.zeros(test_data.shape[0]),
                np.ones(test2_data.shape[0])
            ]).reshape(-1, 1)

        self.ood_test = Test_Dataset(
            ood_test_data,
            self.output_dim,
            ood_test_targets,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)

    def get_split(self, *args):
        return self.train, self.val, self.test
    
    def get_corrupted_split(self, corruption_type, corruption_value):
        corrupted_data_path = "./data/CIFAR-10-C/"
        corrupted_data_files = os.listdir(corrupted_data_path)
        corrupted_data_files.remove('labels.npy')

        if 'README.txt' in corrupted_data_files:
            corrupted_data_files.remove('README.txt')
        labels = torch.from_numpy(
            np.load(os.path.join(corrupted_data_path, 'labels.npy'), allow_pickle=True)).long()
        
        corrupted_data_file = corrupted_data_files[corruption_type]
        map = np.lib.format.open_memmap(corrupted_data_path + corrupted_data_file, mode='r+')
        subset = map[corruption_value*10000:(corruption_value+1)*10000]
        
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        test_data = (subset/255.0 - mean)/std
        test_data = np.transpose(test_data, (0, 3, 1, 2))

        return Test_Dataset(
            test_data,
            self.output_dim,
            labels.reshape(-1, 1),
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def len_train(self):
        return len(self.train)

    def get_ood_datasets(self):
        return self.ood_test
    


def imagenet_loaders(args, valid_size=0.01):

    data_dir = "../../proyectos/ada2/ludvins/ImageNet/"
    from timm.data import create_transform
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
    T = transforms.Compose([
        transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


    T_val = transforms.Compose([
        transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    print("Loading Test Data...")
    test_dataset = torchvision.datasets.ImageNet(data_dir, "val", transform = T_val)
    print("Loading Train Data...")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    train_dataset = torchvision.datasets.ImageNet(data_dir, "train", transform=T)
    if valid_size > 0:
        valid_dataset = torchvision.datasets.ImageNet(data_dir, "train",
            transform=create_transform(
                input_size=224,
                scale=(0.08, 0.1),
                is_training=True,
                color_jitter=0.4,
                auto_augment=None, #'original', #'v0' #'rand-m9-mstd0.5-inc1', #'v0', 'original'
                interpolation='bicubic',
                re_prob=0.25, #0.25,
                re_mode='pixel',
                re_count=1,
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            )
        )
        num_train = len(train_dataset)
        
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)
        print(len(train_idx))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, pin_memory=False)
	
        print("Loading Val Data...")
        xs, ys = [], []
        for _ in range(1):
            for x, y in val_loader:
                xs.append(x); ys.append(y)
        xs = torch.cat(xs); ys = torch.cat(ys)
        valid_dataset = torch.utils.data.TensorDataset(xs, ys)
        val_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        print("ImageNet Data Prepared")
        return train_loader, val_loader, test_loader
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False)
        print(len(train_dataset))
        return train_loader, test_loader, train_dataset


class Airline_Dataset(Dataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        data = pd.read_csv("./data/airline.csv")
        # Convert time of day from hhmm to minutes since midnight
        data.ArrTime = 60 * np.floor(data.ArrTime / 100) + np.mod(data.ArrTime, 100)
        data.DepTime = 60 * np.floor(data.DepTime / 100) + np.mod(data.DepTime, 100)

        # Pick out the data
        Y = data["ArrDelay"].values[:800000].reshape(-1, 1)
        names = [
            "Month",
            "DayofMonth",
            "DayOfWeek",
            "plane_age",
            "AirTime",
            "Distance",
            "ArrTime",
            "DepTime",
        ]
        X = data[names].values[:800000]

        self.n_train = 600000
        self.n_val = 100000
        self.train = Training_Dataset(X[: self.n_train], Y[: self.n_train],
            self.output_dim)
        self.input_dim = X.shape[1]

        self.train_test = Test_Dataset(
            X[:self.n_train],
            self.output_dim,
            Y[:self.n_train],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        
        self.val = Test_Dataset(
            X[self.n_train:self.n_train + self.n_val],
            self.output_dim,
            Y[self.n_train:self.n_train + self.n_val],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            X[self.n_train + self.n_val:],
            self.output_dim,
            Y[self.n_train + self.n_val:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return self.len_data

    def len_train(self):
        return self.n_train

    def get_split(self, split, *args):
        return self.train, self.val, self.test


class Year_Dataset(Dataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1
        try:
            data = pd.read_csv(
                "./data/YearPredictionMSD.txt", header=None, delimiter=","
            ).values
        except:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
            with urlopen(url) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall("./data/")

            data = pd.read_csv(
                "./data/YearPredictionMSD.txt", header=None, delimiter=","
            ).values

        self.len_data = data.shape[0]

        X = data[:, 1:]
        self.input_dim = X.shape[1]
        Y = data[:, 0].reshape(-1, 1)

        self.n_train = 400000
        self.n_val = 63715
        self.train = Training_Dataset(X[: self.n_train], Y[: self.n_train],
            self.output_dim)

        self.train_test = Test_Dataset(
            X[:self.n_train],
            self.output_dim,
            Y[:self.n_train],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

        self.val = Test_Dataset(
            X[self.n_train:self.n_train + self.n_val],
            self.output_dim,
            Y[self.n_train:self.n_train + self.n_val],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            X[self.n_train + self.n_val:],
            self.output_dim,
            Y[self.n_train + self.n_val:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return self.len_data

    def len_train(self):
        return self.n_train

    def get_split(self, split, *args):
        return self.train, self.val, self.test
        

class Taxi_Dataset(Dataset):
    def __init__(self):
        self.type = "regression"
        self.output_dim = 1

        if os.path.exists("data/taxi.csv"):
            print("Taxi csv file found.")
            data = pd.read_csv("data/taxi.csv")
        elif os.path.exists("data/taxi.zip"):
            print("Taxi zip file found.")
            data = pd.read_csv("data/taxi.zip", compression="zip", dtype=object)
        else:
            print("Downloading Taxi Dataset...")
            url = (
                "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
            )
            data = pd.read_parquet(url)
            data.to_csv("data/taxi.csv")

        data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])
        data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_dropoff_datetime"])
        data["day_of_week"] = data["tpep_pickup_datetime"].dt.dayofweek
        data["day_of_month"] = data["tpep_pickup_datetime"].dt.day
        data["month"] = data["tpep_pickup_datetime"].dt.month
        print(data["tpep_pickup_datetime"].dt)
        data["time_of_day"] = (
            data["tpep_pickup_datetime"] - data["tpep_pickup_datetime"].dt.normalize()
        ) / pd.Timedelta(seconds=1)
        data["trip_duration"] = (
            data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]
        ).dt.total_seconds()
        data = data[
            [
                "time_of_day",
                "day_of_week",
                "day_of_month",
                "month",
                "PULocationID",
                "DOLocationID",
                "trip_distance",
                "trip_duration",
                "total_amount"
            ]
        ]
        data = data[data["trip_duration"] >= 10]
        data = data[data["trip_duration"] <= 5 * 3600]
        data = data.astype(float)
        data = data.values
        X = data[:, :-1]
        Y = data[:, -1][:, np.newaxis]
        self.input_dim = X.shape[1]

        self.n_train = int(X.shape[0] * 0.8)
        self.n_val = int(X.shape[0] * 0.1)
        self.train = Training_Dataset(X[: self.n_train], Y[: self.n_train],
            self.output_dim)

        self.train_test = Test_Dataset(
            X[:self.n_train],
            self.output_dim,
            Y[:self.n_train],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

        self.val = Test_Dataset(
            X[self.n_train:self.n_train + self.n_val],
            self.output_dim,
            Y[self.n_train:self.n_train + self.n_val],
            self.train.inputs_mean,
            self.train.inputs_std,
        )
        self.test = Test_Dataset(
            X[self.n_train + self.n_val:],
            self.output_dim,
            Y[self.n_train + self.n_val:],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def __len__(self):
        return self.len_data

    def len_train(self):
        return self.n_train

    def get_split(self, split, *args):
        return self.train, self.val, self.test

        
def get_dataset(dataset_name):
    d = {
        "synthetic": Synthetic_Dataset,
        "synthetic2": Synthetic2_Dataset,
        "MNIST": MNIST_Dataset,
        "FMNIST": FMNIST_Dataset,
        "CIFAR10": CIFAR10_Dataset,
        "Airline": Airline_Dataset,
        "Year": Year_Dataset,
        "Taxi": Taxi_Dataset
    }

    return d[dataset_name]()
