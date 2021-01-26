import resource

from dali_loader import DALIDataloader, HybridTrainPipe_CIFAR


soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (576460752303423488, hard))

pip_train = HybridTrainPipe_CIFAR(device_id=0)
train_loader = DALIDataloader(
    pipeline=pip_train
)

for i, data in enumerate(train_loader):
    print(data[0])
    print(data[1])
    break
