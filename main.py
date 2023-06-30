import argparse
import yaml
import torch
import sys
import importlib

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split
import lightning_fabric as lf

from engine import Engine

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='./configs/mnist.yaml', help='configuration file')
parser.add_argument('--mode', type=str, default='train', help='train or test')

args = parser.parse_args()

with open(args.config, 'r') as f :
    config = yaml.safe_load(f)
    
print(args)
print(config)
print('Python Version:', sys.version)
print('PyTorch Version:', torch.__version__)
print('Number of GPUs:', torch.cuda.device_count())

def train() :
    lf.utilities.seed.seed_everything(seed = config['random_seed']) # seed 고정
    
    # ⚡⚡ 1. Set 'Dataset', 'DataLoader'
    train_dataset = importlib.import_module('dataloader.' + config['dataloader']).__getattribute__('training_dataset')
    # importlib.import_module() : import할 모듈의 이름을 문자열로 전달하면, 해당 모듈을 import해줌
    # __getattribute__() : 해당 모듈에서 training_dataset 함수를 가져옴. 즉, train_dataset = training_dataset()이 됨
    train_dataset = train_dataset()
    
    train_dataset, val_dataset = random_split(train_dataset, [55000, 5000])
    
    train_dataloader = DataLoader(
            dataset = train_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=True # True : Tensor를 CUDA 고정 메모리에 올림
        )
    
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True
    )

    
    # ⚡⚡ 2. Set 'Model', 'Loss', 'Optimizer', 'Scheduler'
    model = importlib.import_module('models.' + config['model']).__getattribute__('MainModel')
    model = model()
    
    loss_function = importlib.import_module('loss.' + config['loss']).__getattribute__('loss_function')
    
    optimizer = importlib.import_module('optimizer.' + config['optimizer']).__getattribute__('Optimizer')
    optimizer = optimizer(model.parameters(), **config['optimizer_config'])
    
    scheduler = importlib.import_module('scheduler.' + config['scheduler']).__getattribute__('Scheduler')
    scheduler = scheduler(optimizer, **config['scheduler_config'])
    
    # ⚡⚡  3. Set 'engine' for training/validation and 'Trainer' 
    # resume training from an old checkpoint
    if config['resume_checkpoint']  is not None :
        engine = Engine.load_from_checkpoint(model = model, optimizer=optimizer, loss_function=loss_function, scheduler=scheduler, checkpoint_path = config['resume_checkpoint'])
        print(config['resume_checkpoint'] + "are loaded")
    else :
        engine = Engine(model, loss_function, optimizer, scheduler)
    
    # ⚡⚡ 4. Init ModelCheckpoint callback, monitoring "val_loss"
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor = "val_loss",
        mode = "min",
        filename = "{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
    )
    
    # ⚡⚡ 5. Set pl.Trainer
    trainer = pl.Trainer(
        deterministic=True, # 모델의 훈련을 재현 가능하게 만들어주는 옵션. 속도 저하 가능성 있음
        default_root_dir = config['default_root_dir'], # 로그와 체크포인트 저장할 디렉토리 지정
        devices = config['devices'], # 사용할 GPU 지정
        val_check_interval = 1.0, # Check val every n train epochs
        max_epochs = config['max_epoch'], # 최대 epoch 지정
        sync_batchnorm = True, # 배치 정규화를 동기화
        callbacks = [checkpoint_callback], # train 도중 특정 작업을 수행하기 위한 콜백 함수 설정
        accelerator = config['accelerator'], # gpu or cpu
        num_sanity_val_steps = config['num_sanity_val_steps'], # train 전 검증에 오류가 없는지 확인.
        gradient_clip_val=1.0, # 경사하강법을 사용하여 파라미터를 업데이트 할 때, 경사가 너무 커지는 것을 방지하기 위해 경사를 잘라내는 최대값을 설정
        profiler = config['profiler'], # 프로파일러
    )
    
    # ⚡⚡ 6. Start training
    
    if config['resume_checkpoint'] is not None :
        trainer.fit(engine, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=config['resume_checkpoint'])
        print(config['resume_checkpoint'] + "are loaded")
    else :
        trainer.fit(engine, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        print("no checkpoint are loaded")
        

def test() :
    print("test")
    
    lf.utilities.seed.seed_everything(seed = config['random_seed']) # seed 고정
    
    test_dataset = importlib.import_module('dataloader.' + config['dataloader']).__getattribute__('test_dataset')
    test_dataset = test_dataset()
    
    # ⚡⚡ 1. Set 'Dataset', 'DataLoader
    test_dataloader = DataLoader(
            dataset = test_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=True
        )
    
    # ⚡⚡ 2. Set 'Model', 'Loss', 'Optimizer', 'Scheduler'
    model = importlib.import_module('models.' + config['model']).__getattribute__('MainModel')
    model = model()
    
    loss_function = importlib.import_module('loss.' + config['loss']).__getattribute__('loss_function')
    
    optimizer = importlib.import_module('optimizer.' + config['optimizer']).__getattribute__('Optimizer')
    optimizer = optimizer(model.parameters(), **config['optimizer_config'])
    
    scheduler = importlib.import_module('scheduler.' + config['scheduler']).__getattribute__('Scheduler')
    scheduler = scheduler(optimizer, **config['scheduler_config'])
    
    # ⚡⚡  3. Load model
    engine = Engine.load_from_checkpoint(model = model, optimizer=optimizer, loss_function=loss_function, scheduler=scheduler, checkpoint_path = config['resume_checkpoint'])

    # ⚡⚡ 4. pl.Trainer & test
    trainer = pl.Trainer(accelerator=config['accelerator'], devices=config['devices'])
    trainer.test(engine, test_dataloaders=test_dataloader)
    
if __name__ == "__main__" :
    
    if args.mode == 'train' :
        train()
    
    elif args.mode == 'test' :
        test()
        