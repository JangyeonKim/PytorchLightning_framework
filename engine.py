import torch
import pytorch_lightning as pl

class Engine(pl.LightningModule) :
    def __init__(self, model, loss_function, optimizer, scheduler) :
        super().__init__()
         # ⚡ model
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.save_hyperparameters(ignore=['model']) # 원하는 hyperparameter 정보를 저장할 수 있음
    
     # ⚡ forward
    def forward(self, x) :
        return self.model(x)
    
     # ⚡ train
    def training_step(self, batch, batch_idx) :
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
     # ⚡ validation
    def validation_step(self, batch, batch_idx) :
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        
        correct = (y_hat.argmax(1) == y).type(torch.float).sum().item()
        size = x.shape[0]
        
        return {'correct': correct, 'size': size}
    
    def validation_epoch_end(self, validation_step_outputs) :
        tot_correct = sum([x['correct'] for x in validation_step_outputs])
        tot_size = sum([x['size'] for x in validation_step_outputs])
        
        acc = tot_correct / tot_size
        self.log('val_acc', acc*100, on_epoch=True, prog_bar=True, sync_dist=True)
    
     # ⚡ test
    def test_step(self, batch, batch_idx) :
        x, y = batch
        y_hat = self(x)
        # loss = self.loss_function(y_hat, y)
        # self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        correct = (y_hat.argmax(1) == y).type(torch.float).sum().item()
        size = x.shape[0]
        
        return {'correct': correct, 'size': size}
    
    def test_epoch_end(self, test_step_outputs) :
        tot_correct = sum([x['correct'] for x in test_step_outputs])
        tot_size = sum([x['size'] for x in test_step_outputs])
        
        acc = tot_correct / tot_size
        self.log('test_acc', acc*100, on_epoch=True, prog_bar=True, sync_dist=True)
    
     # ⚡ optimizer
    def configure_optimizers(self) :
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }