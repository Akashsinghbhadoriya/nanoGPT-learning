import torch
import math
import os

class Trainer:

    def __init__(self, model, config, optimizer, train_loader, val_loader, train_args, device):
        
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_args = train_args
        self.device = device
        self.config = config

        self.model.to(self.device)

    def fit(self):

        self.model.train()

        step = 0
        best_val_loss = 1e9
        data_iter = iter(self.train_loader)
        while step < self.train_args.max_iters:

            lr = self.get_lr(step) if self.train_args.decay_lr else self.train_args.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if step % self.train_args.eval_interval == 0 and step > 0:
                losses = self.estimate_loss()
                print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                is_best_loss = losses['val'] < best_val_loss
                if is_best_loss:
                    best_val_loss = losses['val']
                self.add_checkpoint(step, best_val_loss)

            try:
                x, y = next(data_iter)
            except StopIteration:
                # Re-initialize dataset iterator once an epoch finishes
                data_iter = iter(self.train_loader)
                x, y = next(data_iter)

            x = x.to(self.device)
            y = y.to(self.device)

            logits, loss = self.model(x,y)

            self.optimizer.zero_grad(set_to_none=True)

            loss.backward()

            self.optimizer.step()

            if step % self.train_args.log_interval == 0:
                print(f"step:{step}, train loss:{loss.item():.4f}")

            step+=1

    def add_checkpoint(self, step, best_val_loss):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_args': self.train_args,
            'iter_num': step,
            'best_val_loss': best_val_loss,
            'config': self.config,
        }

        out_dir = os.path.join(self.train_args.out_dir, self.config.name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"saving checkpoint to {out_dir}")
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    def get_lr(self, step):

        min_lr = self.train_args.min_lr
        warmup_iters = self.train_args.warmup_iters
        learning_rate = self.train_args.learning_rate
        lr_decay_iters = self.train_args.lr_decay_iters
        
        if step < warmup_iters:
            return learning_rate * (step + 1) / (warmup_iters + 1)
        
        if step > lr_decay_iters:
            return min_lr
        
        decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
        return min_lr + coeff * (learning_rate - min_lr)
    
    def estimate_loss(self):
        out = {}
        self.model.eval()
        loaders = {
            'train': iter(self.train_loader),
            'val': iter(self.val_loader)
        }

        for split in ['train', 'val']:
            losses = torch.zeros(self.train_args.eval_iters)
            data_iter = loaders[split]
            for k in range(self.train_args.eval_iters):

                try:
                    X, Y = next(data_iter)
                except StopIteration:
                    # Reset the iterator if the dataloader runs out of data
                    if split == 'train':
                        loaders['train'] = iter(self.train_loader)
                    else:
                        loaders['val'] = iter(self.val_loader)
                    data_iter = loaders[split]
                    X, Y = next(data_iter)
                
                X.to(self.device)
                Y.to(self.device)

                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out