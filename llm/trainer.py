import torch
import math

class Trainer:

    def __init__(self, model, optimizer, train_loader, val_loader, train_args, device):
        
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_args = train_args
        self.device = device

        self.model.to(self.device)

    def fit(self):

        self.model.train()

        step = 0
        data_iter = iter(self.train_loader)
        while step < self.train_args.max_iters:

            lr = self.get_lr(step) if self.train_args.decay_lr else self.train_args.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if step % self.train_args.eval_interval == 0:
                losses = self.estimate_loss()
                print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

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