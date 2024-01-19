class config:
    def __init__(self,):
        # basics
        self.batch_size = 2
        self.num_epochs = 10
        self.val_freq = 1
        self.num_workers = 0
        self.exp_name = 'Trial_0'
        self.optim_settings()
        return

    def optim_settings(self,):
        self.lr = 1e-4 
        self.lr_decay = 0.5
        self.step = 5
        