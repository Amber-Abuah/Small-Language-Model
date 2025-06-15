class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.val_losses = []

    def append_val_loss(self, val_loss):
        self.val_losses.append(val_loss)

    def should_stop_early(self):
        if(len(self.val_losses)) < self.patience + 1:
            return False
        
        comparison = [self.val_losses[-i] + self.min_delta < self.val_losses[-i-1] for i in range(1, self.patience + 1)]
        return True not in comparison