import os

class Params:
    def __init__(self, exp_name='prodict_review_prediction', random_seed=1337, all_data_size=40_000, train_frac=0.5):
        self.random_seed = random_seed
        self.exp_name = exp_name
        self.all_data_size = all_data_size
        self.train_frac = train_frac  # Test and val split 50/50
        
    def __str__(self):
        return ", ".join(f"{k}: {v}" for k, v in vars(self).items())
    
params = Params()

current_train_number = 1
def create_dirs():
    global current_train_number
    if not os.path.exists('./runs'):
        os.makedirs('./runs')

    while os.path.exists('./runs/train' + str(current_train_number)):
        current_train_number += 1
    
    os.makedirs('./runs/train' + str(current_train_number))
    
    
def get_output_path():
    return './runs/train' + str(current_train_number) + '/'

def save_params(params, save_path, min_loss=None):
    with open(save_path, 'w') as f:
        f.write(str(params))
        
        if min_loss is not None:
            f.write('\nMinimal loss: ' + str(min_loss))