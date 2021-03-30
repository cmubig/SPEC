
class Arguments():
    def __init__(self,token):
        self.dataset_name = 'zara1'
        self.delim = '\t'
        self.loader_num_workers = 4
        self.min_ped = 2
        self.hist_len = 8
        self.fut_len = 12
        self.loadNpy = 0
        self.untracked_ratio = 1.0
        # Network design
        self.l2d = 1
        self.tanh = 1
        self.n_ch = 2
        self.use_max = 1
        self.targ_ker_num = [50,80] # [7,28]
        self.targ_ker_size = [2,2]
        self.targ_pool_size = [2,2]
        self.cont_ker_num = [-1,160]
        self.cont_ker_size = [2,2]
        self.cont_pool_size = [2,2]
        self.n_fc = -1
        self.fc_width = [300,120,80,5] # 280,200,120,80
        self.output_size = 11
        self.neighbor = 1
        self.drop_rate = 0.0
        self.lock_l2d = 0
        # HyperParam
        self.seed = 1
        self.ave_spd = 0.5
        self.loss_balance = 75.0
        # Training
        self.loadModel = ''
        self.batch_size = 64
        self.n_epoch = 100
        self.n_iteration = 300
        self.lr = 0.001
        self.start = 0
        # Validation and Output
        self.batch_size_val = 2
        self.batch_size_tst = 2
        self.n_batch_val = 6
        self.n_batch_tst = 4
        self.val_freq = 1
        self.n_guess = 2
        self.n_sample = 20
        self.coef = 1.000000001
        self.task_name = "ut"
        self.plotting_weights=0

        self.token = token


        if token=='hotel':
            self.dataset_name = 'hotel'
            self.n_epoch = 5

        elif token == '':
            pass


        else:
            print("no token!!!")

        print("task:", self.token, self.dataset_name, self.task_name)
