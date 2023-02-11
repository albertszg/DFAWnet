import torch
import os
import numpy as np
import random
def seed_torch(seed=1029):
	random.seed(seed)
	# os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)# 为CPU设置随机种子
	torch.cuda.manual_seed(seed)# 为当前GPU设置随机种子
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	print('Setting the seed_torch done!')
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.deterministic = True
