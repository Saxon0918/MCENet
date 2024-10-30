import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train
from src.preprocess_data import ADNI_Dataset, ADNI_three_Dataset

parser = argparse.ArgumentParser(description='Alzheimer Analysis')
parser.add_argument('-f', default='', type=str)

# Model
parser.add_argument('--model', type=str, default='MCIE')

# hyperparameter
parser.add_argument('--nlevels', type=int, default=2, help='the layers of cross-modal information enhancement module')
parser.add_argument('--num_heads', type=int, default=8, help='number of heads')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--clip', type=float, default=0.8, help='gradient clip value')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use')
parser.add_argument('--num_epochs', type=int, default=40, help='number of epochs')
parser.add_argument('--when', type=int, default=10, help='decay learning rate')
parser.add_argument('--batch_chunk', type=int, default=1, help='number of chunks per batch')
parser.add_argument('--log_interval', type=int, default=4, help='frequency of result logging')
parser.add_argument('--seed', type=int, default=11111, help='random seed')
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--name', type=str, default='MCIE')

# Cross-Modal Information Enhancement Module
parser.add_argument('--mri', default=True, help='use Cross-Modal Information Enhancement Module to enhance MRI')
parser.add_argument('--av45', default=True, help='use Cross-Modal Information Enhancement Module to enhance AV45-PET')
parser.add_argument('--fdg', default=True, help='use Cross-Modal Information Enhancement Module to enhance FDG-PET')
parser.add_argument('--gene', default=True, help='use Cross-Modal Information Enhancement Module to enhance Gene')
parser.add_argument('--attn_dropout', type=float, default=0.1, help='attention dropout')
parser.add_argument('--relu_dropout', type=float, default=0.1, help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25, help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1, help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.1, help='output layer dropout')
args = parser.parse_args()
torch.manual_seed(args.seed)

# CUDA
use_cuda = False
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

# ---------------------Load dataset---------------------
train_data = ADNI_Dataset(random_seed=args.seed, mode='train')
valid_data = ADNI_Dataset(random_seed=args.seed, mode='val')
test_data = ADNI_Dataset(random_seed=args.seed, mode='test')

# three class
# train_data = ADNI_three_Dataset(random_seed=args.seed, mode='train')
# valid_data = ADNI_three_Dataset(random_seed=args.seed, mode='val')
# test_data = ADNI_three_Dataset(random_seed=args.seed, mode='test')

if torch.cuda.is_available():
    g_cuda = torch.Generator(device='cuda')
else:
    g_cuda = torch.Generator()
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=g_cuda)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, generator=g_cuda)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, generator=g_cuda)

# ---------------------Model hyperparameters---------------------
hyp_params = args
hyp_params.orig_d_m, hyp_params.orig_d_a, hyp_params.orig_d_f, hyp_params.orig_d_g = 140, 140, 140, 100
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = args.model
hyp_params.output_dim = 1
hyp_params.criterion = 'L1Loss'
# hyp_params.criterion = 'CrossEntropyLoss'
# this is test
#sda
if __name__ == '__main__':
    test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
