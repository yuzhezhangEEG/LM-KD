from torch import nn
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import numpy
import time
import argparse
import random
import os
from sklearn import metrics
import torch
from transformers import GPT2Model, GPT2Config
import hdf5storage
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
seed=123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--EPOCH', type=int, default=1, help='Get the EPOCH')
parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=32, help='Get the TRAIN_BATCH_SIZE')
parser.add_argument('--TEST_BATCH_SIZE', type=int, default=64, help='Get the TEST_BATCH_SIZE')
parser.add_argument('--start_num', type=int, default=1, help='Get the start Subject_num')
parser.add_argument('--end_num', type=int, default=32, help='Get the start Subject_num')
parser.add_argument('--gpu_num', type=int, default=3, help='Get the GPU_NUM')
parser.add_argument('--isbidirectional', type=bool, default=False, help='Get if is bidirectional')
parser.add_argument('--kd_T', default=4.0, type=float, help='T for Temperature scaling')
parser.add_argument('--kd_mode', default='cse', choices=['cse', 'mse'], type=str, help='')
parser.add_argument('--labeltype', default='valence', choices=['valence','arousal','both'], type=str, help='Get the labeltype')
parser.add_argument('--peritype', default='GSR', choices=['EOG', 'EMG', 'GSR', 'Resp', 'Plet', 'Temp', 'All'], type=str, help='Get the peritype')
parser.add_argument('--losstype', default='cross', choices=['cross', 'KL'], type=str, help='Get the losstype')
parser.add_argument('--LR', type=float, default=1e-3, help='Get the LR')
parser.add_argument('--LRtype', default='fix', choices=['fix', 'ReduceLROnPlateau'], type=str, help='Get the LRtype')
parser.add_argument('--dropout', type=float, default=0.2, help='Get the dropout')
parser.add_argument('--regular_coeff', type=float, default=1e-4, help='Get the regularization coefficient')
parser.add_argument('--distill_decay', action='store_true', default=False, help='distillation decay')
parser.add_argument('--gamma', type=float, default=1.0, help='weight for classification')
parser.add_argument('--alpha', type=float, default=1.0, help='weight balance for student loss')
parser.add_argument('--beta', type=float, default=0.0, help='weight balance for other losses')
parser.add_argument('--factor', default=2, type=int)
parser.add_argument('--convs', action='store_true')
parser.add_argument("--hidden_dim", type=int, default=32, help='teacher_model_hidden_dim') #将输入的外周信号和脑电信号映射到的维度
parser.add_argument("--gpt_n_embd", type=int, default=768, help='gpt_paramater') #输入到分类层前及GPT前的特征维度，也是GPT的嵌入维度
parser.add_argument("--gpt_n_layer", type=int, default=12, help='gpt_paramater') #GPT的Transformer层数
parser.add_argument("--gpt_n_head", type=int, default=12, help='gpt_paramater') #GPT的注意力头数
parser.add_argument("--gpt_resid_pdrop", type=float, default=0.1, help='gpt_paramater') # 嵌入、编码器和池化器中所有全连接层的丢失概率
parser.add_argument("--gpt_attn_pdrop", type=float, default=0.1, help='gpt_paramater') #注意力的丢失率
parser.add_argument("--gpt_embd_pdrop", type=float, default=0.1, help='gpt_paramater') # 嵌入的丢失率
parser.add_argument('--gpt_activation_function', default='gelu', choices=['relu', 'silu', 'gelu', 'tanh', 'gelu_new'], type=str, help='gpt_paramater') #激活函数
parser.add_argument("----num_chunks", type=int, default=15, help='--num_chunks')
parser.add_argument("----chunk_len", type=int, default=64, help='--chunk_len')
parser.add_argument("----chunk_stride", type=int, default=32, help='--chunk_stride')
parser.add_argument('--num_layers', type=float, default=2, help='Get the num_layers')
parser.add_argument('--Isinit', action='store_true', default=False, help='Isinit')
parser.add_argument('--Israndom', action='store_true', default=False, help='Israndom')
parser.add_argument('--Isshuffle', action='store_true', default=False, help='Isshuffle')
parser.add_argument('--ACCU_TRAIN_BATCH_SIZE', type=int, default=2048, help='Get the ACCU_TRAIN_BATCH_SIZE')
parser.add_argument('--Ischannelnorm', action='store_true', default=False, help='Ischannelnorm')
args = parser.parse_args()

#教师1模型-------------------------------------------------------------------------------------------
class CrossTransformer(nn.Module):
    def __init__(self, d1, d2, seq_length, feature_dim, outputdim):
        super(CrossTransformer, self).__init__()
        self.fc_map_a = nn.Linear(d1, feature_dim)
        self.fc_map_b = nn.Linear(d2, feature_dim)
        self.layer_norm_a = nn.LayerNorm(normalized_shape=[seq_length, d1])
        self.layer_norm_b = nn.LayerNorm(normalized_shape=[seq_length, d2])
        self.dropout = nn.Dropout(0.1)

        self.feature_dim = feature_dim
        self.seq_length = seq_length

        # Transformer模型通常需要一个位置编码层
        self.pos_encoder = nn.Embedding(seq_length, feature_dim)

        # 定义两个自注意力模块，分别处理两个序列
        self.self_attn1 = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)
        self.self_attn2 = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8)

        # 定义两个互注意力模块，分别处理两个序列的交叉信息
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8)

        # 为了融合两个序列的信息，使用一个线性层将其投影到768维
        self.fusion_linear = nn.Linear(feature_dim * 2, outputdim)


    def forward(self, seq1, seq2):
        seq1 = self.layer_norm_a(seq1)
        seq1 = self.dropout(F.relu(self.fc_map_a(seq1)))

        seq2 = self.layer_norm_b(seq2)
        seq2 = self.dropout(F.relu(self.fc_map_b(seq2)))

        # 添加位置编码
        pos = torch.arange(self.seq_length, device=seq1.device).unsqueeze(0).repeat(seq1.size(0), 1)
        seq1 = seq1 + self.pos_encoder(pos)
        seq2 = seq2 + self.pos_encoder(pos)

        # 自注意力处理
        seq1 = self.self_attn1(seq1.permute(1, 0, 2)).permute(1, 0, 2)
        seq2 = self.self_attn2(seq2.permute(1, 0, 2)).permute(1, 0, 2)

        # 交叉注意力处理
        seq1_att, _ = self.cross_attn1(seq1, seq2, seq2)
        seq2_att, _ = self.cross_attn2(seq2, seq1, seq1)

        # 合并特征
        combined_features = torch.cat([seq1_att, seq2_att], dim=-1)
        combined_features = torch.mean(combined_features, dim=1)  # 融合时序信息

        # 投影到768维
        fused_features = self.fusion_linear(combined_features)

        return fused_features


class finalclassifier(nn.Module):
    def __init__(self, gpt_n_embd, num_chunks, num_classes):
        super(finalclassifier, self).__init__()
        # 定义三个用于分类的线性层
        self.classifier = nn.Sequential(
            nn.Linear(gpt_n_embd*num_chunks, gpt_n_embd*num_chunks//2),
            nn.ReLU(),
            nn.Linear(gpt_n_embd*num_chunks//2, gpt_n_embd*num_chunks//4),
            nn.ReLU(),
            nn.Linear(gpt_n_embd*num_chunks//4, gpt_n_embd),
            nn.ReLU(),
        )
        self.classifier2 = nn.Linear(gpt_n_embd, num_classes)
    def forward(self, fused_features):
        # 分类
        fused_features = fused_features.contiguous().view(fused_features.size(0), -1)
        middle_feature = self.classifier(fused_features)
        logits = self.classifier2(middle_feature)
        return middle_feature, logits

#教师2模型-------------------------------------------------------------------------------------------
class DGCNN(nn.Module):
    def __init__(self, num_electrodes=62, in_channels=5, num_embed=3, k=2, relu_is=1, layers=None, dropout_rate=0.5):
        # num_electrodes(int): The number of electrodes.
        # in_channels(int): The feature dimension of each electrode.
        # num_classes(int): The number of classes to predict.
        # k_(int): The number of graph convolutional layers.
        # relu_is(int): The function we use
        # out_channel(int): The feature dimension of  the graph after GCN.
        super(DGCNN, self).__init__()

        if layers is None:
            layers = [128]
        self.dropout_rate = dropout_rate
        self.layers = layers
        self.k = k
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.num_classes = num_embed
        self.relu_is = relu_is
        # self.get_param()

        self.graphConvs = nn.ModuleList()
        self.graphConvs.append(GraphConv(self.k, self.in_channels, self.layers[0]))
        for i in range(len(self.layers) - 1):
            self.graphConvs.append(GraphConv(self.k, self.layers[i], self.layers[i + 1]))

        self.fc = nn.Linear(self.num_electrodes * self.layers[-1], num_embed, bias=True)
        self.adj = nn.Parameter(torch.Tensor(self.num_electrodes, self.num_electrodes))
        self.adj_bias = nn.Parameter(torch.Tensor(1))
        self.relu = nn.ReLU(inplace=True)
        self.b_relus = nn.ModuleList()
        for i in range(len(self.layers)):
            self.b_relus.append(B1ReLU(self.layers[i]))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.init_weight()



    def init_weight(self):
        nn.init.xavier_uniform_(self.adj)
        nn.init.trunc_normal_(self.adj_bias, mean=0, std=0.1)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        adj = self.relu(self.adj + self.adj_bias)
        lap = laplacian(adj)
        for i in range(len(self.layers)):
            x = self.graphConvs[i](x, lap)
            x = self.dropout(x)
            x = self.b_relus[i](x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class B1ReLU(nn.Module):
    def __init__(self, bias_shape):
        super(B1ReLU, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(1, 1, bias_shape))
        self.relu = nn.ReLU()
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.relu(self.bias + x)

def laplacian(w):
    """
    calculate the laplacian of the adjacency matrix
    :param w: the adjacency matrix
    :return: l: the normalized Laplacian matrix
    """
    # d is the sum of each row of a matrix.
    d = torch.sum(w, dim=1)
    # reciprocal square root of a vector
    d_re = 1 / torch.sqrt(d + 1e-5)
    # create a matrix with the d_re vector as its diagonal elements
    d_matrix = torch.diag_embed(d_re)
    # calculate the laplacian matrix
    lap = torch.eye(d_matrix.shape[0], device=w.device) - torch.matmul(torch.matmul(d_matrix, w), d_matrix)
    return lap


class GraphConv(nn.Module):
    """
    Graph convolution based on Chebyshev polynomials
    """

    def __init__(self, k, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.weight = nn.Parameter(torch.Tensor(k * in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)
        # self.truncated_normal_(self.weight)

    def chebyshev_polynomial(self, x, lap):
        """
        calculate the chebyshev polynomial
        :param x : input x
        :param lap: the input laplacian matrix
        :return: the chebyshev polynomial components
        """
        t = torch.ones(x.shape[0], x.shape[1], x.shape[2]).to(x.device)
        if self.k == 1:
            return t.unsqueeze(1)
        if self.k == 2:
            return torch.cat((t.unsqueeze(1), torch.matmul(lap, x).unsqueeze(1)), dim=1)
        elif self.k > 2:
            # T_0 of chebyshev polynomials, just x (identity matrix multiply x), shape: (batch, ele_channel, in_channel)
            tk_minus_one = x
            # T_1 of chebyshev polynomials, shape: (batch, ele_channel, in_channel)
            tk = torch.matmul(lap, x)
            # add the T_0, T_1, T_2 items to the Chebyshev components, t shape: (batch, 3, ele_channel, in_channel)
            t = torch.cat((t.unsqueeze(1), tk_minus_one.unsqueeze(1), tk.unsqueeze(1)), dim=1)
            for i in range(3, self.k):
                # T_(k-1) and T_(k-2)
                tk_minus_two, tk_minus_one = tk_minus_one, tk
                # calculate the T_(k), shape: (batch, ele_channel, in_channel)
                tk = 2 * torch.matmul(lap, tk_minus_one) - tk_minus_two
                # add the T_k items to the Chebyshev components, shape: (batch, i+1, ele_channel, in_channel)
                t = torch.cat((t, tk.unsqueeze(1)), dim=1)
            return t

    def forward(self, x, lap):
        """
        :param x: (batch_size, ele_channel, in_channel)
        :param lap: the laplacian matrix
        :return: the result of Graph conv
        """
        # obtain the chebyshev polynomial, t shape: (batch, k, ele_channel, in_channel)
        cp = self.chebyshev_polynomial(x, lap)
        # transpose cp to: (batch, ele_channel, in_channel, k)
        cp = cp.permute(0, 2, 3, 1)
        # reshape cp to: (batch, ele_channel, in_channel * k)
        cp = cp.flatten(start_dim=2)
        # perform filter operation of order K
        out = torch.matmul(cp, self.weight)
        return out




#学生模型-------------------------------------------------------------------------------------------
class GRUNeT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isbidirectional, direct_num, num_classes):
        super(GRUNeT, self).__init__()
        self.isbidirectional = isbidirectional
        self.network = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first = True, bidirectional=isbidirectional)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*direct_num, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_classes)
        )
    def forward(self, x):
        #print(x.shape) #(batchsize, timelength, input_size)
        output, hn = self.network(x)
        #print(hn.shape) #(num_layers * num_directions, batchsize, hidden_size)
        h = hn[-(1 + int(self.isbidirectional)):]  # 用最后一个hidden layer的结果
        h = torch.cat(h.split(1), dim=-1).squeeze(0)  # 在上一步操作中，0维中只有一个元素，用squeeze把0维缩掉，变成两维( batch_size, hidden_out)
        output = self.classifier(h)
        # print(h.shape) #(batchsize, hidden_size*num_directions)

        return h, output


#其他函数----------------------------------------------------------------------------------------
class CAMKD(nn.Module):
    def __init__(self):
        super(CAMKD, self).__init__()
        # self.crit_ce = nn.CrossEntropyLoss()
        self.crit_ce = nn.CrossEntropyLoss(reduction='none')
        self.crit_mse = nn.MSELoss(reduction='none')
        # self.crit_mse = nn.MSELoss(reduction='mean')

    def forward(self, trans_feat_s_list, mid_feat_t_list, output_feat_t_list, target):
        bsz = target.shape[0]
        loss_t = [self.crit_ce(logit_t, target) for logit_t in output_feat_t_list]
        num_teacher = len(trans_feat_s_list)
        loss_t = torch.stack(loss_t, dim=0)
        weight = (1.0 - F.softmax(loss_t, dim=0)) / (num_teacher - 1)
        loss_st = []
        for mid_feat_s, mid_feat_t in zip(trans_feat_s_list, mid_feat_t_list):
            tmp_loss_st = self.crit_mse(mid_feat_s, mid_feat_t).reshape(bsz, -1).mean(-1)
            loss_st.append(tmp_loss_st)
        loss_st = torch.stack(loss_st, dim=0)
        loss = torch.mul(weight, loss_st).sum()
        # loss = torch.mul(attention, loss_st).sum()
        loss /= (1.0 * bsz * num_teacher)

        # avg weight
        # loss_st = []
        # for mid_feat_s, mid_feat_t in zip(trans_feat_s_list, mid_feat_t_list):
        #     tmp_loss_st = self.crit_mse(mid_feat_s, mid_feat_t)
        #     loss_st.append(tmp_loss_st)
        # loss_st = torch.stack(loss_st, dim=0)
        # loss = loss_st.mean(0)
        return loss, weight

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, is_ca=False):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        if is_ca:
            loss = (nn.KLDivLoss(reduction='none')(p_s, p_t) * (self.T**2)).sum(-1)
        else:
            loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T**2)
        return loss

def init_weights(m, feature_dim):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=feature_dim ** -0.5)
    elif isinstance(m, nn.MultiheadAttention):
        nn.init.xavier_uniform_(m.in_proj_weight)
        nn.init.constant_(m.in_proj_bias, 0)
        # Separate out the Q, K, V projections into different segments for clarity
        nn.init.xavier_uniform_(m.out_proj.weight)
        nn.init.constant_(m.out_proj.bias, 0)

def find_max_values_and_indices(data_list):
    if not data_list:  # 如果列表为空，返回None
        return None

    max_value = max(data_list)  # 找到列表中的最大值
    indices = [index for index, value in enumerate(data_list) if value == max_value]  # 获取所有最大值的索引

    return max_value, indices

def find_max_value_by_indices(array, indices):
    if not array or not indices:  # 检查输入是否为空
        return None

    # 初始化max_value为第一个索引对应的数组值，并记录这个索引
    max_value = array[indices[0]]
    max_index = indices[0]

    # 遍历索引列表中的剩余部分
    for index in indices[1:]:
        # 检查每个索引对应的数组值是否大于当前的max_value
        if array[index] > max_value:
            max_value = array[index]
            max_index = index

    return max_index, max_value


num_epochs = args.EPOCH
TRAIN_BATCH_SIZE = args.TRAIN_BATCH_SIZE
TEST_BATCH_SIZE = args.TEST_BATCH_SIZE
gpu_num = args.gpu_num
startnum = args.start_num
endnum = args.end_num
labeltype = args.labeltype #0: valence; 1: arousal; 2: both;
peritype = args.peritype # ECG: 32 * 256 : 35 * 256; GSR: 35 * 256 : 36 * 256; Resp: 36 * 256 : 37 * 256; Temp: 37 * 256 : 38 * 256;
dropout = args.dropout
hidden_dim = args.hidden_dim
gpt_n_embd = args.gpt_n_embd
gpt_n_layer = args.gpt_n_layer
gpt_n_head = args.gpt_n_head
gpt_resid_pdrop = args.gpt_resid_pdrop
gpt_attn_pdrop = args.gpt_attn_pdrop
gpt_embd_pdrop = args.gpt_embd_pdrop
gpt_activation_function = args.gpt_activation_function
num_chunks = args.num_chunks
chunk_len = args.chunk_len
chunk_stride = args.chunk_stride
num_layers = args.num_layers
isbidirectional = args.isbidirectional
Isinit = args.Isinit
Israndom = args.Israndom
Isshuffle = args.Isshuffle
alpha = args.alpha
beta = args.beta
ACCU_TRAIN_BATCH_SIZE = args.ACCU_TRAIN_BATCH_SIZE
Ischannelnorm = args.Ischannelnorm

# EOG: 0-256; EMG: 256-512; GSR: 512-640 Resp: 640-768 Plet: 768-896 Temp: 896-1024
frequency = 128
chunk_len = frequency // 2
chunk_stride = frequency // 4
peri_index_list = []
peri_length = 1
if peritype == 'EOG':
    peri_index_list = list(range(32, 34))
    peri_length = 2
elif peritype == 'EMG':
    peri_index_list = list(range(34, 36))
    peri_length = 2
elif peritype == 'GSR':
    peri_index_list = list(range(36, 37))
elif peritype == 'Resp':
    peri_index_list = list(range(37, 38))
elif peritype == 'Plet':
    peri_index_list = list(range(38, 39))
elif peritype == 'Temp':
    peri_index_list = list(range(39, 40))
elif peritype == 'All':
    peri_index_list = list(range(32, 40))
    peri_length = 8

classnum = 2
if labeltype == "both":
    classnum = 4
divide_num = 15   #每个60s的trial，以4s窗口切割，变为15个sample
num_fold = 1
direct_num = 1
if isbidirectional:
    direct_num = 2


string_gpu = "cuda:" + str(gpu_num)
device = torch.device(string_gpu if torch.cuda.is_available() else "cpu")

#读取外周生理信号数据
savepath = '/data/eegdata/Deap/zyz/deap_preprocess_eeg+peripheral_32_40_40_60_128.mat'
if Ischannelnorm:
    savepath = '/data/eegdata/Deap/zyz/deap_preprocess_eeg+peripheral_32_40_40_60_128_channelnorm.mat'
X = scipy.io.loadmat(savepath)['X'] # (32,40,40,60,128)
Y = scipy.io.loadmat(savepath)['Y']  # (32,40,4)
all_train_index_list = []
all_val_index_list = []
all_test_index_list = []
start_index = 0
for i in range(Y.shape[0]):
    Ytemp = 0
    if labeltype == "valence":
        Ytemp = Y[i, :, 0]
    else:
        Ytemp = Y[i, :, 1]
    num_0_list = []
    num_1_list = []
    train_index_list = []
    val_index_list = []
    test_index_list = []
    for j in range(len(Ytemp)):
        if Ytemp[j] == 0:
            num_0_list.append(j)
        else:
            num_1_list.append(j)
    train_num0_num = int(len(num_0_list)/5*4)
    val_num0_num = int(len(num_0_list) / 10)
    test_num0_num = len(num_0_list) - train_num0_num - val_num0_num
    if test_num0_num-val_num0_num>1:
        val_num0_num = val_num0_num + 1
        test_num0_num = test_num0_num - 1
    train_num1_num = int(len(num_1_list) / 5 * 4)
    val_num1_num = int(len(num_1_list) / 10)
    test_num1_num = len(num_1_list) - train_num1_num - val_num1_num
    if test_num1_num-val_num1_num>1:
        val_num1_num = val_num1_num + 1
        test_num1_num = test_num1_num - 1

    train_index_list = num_0_list[:train_num0_num]
    train_index_list.extend(num_1_list[:train_num1_num])
    val_index_list = num_0_list[train_num0_num:train_num0_num+val_num0_num]
    val_index_list.extend(num_1_list[train_num1_num:train_num1_num+val_num1_num])
    test_index_list = num_0_list[-test_num0_num:]
    test_index_list.extend(num_1_list[-test_num1_num:])

    for k in range(len(train_index_list)):
        all_train_index_list.extend(range(start_index + train_index_list[k]*15, start_index + train_index_list[k]*15+15))
    for k in range(len(val_index_list)):
        all_val_index_list.extend(range(start_index + val_index_list[k]*15, start_index + val_index_list[k]*15+15))
    for k in range(len(test_index_list)):
        all_test_index_list.extend(range(start_index + test_index_list[k]*15, start_index + test_index_list[k]*15+15))
    start_index = start_index + 600




a = torch.Tensor(X) # (32,40,40,60,128)
a = a.contiguous().view(a.size(0), a.size(1), a.size(2), divide_num, -1, a.size(-1))  # (32, 40, 40, 15, 4, 128)
a = a.permute(0, 1, 3, 2, 4, 5)  # (32, 40, 15, 40, 4, 128)
a = a.contiguous().view(a.size(0), -1, a.size(-3), a.size(-2) , a.size(-1))  # (32, 600, 40, 4, 128)
a = a.contiguous().view(-1, a.size(-3), a.size(-2) , a.size(-1))  # (32, 600, 40, 4, 128)
X = a.numpy()
print(X.shape)

b = torch.Tensor(Y)  # (32,40,4)
b = b.repeat_interleave(divide_num, dim=1) # (32,600,4)
b = b.contiguous().view(-1, b.size(-1))
Y = b.numpy() # (32,600,4)
print(Y.shape)
if Isshuffle:
    shuffle_indices = numpy.random.permutation(X.shape[0])
    X = X[shuffle_indices]
    Y_shuffled = Y[shuffle_indices]
# datapath1 = '/data1/eegdata/hci/zyz/hci_preprocess_eeg+peripheral_4s_28_XX_38_4_256.mat'
# X = hdf5storage.loadmat(datapath1)['X']  #28(XX,38,10,256)
# Y = hdf5storage.loadmat(datapath1)['Y']  #28(XX,4)

train_num_list = [309, 331, 254, 314, 316, 316, 312, 308, 187, 314, 323, 323, 323, 254, 337, 328, 316, 309, 316, 316, 312, 308, 314, 314, 319, 323, 302, 310]
val_num_list = [46, 41, 46, 48, 36, 44, 50, 42, 46, 48, 32, 32, 33, 46, 34, 32, 39, 45, 36, 44, 50, 42, 30, 48, 49, 32, 51, 35]
test_num_list = [43, 26, 22, 36, 46, 38, 36, 48, 45, 36, 43, 43, 42, 39, 27, 38, 43, 44, 46, 38, 36, 48, 54, 36, 30, 43, 45, 53]
all_acc_student_with_teacher1 = []
all_f1_student_with_teacher1 = []
all_prelist_student_with_teacher1 = []
all_truelist_student_with_teacher1 = []
all_maxindex1 = []
all_acc_student_with_teacher2 = []
all_f1_student_with_teacher2 = []
all_prelist_student_with_teacher2 = []
all_truelist_student_with_teacher2 = []
all_maxindex2 = []
all_acc_student_with_teacher3 = []
all_f1_student_with_teacher3 = []
all_prelist_student_with_teacher3 = []
all_truelist_student_with_teacher3 = []
all_maxindex3 = []
all_acc_student_with_teacher4 = []
all_f1_student_with_teacher4 = []
all_prelist_student_with_teacher4 = []
all_truelist_student_with_teacher4 = []
all_maxindex4 = []
for Subject_num in range(1):
    print("Subject " + str(Subject_num) + "-------------------------")
    X_sub = X  # (600, 40, 4, 128)
    a = torch.Tensor(X_sub)
    a = a.contiguous().view(a.size(0), a.size(1), -1)  # (600, 40, 4*128)
    X_sub = a.numpy()  # (XX,38*256,10)
    Y_sub = Y # (XX,4)
    student_average_acc = 0
    student_acc_list = []
    teacher_average_acc = 0
    teacher_acc_list = []

    student_with_teacher_average_acc_val = []
    student_with_teacher_average_acc = []
    student_with_teacher_average_f1 = []
    student_with_teacher_prelist = []
    student_with_teacher_truelist = []

    alllist = [i for i in range(X_sub.shape[0])]
    train_num = int(X_sub.shape[0]/5*4)  #15360
    val_num = int(X_sub.shape[0]/10)     #1920
    test_num = int(X_sub.shape[0]/10)    #1920

    train_index_list = []
    val_index_list = []
    test_index_list = []
    for group in range(32):
        start_index = group * 600
        train_index_list.extend(range(start_index, start_index + 480))
        val_index_list.extend(range(start_index + 480, start_index + 540))
        test_index_list.extend(range(start_index + 540, start_index + 600))
    print(len(train_index_list)) #15360
    print(len(val_index_list))  #1920
    print(len(test_index_list)) #1920
    #splitfoldlist = split_list(alllist, group_num=num_fold)
    # for index in range(len(splitfoldlist)):
    #     print(splitfoldlist[index])
    print(time.asctime(time.localtime(time.time())))
    for fold in range(num_fold):
        print("\n\n\n\nThis is Fold!!!!!!!!!!------------------------------------", str(fold + 1))
        # testnum = X_sub.shape[0] // num_fold
        # testlist = [i for i in range(testnum * fold, testnum * (fold + 1))]
        # testlist = splitfoldlist[fold]
        #trainlist = [i for i in alllist if i not in testlist]

        Xtrain = numpy.squeeze(X_sub[train_index_list, :, :])  # (XX,38*256,10)
        Xval = numpy.squeeze(X_sub[val_index_list, :, :])  # (XX,38*256,10)
        Xtest = numpy.squeeze(X_sub[test_index_list, :, :])  # (XX,38*256,10)
        if labeltype == "valence":
            Ytrain = numpy.squeeze(Y_sub[train_index_list, 0])  # (XX,) #HCI 2:vlaence 1:arousal
            Yval = numpy.squeeze(Y_sub[val_index_list, 0])  # (XX,) #HCI 2:vlaence 1:arousal
            Ytest = numpy.squeeze(Y_sub[test_index_list, 0])  # (XX,) #HCI 2:vlaence 1:arousal
        elif labeltype == "arousal":
            Ytrain = numpy.squeeze(Y_sub[train_index_list, 1])  # (XX,) #HCI 2:vlaence 1:arousal
            Yval = numpy.squeeze(Y_sub[val_index_list, 1])  # (XX,) #HCI 2:vlaence 1:arousal
            Ytest = numpy.squeeze(Y_sub[test_index_list, 1])  # (XX,) #HCI 2:vlaence 1:arousal
        else:
            ytrain = Y_sub[train_index_list, :]
            ytrain_temp = []
            for i in range(len(ytrain)):
                ytrain_temp.append(2 * ytrain[i, 0] + ytrain[i, 1])  # HCI 2:vlaence 1:arousal
            ytrain = ytrain_temp
            Ytrain = numpy.squeeze(ytrain)

            yval = Y_sub[val_index_list, :]
            yval_temp = []
            for i in range(len(yval)):
                yval_temp.append(2 * yval[i, 0] + yval[i, 1])  # HCI 2:vlaence 1:arousal
            yval = yval_temp
            Yval = numpy.squeeze(yval)

            ytest = Y_sub[test_index_list, :]
            ytest_temp = []
            for i in range(len(ytest)):
                ytest_temp.append(2 * ytest[i, 0] + ytest[i, 1])  # HCI 2:vlaence 1:arousal
            ytest = ytest_temp
            Ytest = numpy.squeeze(ytest)
        truelist_sub = Ytest
        truelist_sub = [int(dd) for dd in truelist_sub]
        print(Xtrain.shape)
        print(Xval.shape)
        print(Xtest.shape)
        train_data = TensorDataset(torch.tensor(Xtrain, dtype=torch.float32), torch.tensor(Ytrain, dtype=torch.long))  # 训练的数据集
        train_loader = DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=Israndom)
        val_data = TensorDataset(torch.tensor(Xval, dtype=torch.float32), torch.tensor(Yval, dtype=torch.long))  # 验证的数据集
        val_loader = DataLoader(dataset=val_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        test_data = TensorDataset(torch.tensor(Xtest, dtype=torch.float32), torch.tensor(Ytest, dtype=torch.long))
        test_loader = DataLoader(dataset=test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        len_train_data = len(train_data)
        len_val_data = len(val_data)
        len_test_data = len(test_data)

        # ------------------------------------------------------------------------------------------------------#


        print('\n\nTraining...')
        num_inputs1 = peri_length
        num_inputs2 = 32
        num_length = 4 * frequency
        # 创建学生网络
        student = GRUNeT(num_inputs1 * frequency, gpt_n_embd, num_layers, isbidirectional,direct_num, classnum).to(device)
        # 创建教师网络1
        gpt_config = GPT2Config(
            vocab_size=50257,  # 词汇表大小设为1，因为我们不处理真正的文本数据
            n_positions=1024,  # 最大序列长度，确保大于或等于你的时间长度T
            n_embd=gpt_n_embd,  # 嵌入层大小，与你的特征维度d相同
            n_layer=gpt_n_layer,  # Transformer层数
            n_head=gpt_n_head,  # 注意力头数
            resid_pdrop=gpt_resid_pdrop,  # 嵌入、编码器和池化器中所有全连接层的丢失概率
            attn_pdrop=gpt_attn_pdrop,  # 嵌入的丢失率
            embd_pdrop=gpt_embd_pdrop,
            activation_function=gpt_activation_function  # 激活函数 可选["relu", "silu", "gelu", "tanh", "gelu_new"]
        )
        # 创建模型
        gpt_model_path = '../../gpt2model'
        gpt_model = GPT2Model.from_pretrained(gpt_model_path, config=gpt_config).to(device)
        gpt_model.eval()  # 设置为评估模式
        learnable_token = torch.randn((1, gpt_n_embd), requires_grad=True).to(device)
        teacher1 = CrossTransformer(num_inputs1, num_inputs2, chunk_len, hidden_dim, gpt_n_embd).to(device)
        finalclass_teacher1 = finalclassifier(gpt_n_embd, num_chunks, classnum).to(device)
        #创建教室网络2
        teacher2 = DGCNN(num_inputs2, num_length, num_embed=gpt_n_embd).to(device)
        finalclass_teacher2 = finalclassifier(gpt_n_embd, 1, classnum).to(device)
        if Isinit:
            print("initializ！！！！")
            teacher1.apply(lambda m: init_weights(m, hidden_dim))
            finalclass_teacher1.apply(lambda m: init_weights(m, hidden_dim))
            finalclass_teacher2.apply(lambda m: init_weights(m, hidden_dim))
            student.apply(lambda m: init_weights(m, hidden_dim))

        optimizer = torch.optim.AdamW([{'params': teacher1.parameters()},{'params': finalclass_teacher1.parameters()},{'params': teacher2.parameters()},{'params': finalclass_teacher2.parameters()},{'params': student.parameters()}], lr=args.LR, weight_decay=args.regular_coeff)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        # criterion = nn.BCEWithLogitsLoss()
        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(args.kd_T)
        criterion_kd = CAMKD()
        criterion_construct = torch.nn.MSELoss(reduction='mean')  # 重构损失
        Best_trainacc_1 = 0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}:')

            train_loss = .0
            train_acc = .0
            teacher1.train()
            finalclass_teacher1.train()
            teacher2.train()
            finalclass_teacher2.train()
            student.train()
            trainstepsum = 0
            accumulation_steps = ACCU_TRAIN_BATCH_SIZE // TRAIN_BATCH_SIZE
            for trainstep, (batch_x, batch_y) in enumerate(train_loader):
                #教师网络1输入处理及计算相关loss--------------------------------------------------------------------------
                batch_x1_teacher1 = batch_x[:, peri_index_list, :]
                batch_x2_teacher1 = batch_x[:, :32, :]
                batch_x1_teacher1 = batch_x1_teacher1.permute(0, 2, 1)  # (XX, 4*128, peri_index_list)
                batch_x1_teacher1 = batch_x1_teacher1.unfold(1, chunk_len,
                                                             chunk_stride)  # (XX, num_chunks, chunk_len, peri_index_list)
                batch_x1_teacher1 = batch_x1_teacher1.permute(0, 1, 3, 2)
                # print(batch_x1.shape)
                batch_x1_teacher1 = batch_x1_teacher1.contiguous().view(-1, batch_x1_teacher1.size(2),
                                                                        batch_x1_teacher1.size(
                                                                            3))  # (XX*num_chunks, chunk_len, peri_index_list)
                batch_x2_teacher1 = batch_x2_teacher1.permute(0, 2, 1)  # (XX, 4*128, 32)
                batch_x2_teacher1 = batch_x2_teacher1.unfold(1, chunk_len,
                                                             chunk_stride)  # (XX, num_chunks, chunk_len, peri_index_list)
                batch_x2_teacher1 = batch_x2_teacher1.permute(0, 1, 3, 2)
                # print(batch_x2.shape)
                batch_x2_teacher1 = batch_x2_teacher1.contiguous().view(-1, batch_x2_teacher1.size(2),
                                                                        batch_x2_teacher1.size(
                                                                            3))  # (XX*num_chunks, chunk_len, peri_index_list)
                batch_x1_teacher1, batch_x2_teacher1, batch_y= batch_x1_teacher1.to(device), batch_x2_teacher1.to(device), batch_y.to(device)


                fusedfeature = teacher1(batch_x1_teacher1, batch_x2_teacher1)
                fusedfeature = fusedfeature.contiguous().view(-1, num_chunks, fusedfeature.size(-1))
                # print(fusedfeature.shape)
                losses = []
                for i in range(2, num_chunks + 1):  # 从第二个时间点开始蒙版
                    masked_data = fusedfeature.clone()  # 复制原始数据
                    masked_data[:, i:, :] = 0  # 从第 i 个时间点开始置零
                    masked_data[:, i - 1, :] = learnable_token  # 将第 i 个时间点替换为可学习向量M

                    # 获取模型的输出
                    outputs = gpt_model(inputs_embeds=masked_data)
                    last_hidden_states = outputs.last_hidden_state  # 获取最后隐藏层状态

                    # 选取第 i 个时间点的隐藏状态进行损失计算
                    predicted_hidden_state = last_hidden_states[:, i - 1, :]
                    actual_hidden_state = fusedfeature[:, i - 1, :]  # 真实的隐藏状态
                    loss = criterion_construct(predicted_hidden_state,
                                     actual_hidden_state.detach())  # 计算损失，确保不计算actual_hidden_state的梯度
                    losses.append(loss)
                # 汇总所有重构损失
                construct_loss_teacher1 = sum(losses) / (num_chunks - 1)
                feat_t1, logit_t1 =  finalclass_teacher1(fusedfeature)
                class_loss_teacher1 = criterion_cls(logit_t1, batch_y)
                loss_teacher1 = class_loss_teacher1 + construct_loss_teacher1

                #教师网络2输入处理及计算相关loss-------------------------------------------------------------------------
                batch_x2_teacher2 = batch_x[:, :32, :]
                batch_x2_teacher2 = batch_x2_teacher2.to(device)
                feat_t2 = teacher2(batch_x2_teacher2)
                _, logit_t2 = finalclass_teacher2(feat_t2)
                loss_teacher2 = criterion_cls(logit_t2, batch_y)


                #学生网络输入处理及计算相关loss--------------------------------------------------------------------------
                batch_x1_student = batch_x[:, peri_index_list, :]
                batch_x1_student = batch_x1_student.contiguous().view(batch_x1_student.size(0),
                                                                      batch_x1_student.size(1), -1, frequency)
                batch_x1_student = batch_x1_student.permute(0, 2, 1, 3)  # (XX, 4, peri_index_list, frequency)
                batch_x1_student = batch_x1_student.contiguous().view(batch_x1_student.size(0),
                                                                      batch_x1_student.size(1), -1)
                batch_x1_student = batch_x1_student.to(device)
                feat_s, logit_s = student(batch_x1_student)
                loss_student = criterion_cls(logit_s, batch_y)

                #计算蒸馏相关loss-----------------------------------------------------------------------------------

                criterion_cls_lc = nn.CrossEntropyLoss(reduction='none')
                loss_t_list = [criterion_cls_lc(logit_t1, batch_y), criterion_cls_lc(logit_t2, batch_y)]
                loss_t = torch.stack(loss_t_list, dim=0)
                attention = (1.0 - F.softmax(loss_t, dim=0))
                # logist 蒸馏
                loss_dist1_list1 = [criterion_div(logit_s, logit_t1, is_ca=True),
                                 criterion_div(logit_s, logit_t2, is_ca=True)]
                loss_dist1 = torch.stack(loss_dist1_list1, dim=0)
                bsz1 = loss_dist1.shape[1]
                loss_dist1 = (torch.mul(attention, loss_dist1).sum()) / (1.0 * bsz1 * 2)

                # middle_feature 蒸馏
                # loss_dist_list2 = [criterion_div(feat_s, feat_t1, is_ca=True),
                #                   criterion_div(feat_s, feat_t2, is_ca=True)]
                # loss_dist2 = torch.stack(loss_dist_list2, dim=0)
                # bsz2 = loss_dist2.shape[1]
                # loss_dist2 = (torch.mul(attention, loss_dist2).sum()) / (1.0 * bsz1 * 2)



                # mid_feat_t_list = [feat_t1, feat_t2]
                # # cal_weight = CalWeight(feat_s, mid_feat_t_list, args)
                # # trans_feat_s_list, output_feat_t_list = cal_weight(feat_s, mid_feat_t_list, model_t_list)
                # trans_feat_s_list = [feat_s, feat_s]
                # output_feat_t_list = [class_t1(feat_s), class_t2(feat_s)]
                # loss_kd, weight = criterion_kd(trans_feat_s_list, mid_feat_t_list, output_feat_t_list, batch_y)


                #计算所有loss---------------------------------------------------------------------------------------
                #loss = loss_teacher1 + loss_teacher2 + loss_student + loss_dist1 + loss_dist2
                loss = loss_teacher1 + loss_teacher2 + alpha*loss_student + loss_dist1
                loss = loss / accumulation_steps
                loss.backward()
                if (trainstep + 1) % accumulation_steps == 0:
                    optimizer.step()  # 更新参数
                    optimizer.zero_grad()  # 清空梯度

                train_loss += loss.data.item()
                trainpred_y = (torch.max(logit_s, 1)[1]).cpu().numpy()  # torch.max(test_out,1)返回的是test_out中每一行最大的数)
                accuracy0 = (trainpred_y == batch_y.squeeze().cpu().numpy()).astype(int).sum()
                train_acc = train_acc + accuracy0
                # train_acc += ((torch.sigmoid(out)>=0.5) == batch_y).sum().data.item()
                trainstepsum = trainstepsum + 1
            print('Train loss: {:.6f}, Train acc: {:.6f}'.format(train_loss / trainstepsum, train_acc / len_train_data))
            if args.LRtype == 'ReduceLROnPlateau':
                scheduler.step(train_loss / trainstepsum)
            if (train_acc / len_train_data) > Best_trainacc_1:
                Best_trainacc_1 = train_acc / len_train_data
            if (epoch + 1) % 1 == 0:
                teacher1.eval()
                finalclass_teacher1.eval()
                teacher2.train()
                finalclass_teacher2.train()
                student.eval()
                with torch.no_grad():
                    #验证集
                    valstepsum = 0
                    eval_loss = .0
                    eval_acc = .0
                    for trainstep, (batch_x, batch_y) in enumerate(val_loader):
                        batch_x1_student = batch_x[:, peri_index_list, :]
                        batch_x1_student = batch_x1_student.contiguous().view(batch_x1_student.size(0),
                                                                              batch_x1_student.size(1), -1, frequency)
                        batch_x1_student = batch_x1_student.permute(0, 2, 1, 3)  # (XX, 4, peri_index_list, frequency)
                        batch_x1_student = batch_x1_student.contiguous().view(batch_x1_student.size(0),
                                                                              batch_x1_student.size(1), -1)
                        batch_x1_student, batch_y = batch_x1_student.to(device), batch_y.to(device)

                        feat_s, logit_s = student(batch_x1_student)
                        loss_student = criterion_cls(logit_s, batch_y)
                        loss = loss_student

                        eval_loss += loss.data.item()
                        testpred_y = (
                        torch.max(logit_s, 1)[1]).cpu().numpy()  # torch.max(test_out,1)返回的是test_out中每一行最大的数)
                        accuracy0 = (testpred_y == batch_y.squeeze().cpu().numpy()).astype(int).sum()
                        eval_acc = eval_acc + accuracy0
                        valstepsum = valstepsum + 1
                    print('Eval loss: {:.6f}, Eval acc: {:.6f}'.format(eval_loss / valstepsum,
                                                                       eval_acc / len_val_data))
                    #print(eval_acc / len_test_data)
                    student_with_teacher_average_acc_val.append(eval_acc / len_val_data)

                    #测试集
                    teststepsum = 0
                    test_loss = .0
                    test_acc = .0
                    prelist_temp = []
                    for trainstep, (batch_x, batch_y) in enumerate(test_loader):
                        batch_x1_student = batch_x[:, peri_index_list, :]
                        batch_x1_student = batch_x1_student.contiguous().view(batch_x1_student.size(0),
                                                                              batch_x1_student.size(1), -1, frequency)
                        batch_x1_student = batch_x1_student.permute(0, 2, 1, 3)  # (XX, 4, peri_index_list, frequency)
                        batch_x1_student = batch_x1_student.contiguous().view(batch_x1_student.size(0),
                                                                              batch_x1_student.size(1), -1)
                        batch_x1_student, batch_y = batch_x1_student.to(device), batch_y.to(device)

                        feat_s, logit_s = student(batch_x1_student)
                        loss_student = criterion_cls(logit_s, batch_y)
                        loss = loss_student

                        test_loss += loss.data.item()
                        testpred_y = (torch.max(logit_s, 1)[1]).cpu().numpy()  # torch.max(test_out,1)返回的是test_out中每一行最大的数)
                        accuracy0 = (testpred_y == batch_y.squeeze().cpu().numpy()).astype(int).sum()
                        test_acc = test_acc + accuracy0
                        teststepsum = teststepsum + 1
                        prelist_temp.extend(testpred_y)
                    print('Test loss: {:.6f}, Test acc: {:.6f}'.format(test_loss / teststepsum, test_acc / len_test_data))
                    accacc = metrics.accuracy_score(truelist_sub, prelist_temp)
                    #print(test_acc / len_test_data, accacc)
                    f1f1 = metrics.f1_score(truelist_sub, prelist_temp, average='weighted')

                    student_with_teacher_average_acc.append(accacc)
                    student_with_teacher_average_f1.append(f1f1)
                    student_with_teacher_prelist.append(prelist_temp)
                    student_with_teacher_truelist.append(truelist_sub)


        print("Fold:", str(fold + 1), "student_with_teacher Train Best Acc:", Best_trainacc_1)

    print("--------------------------------------------------------------")
    # 以测试集准确率最大为准：
    max_value_acc1, indices = find_max_values_and_indices(student_with_teacher_average_acc)
    max_index, max_value_f11 = find_max_value_by_indices(student_with_teacher_average_f1, indices)
    max_prelist1 = student_with_teacher_prelist[max_index]
    max_truelist1 = student_with_teacher_truelist[max_index]
    max_index1 = max_index
    print("Best Test ACC-based:")
    print("max_acc:", max_value_acc1)
    print("max_f1:", max_value_f11)
    print("max_prelist:", max_prelist1)
    print("max_truelist:", max_truelist1)
    print("max_index:", max_index1)

    # 以测试集F1最大为准：
    max_value_f12, indices = find_max_values_and_indices(student_with_teacher_average_f1)
    max_index, max_value_acc2 = find_max_value_by_indices(student_with_teacher_average_acc, indices)
    max_prelist2 = student_with_teacher_prelist[max_index]
    max_truelist2 = student_with_teacher_truelist[max_index]
    max_index2 = max_index
    print("Best Test F1-based:")
    print("max_acc:", max_value_acc2)
    print("max_f1:", max_value_f12)
    print("max_prelist:", max_prelist2)
    print("max_truelist:", max_truelist2)
    print("max_index:", max_index2)

    # 以验证集ACC最大为准：如果相同优先测试集ACC高
    max_value_acc_val, indices = find_max_values_and_indices(student_with_teacher_average_acc_val)
    max_index, max_value_acc3 = find_max_value_by_indices(student_with_teacher_average_acc, indices)
    max_value_f13 = student_with_teacher_average_f1[max_index]
    max_prelist3 = student_with_teacher_prelist[max_index]
    max_truelist3 = student_with_teacher_truelist[max_index]
    max_index3 = max_index
    print("Best Val ACC-based + Best Test ACC:")
    print("max_acc:", max_value_acc3)
    print("max_f1:", max_value_f13)
    print("max_prelist:", max_prelist3)
    print("max_truelist:", max_truelist3)
    print("max_index:", max_index3)

    # 以验证集ACC最大为准：如果相同优先测试集F1高
    max_value_acc_val, indices = find_max_values_and_indices(student_with_teacher_average_acc_val)
    max_index, max_value_f14 = find_max_value_by_indices(student_with_teacher_average_f1, indices)
    max_value_acc4 = student_with_teacher_average_acc[max_index]
    max_prelist4 = student_with_teacher_prelist[max_index]
    max_truelist4 = student_with_teacher_truelist[max_index]
    max_index4 = max_index
    print("Best Val ACC-based + Best Test F1:")
    print("max_acc:", max_value_acc4)
    print("max_f1:", max_value_f14)
    print("max_prelist:", max_prelist4)
    print("max_truelist:", max_truelist4)
    print("max_index:", max_index4)

    # ------------------------------------------------------------------------------------------------------#
    print("---------------------------------------------------------------------")
    print(time.asctime(time.localtime(time.time())))
    all_acc_student_with_teacher1.append(max_value_acc1)
    all_f1_student_with_teacher1.append(max_value_f11)
    all_truelist_student_with_teacher1.append(max_truelist1)
    all_prelist_student_with_teacher1.append(max_prelist1)
    all_maxindex1.append(max_index1)
    print("Best Test ACC-based:")
    print("Now all_acc_student_with_teacher1:", all_acc_student_with_teacher1)
    print("Now all_f1_student_with_teacher1:", all_f1_student_with_teacher1)
    print("Now all_truelist_student_with_teacher1:", all_truelist_student_with_teacher1)
    print("Now all_prelist_student_with_teacher1:", all_prelist_student_with_teacher1)
    print("Now all_maxindex1:", all_maxindex1)

    all_acc_student_with_teacher2.append(max_value_acc2)
    all_f1_student_with_teacher2.append(max_value_f12)
    all_truelist_student_with_teacher2.append(max_truelist2)
    all_prelist_student_with_teacher2.append(max_prelist2)
    all_maxindex2.append(max_index2)
    print("Best Test F1-based:")
    print("Now all_acc_student_with_teacher2:", all_acc_student_with_teacher2)
    print("Now all_f1_student_with_teacher2:", all_f1_student_with_teacher2)
    print("Now all_truelist_student_with_teacher2:", all_truelist_student_with_teacher2)
    print("Now all_prelist_student_with_teacher2:", all_prelist_student_with_teacher2)
    print("Now all_maxindex2:", all_maxindex2)

    all_acc_student_with_teacher3.append(max_value_acc3)
    all_f1_student_with_teacher3.append(max_value_f13)
    all_truelist_student_with_teacher3.append(max_truelist3)
    all_prelist_student_with_teacher3.append(max_prelist3)
    all_maxindex3.append(max_index3)
    print("Best Val ACC-based + Best Test ACC:")
    print("Now all_acc_student_with_teacher3:", all_acc_student_with_teacher3)
    print("Now all_f1_student_with_teacher3:", all_f1_student_with_teacher3)
    print("Now all_truelist_student_with_teacher3:", all_truelist_student_with_teacher3)
    print("Now all_prelist_student_with_teacher3:", all_prelist_student_with_teacher3)
    print("Now all_maxindex3:", all_maxindex3)

    all_acc_student_with_teacher4.append(max_value_acc4)
    all_f1_student_with_teacher4.append(max_value_f14)
    all_truelist_student_with_teacher4.append(max_truelist4)
    all_prelist_student_with_teacher4.append(max_prelist4)
    all_maxindex4.append(max_index4)
    print("Best Val ACC-based + Best Test F1:")
    print("Now all_acc_student_with_teacher4:", all_acc_student_with_teacher4)
    print("Now all_f1_student_with_teacher4:", all_f1_student_with_teacher4)
    print("Now all_truelist_student_with_teacher4:", all_truelist_student_with_teacher4)
    print("Now all_prelist_student_with_teacher4:", all_prelist_student_with_teacher4)
    print("Now all_maxindex4:", all_maxindex4)
# print("Final all_acc_student:", all_acc_student)
# print("Final all_acc_teacher:", all_acc_teacher)

print("---------------------------------------------------------------------")
print("Best Test ACC-based:")
print("Final all_acc_student_with_teacher1:", all_acc_student_with_teacher1)
print("Final all_f1_student_with_teacher1:", all_f1_student_with_teacher1)
print("Final all_truelist_student_with_teacher1:", all_truelist_student_with_teacher1)
print("Final all_prelist_student_with_teacher1:", all_prelist_student_with_teacher1)
print("Final all_maxindex1:", all_maxindex1, max(all_maxindex1))
print(sum(all_acc_student_with_teacher1) / len(all_acc_student_with_teacher1))
print(sum(all_f1_student_with_teacher1) / len(all_f1_student_with_teacher1))

print("---------------------------------------------------------------------")
print("Best Test F1-based:")
print("Final all_acc_student_with_teacher2:", all_acc_student_with_teacher2)
print("Final all_f1_student_with_teacher2:", all_f1_student_with_teacher2)
print("Final all_truelist_student_with_teacher2:", all_truelist_student_with_teacher2)
print("Final all_prelist_student_with_teacher2:", all_prelist_student_with_teacher2)
print("Final all_maxindex2:", all_maxindex2, max(all_maxindex2))
print(sum(all_acc_student_with_teacher2) / len(all_acc_student_with_teacher2))
print(sum(all_f1_student_with_teacher2) / len(all_f1_student_with_teacher2))

print("---------------------------------------------------------------------")
print("Best Val ACC-based + Best Test ACC:")
print("Final all_acc_student_with_teacher3:", all_acc_student_with_teacher3)
print("Final all_f1_student_with_teacher3:", all_f1_student_with_teacher3)
print("Final all_truelist_student_with_teacher3:", all_truelist_student_with_teacher3)
print("Final all_prelist_student_with_teacher3:", all_prelist_student_with_teacher3)
print("Final all_maxindex3:", all_maxindex3, max(all_maxindex3))
print(sum(all_acc_student_with_teacher3) / len(all_acc_student_with_teacher3))
print(sum(all_f1_student_with_teacher3) / len(all_f1_student_with_teacher3))

print("---------------------------------------------------------------------")
print("Best Val ACC-based + Best Test F1:")
print("Final all_acc_student_with_teacher4:", all_acc_student_with_teacher4)
print("Final all_f1_student_with_teacher4:", all_f1_student_with_teacher4)
print("Final all_truelist_student_with_teacher4:", all_truelist_student_with_teacher4)
print("Final all_prelist_student_with_teacher4:", all_prelist_student_with_teacher4)
print("Final all_maxindex4:", all_maxindex4, max(all_maxindex4))
print(sum(all_acc_student_with_teacher4) / len(all_acc_student_with_teacher4))
print(sum(all_f1_student_with_teacher4) / len(all_f1_student_with_teacher4))