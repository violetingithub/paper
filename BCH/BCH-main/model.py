import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo, coo_matrix
import time
import random
from numba import jit
import heapq

init = nn.init.xavier_uniform_


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size=100):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset

    def forward(self, adjacency, embedding):
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(item_embeddings)
      #  final1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in final]))
      #  item_embeddings = torch.sum(final1, 0)
        item_embeddings = np.sum(final, 0) / (self.layers+1)
        return item_embeddings


class LineConv(Module):
    def __init__(self, layers, batch_size,emb_size=100):
        super(LineConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers

    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.cuda.FloatTensor(1,self.emb_size).fill_(0)
        # zeros = torch.zeros([1,self.emb_size])
        item_embedding = torch.cat([zeros, item_embedding], 0)
        seq_h = []
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), session_len)
        session = [session_emb_lgcn]
        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)
        #session1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in session]))
        #session_emb_lgcn = torch.sum(session1, 0)
        session_emb_lgcn = np.sum(session, 0)/ (self.layers+1)
        return session_emb_lgcn


class BCH(Module):
    def __init__(self, adjacency, n_node, lr, layers, l2, beta, lam, eps, dataset, n_trm_layers, hidden_dropout_prob,
                 n_attention_heads, attention_probs_dropout_prob, emb_size=100, batch_size=100):
        super(BCH, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.dataset = dataset
        self.lam = lam
        self.eps = eps
        self.K = 10
        self.w_k = 10
        self.num = 5000

        self.n_trm_layers = n_trm_layers
        self.hidden_dropout_prob = hidden_dropout_prob
        self.n_attention_heads = n_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob


        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        if dataset == 'Nowplaying':
            index_fliter = (values < 0.05).nonzero()
            values = np.delete(values, index_fliter)
            indices1 = np.delete(indices[0], index_fliter)
            indices2 = np.delete(indices[1], index_fliter)
            indices = [indices1, indices2]
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.adjacency = adjacency
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_len = 200
        if self.dataset =='retailrocket':
            self.pos_len= 300
        self.pos_embedding = nn.Embedding(self.pos_len, self.emb_size)
        self.HyperGraph = HyperConv(self.layers, dataset)
        self.LineGraph = LineConv(self.layers, self.batch_size)
        self.sasrec = SASRec(self.n_trm_layers, self.hidden_dropout_prob, self.n_attention_heads, self.attention_probs_dropout_prob, self.emb_size)
        #def __init__(self, n_trm_layers, hidden_dropout_prob, n_attention_heads, attention_probs_dropout_prob,emb_size=100):
        self.w_1 = nn.Linear(2 * self.emb_size, self.emb_size)
        # retailrocket使用
        # self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.adv_item = torch.cuda.FloatTensor(self.n_node, self.emb_size).fill_(0).requires_grad_(True)
        self.adv_sess = torch.cuda.FloatTensor(self.n_node, self.emb_size).fill_(0).requires_grad_(True)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

     
    def generate_sess_emb(self,item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        # [batch_size, seq_len, emb_size]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)

        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]

        # [batch_size, seq_len, emb_size]
        pos_emb = self.pos_embedding.weight[:len]

        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)  # [batch_size, seq_len, emb_size]


        hs = hs.unsqueeze(-2).repeat(1, len, 1)

        # [batch_size, seq_len, 2*emb_size]；w_1后[batch_size, seq_len, emb_size]
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))  # [batch_size, seq_len, emb_size]

        # retailrocket
        # nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))

        # [batch_size, seq_len, 1]

        beta = torch.matmul(nh, self.w_2)

        beta = beta * mask  # [batch_size, seq_len, emb_size]

        #select = torch.sum(beta * seq_h, 1)  # [batch_size, emb_size]
        select = beta * seq_h

        return select

    def generate_sess_emb_npos(self,item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        # pos_emb = self.pos_embedding.weight[:len]
        # pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = seq_h
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))

        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = beta * seq_h
        # select = torch.sum(beta * seq_h, 1)
        return select



    def example_predicting(self, item_emb, sess_emb):
        x_u = torch.matmul(item_emb, sess_emb)
        pos = torch.softmax(x_u, 0)
        return pos

    def adversarial_item(self, item_emb, tar, sess_emb):
        adv_item_emb = item_emb + self.adv_item
        score = torch.mm(sess_emb, torch.transpose(adv_item_emb, 1, 0))
        loss = self.loss_function(score, tar)
        grad = torch.autograd.grad(loss, self.adv_item, retain_graph=True)[0]
        adv = grad.detach()
        self.adv_item = (F.normalize(adv, p=2, dim=1) * self.eps).requires_grad_(True)

    def adversarial_sess(self, item_emb, tar, sess_emb):
        adv_item_emb = item_emb + self.adv_sess
        score = torch.mm(sess_emb, torch.transpose(adv_item_emb, 1, 0))
        loss = self.loss_function(score, tar)
        grad = torch.autograd.grad(loss, self.adv_sess, retain_graph=True)[0]
        adv = grad.detach()
        self.adv_sess = (F.normalize(adv, p=2, dim=1)*self.eps).requires_grad_(True)

    def diff(self, score_item, score_sess, score_adv2, score_adv1, diff_mask):
        score_item = F.softmax(score_item, dim=1)
        score_sess = F.softmax(score_sess, dim=1)
        score_adv2 = F.softmax(score_adv2, dim=1)
        score_adv1 = F.softmax(score_adv1, dim=1)
        score_item = torch.mul(score_item, diff_mask)
        score_sess = torch.mul(score_sess, diff_mask)
        score_adv2 = torch.mul(score_adv2, diff_mask)
        score_adv1 = torch.mul(score_adv1, diff_mask)

        h1 = torch.sum(torch.mul(score_item, torch.log(1e-8 + ((score_item + 1e-8) / (score_adv2 + 1e-8)))))
        h2 = torch.sum(torch.mul(score_sess, torch.log(1e-8 + ((score_sess + 1e-8) / (score_adv1 + 1e-8)))))

        return h1 + h2

    def SSL_topk(self, anchor, sess_emb, pos, neg):
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 2)

        anchor = F.normalize(anchor + sess_emb, p=2, dim=-1)
        pos = torch.reshape(pos, (self.batch_size, self.K, self.emb_size)) + sess_emb.unsqueeze(1).repeat(1, self.K, 1)
        neg = torch.reshape(neg, (self.batch_size, self.K, self.emb_size)) + sess_emb.unsqueeze(1).repeat(1, self.K, 1)
        pos_score = score(anchor.unsqueeze(1).repeat(1, self.K, 1), F.normalize(pos, p=2, dim=-1))
        neg_score = score(anchor.unsqueeze(1).repeat(1, self.K, 1), F.normalize(neg, p=2, dim=-1))
        pos_score = torch.sum(torch.exp(pos_score / 0.2), 1)
        neg_score = torch.sum(torch.exp(neg_score / 0.2), 1)
        con_loss = -torch.sum(torch.log(pos_score / (pos_score + neg_score)))
        return con_loss

    def topk_func_random(self, score1, score2, item_emb_H, item_emb_L):
        values, pos_ind_H = score1.topk(self.num, dim=0, largest=True, sorted=True)
        values, pos_ind_L = score2.topk(self.num, dim=0, largest=True, sorted=True)
        pos_emb_H = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        pos_emb_L = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        neg_emb_H = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        neg_emb_L = torch.cuda.FloatTensor(self.K, self.batch_size, self.emb_size).fill_(0)
        for i in torch.arange(self.K):
            pos_emb_L[i] = item_emb_L[pos_ind_H[i]]
            pos_emb_H[i] = item_emb_H[pos_ind_L[i]]
        random_slices = torch.randint(self.K, self.num, (self.K,))  # choose negative items
        for i in torch.arange(self.K):
            neg_emb_L[i] = item_emb_L[pos_ind_H[random_slices[i]]]
            neg_emb_H[i] = item_emb_H[pos_ind_L[random_slices[i]]]
        return pos_emb_H, neg_emb_H, pos_emb_L, neg_emb_L

    def forward(self, session_item, session_len, D, A, reversed_sess_item, mask, epoch, tar, train, diff_mask):
        # session_item [batch_size, max_seq_len]

        if train:
            item_embeddings_hg = self.HyperGraph(self.adjacency, self.embedding.weight)
            if self.dataset == 'Tmall':
                # for Tmall dataset, we do not use position embedding to learn temporal order
                sess_emb_hgnn = self.generate_sess_emb_npos(item_embeddings_hg, session_item, session_len,reversed_sess_item, mask)
            else:
                sess_emb_hgnn = self.generate_sess_emb(item_embeddings_hg, session_item, session_len, reversed_sess_item, mask)

            # print('sess_emb_hgnn', sess_emb_hgnn.shape) #  [batch_size, seq_len, emb_size]

            sess_emb_hgnn_T = self.sasrec(session_item, sess_emb_hgnn)  # [batch_size, seq_len, emb_size]
            sess_emb_hgnn = torch.sum(sess_emb_hgnn_T, 1)  # [batch_size, emb_size]

            sess_emb_hgnn = self.w_k * F.normalize(sess_emb_hgnn, dim=-1, p=2)
            item_embeddings_hg = F.normalize(item_embeddings_hg, dim=-1, p=2)
            scores_item = torch.mm(sess_emb_hgnn, torch.transpose(item_embeddings_hg, 1, 0))
            loss_item = self.loss_function(scores_item, tar)

            sess_emb_lg = self.LineGraph(self.embedding.weight, D, A, session_item, session_len)
            scores_sess = torch.mm(sess_emb_lg, torch.transpose(item_embeddings_hg, 1, 0))
            # compute probability of items to be positive examples
            pos_prob_H = self.example_predicting(item_embeddings_hg, sess_emb_hgnn)
            pos_prob_L = self.example_predicting(self.embedding.weight, sess_emb_lg)


            pos_emb_H, neg_emb_H, pos_emb_L, neg_emb_L = self.topk_func_random(pos_prob_H, pos_prob_L, item_embeddings_hg, self.embedding.weight)

            last_item = torch.squeeze(reversed_sess_item[:, 0])
            last_item = last_item - 1
            last = item_embeddings_hg.index_select(0, last_item)
            con_loss = self.SSL_topk(last, sess_emb_hgnn, pos_emb_H, neg_emb_H)
            last = self.embedding(last_item)
            con_loss += self.SSL_topk(last, sess_emb_lg, pos_emb_L, neg_emb_L)

            # compute and update adversarial examples
            self.adversarial_item(item_embeddings_hg, tar, sess_emb_hgnn)
            self.adversarial_sess(item_embeddings_hg, tar, sess_emb_lg)

            adv_emb_item = item_embeddings_hg + self.adv_item
            adv_emb_sess = item_embeddings_hg + self.adv_sess

            score_adv1 = torch.mm(sess_emb_lg, torch.transpose(adv_emb_item, 1, 0))
            score_adv2 = torch.mm(sess_emb_hgnn, torch.transpose(adv_emb_sess, 1, 0))
            # add difference constraint
            loss_diff = self.diff(scores_item, scores_sess, score_adv2, score_adv1, diff_mask)
        else:
            item_embeddings_hg = self.HyperGraph(self.adjacency, self.embedding.weight)
            if self.dataset == 'Tmall':
                sess_emb_hgnn = self.generate_sess_emb_npos(item_embeddings_hg, session_item, session_len, reversed_sess_item, mask)
            else:
                sess_emb_hgnn = self.generate_sess_emb(item_embeddings_hg, session_item, session_len, reversed_sess_item, mask)

            sess_emb_hgnn_T = self.sasrec(session_item, sess_emb_hgnn)  # [batch_size, seq_len, emb_size]
            sess_emb_hgnn = torch.sum(sess_emb_hgnn_T, 1)  # [batch_size, emb_size]


            sess_emb_hgnn = self.w_k * F.normalize(sess_emb_hgnn, dim=-1, p=2)
            item_embeddings_hg = F.normalize(item_embeddings_hg, dim=-1, p=2)
            scores_item = torch.mm(sess_emb_hgnn, torch.transpose(item_embeddings_hg, 1, 0))
            loss_item = self.loss_function(scores_item, tar)
            loss_diff = 0
            con_loss = 0
        return self.beta * con_loss, loss_item, scores_item, loss_diff*self.lam



class SASRec(Module):
    def __init__(self, n_trm_layers, hidden_dropout_prob, n_attention_heads, attention_probs_dropout_prob,emb_size=100):
        super(SASRec, self).__init__()
        self.n_trm_layers = n_trm_layers
        self.emb_size = emb_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.n_attention_heads = n_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        # self.pos_emb = nn.Parameter(init(t.empty(args.max_seq_len, args.latdim)))
        self.n_trm_layer = nn.Sequential(
            *[TransformerLayer(self.n_attention_heads, self.hidden_dropout_prob, self.attention_probs_dropout_prob, self.emb_size) for i in range(self.n_trm_layers)])  # num_trm_layers
        self.LayerNorm = nn.LayerNorm(self.emb_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)   # hidden_dropout_prob
        self.apply(self.init_weights)

    # [batch_size, seq_len]  [batch_size, seq_len, emb_size]
    def forward(self, input_ids, seq_embedings):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)

        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0



        new_seq_embs = [seq_embedings]

        for trm in self.n_trm_layer:
            #seq_embedings = trm(new_seq_embs[-1], extended_attention_mask)
            new_seq_embs.append(trm(new_seq_embs[-1], extended_attention_mask))

        seq_emb = sum(new_seq_embs)  # [batch_size, seq_len, emb_size]

        return seq_emb

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class TransformerLayer(Module):
    def __init__(self, n_attention_heads, hidden_dropout_prob, attention_probs_dropout_prob, emb_size):
        super(TransformerLayer, self).__init__()
        self.n_attention_heads = n_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.emb_size = emb_size

        self.attention = SelfAttentionLayer(self.n_attention_heads, self.hidden_dropout_prob, self.attention_probs_dropout_prob,self.emb_size)
        self.intermediate = IntermediateLayer(self.hidden_dropout_prob, self.emb_size)

        # [batch_size, seq_len, emb_size],     batch_size, 1, seq_len, seq_len] √

    def forward(self, hidden_states, attention_mask):


        attention_output = self.attention(hidden_states, attention_mask)  # [batch_size, seq_len, emb_size]


        intermediate_output = self.intermediate(attention_output)  # [batch_size, seq_len, emb_size]

        return intermediate_output

class SelfAttentionLayer(Module):   #
    def __init__(self, n_attention_heads, hidden_dropout_prob, attention_probs_dropout_prob, emb_size):
        super(SelfAttentionLayer, self).__init__()
        self.emb_size = emb_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.n_attention_heads = n_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.attention_head_size = int(self.emb_size / self.n_attention_heads)
        self.all_head_size = self.n_attention_heads * self.attention_head_size


        self.query = nn.Linear(self.emb_size, self.all_head_size)
        self.key = nn.Linear(self.emb_size, self.all_head_size)
        self.value = nn.Linear(self.emb_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(self.attention_probs_dropout_prob)

        self.dense = nn.Linear(self.emb_size, self.emb_size)
        self.LayerNorm = nn.LayerNorm(self.emb_size)
        self.out_dropout = nn.Dropout(self.hidden_dropout_prob)

        self.apply(self.init_weights)

    # x: [batch_size, seq_len, all_head_size]
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # [batch_size, num_attention_heads, seq_len, attention_head_size]

        return x.permute(0, 2, 1, 3)  # [batch_size, num_attention_heads, seq_len, attention_head_size]

    # [batch_size, seq_len, emb_size],     [batch_size, 1, seq_len, seq_len]
    def forward(self, input_tensor, attention_mask):

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [batch_size, num_attention_heads, seq_len, attention_head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class IntermediateLayer(Module):
    def __init__(self, hidden_dropout_prob, emb_size):
        super(IntermediateLayer, self).__init__()
        self.emb_size = emb_size
        self.hidden_dropout_prob = hidden_dropout_prob


        self.layers = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size * 4, bias=True),
            nn.GELU(),
            nn.Linear(self.emb_size * 4, self.emb_size, bias=True),
            nn.Dropout(self.hidden_dropout_prob),
            nn.LayerNorm(self.emb_size)
        )

    def forward(self, x):
        return self.layers(x)



def forward(model, i, data, epoch, train):
    tar, session_len, session_item, reversed_sess_item, mask, diff_mask = data.get_slice(i)
    diff_mask = trans_to_cuda(torch.Tensor(diff_mask).long())
    A_hat, D_hat = data.get_overlap(session_item)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    A_hat = trans_to_cuda(torch.Tensor(A_hat))
    D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    con_loss, loss_item, scores_item, loss_diff = model(session_item, session_len, D_hat, A_hat, reversed_sess_item, mask, epoch,tar, train, diff_mask)
    return tar, scores_item, con_loss, loss_item, loss_diff



@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    # k_largest_scores = [item[0] for item in n_candidates]
    return ids#, k_largest_scores


def train_test(model, train_data, test_data, epoch):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i in slices:
        model.zero_grad()
        tar, scores_item, con_loss, loss_item, loss_diff = forward(model, i, train_data, epoch, train=True)
        # loss = model.loss_function(scores + 1e-8, targets)
        loss = loss_item + con_loss + loss_diff
        loss.backward()
#        print(loss.item())
        model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar, scores_item, con_loss, loss_item, loss_diff = forward(model, i, test_data, epoch, train=False)
        scores = trans_to_cpu(scores_item).detach().numpy()
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(20, scores[idd]))
        index = np.array(index)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0]+1))
    return metrics, total_loss

