import torch
import torch.nn as nn
import torch.nn.functional as F

from convols import LightSignedConv, LightSignedConv2


class LightSignedGCN(nn.Module):
    def __init__(self, num_u, num_v, num_layer=2, dim=64, reg=1e-4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.M = num_u  # user number
        self.N = num_v  # item number
        self.num_layer = num_layer
        self.dim = dim
        self.reg = reg
        self.embed_dim = dim

        self.user_embedding = nn.Parameter(torch.empty(self.M, self.dim))
        self.item_embedding = nn.Parameter(torch.empty(self.N, self.dim))
        nn.init.xavier_normal_(self.user_embedding)
        nn.init.xavier_normal_(self.item_embedding)
        self.user_neg_embedding = nn.Parameter(torch.empty(self.M, self.dim))
        self.item_neg_embedding = nn.Parameter(torch.empty(self.N, self.dim))
        nn.init.xavier_normal_(self.user_neg_embedding)
        nn.init.xavier_normal_(self.item_neg_embedding)

        # self.pos_linear = nn.Linear(self.dim, self.dim)
        # self.neg_linear = nn.Linear(self.dim, self.dim)

        self.conv = nn.ModuleList()
        hidden_dim = self.dim
        for i in range(self.num_layer):
            if i == 0:
                self.conv.append(LightSignedConv(hidden_dim, hidden_dim, True))
            else:
                self.conv.append(LightSignedConv(hidden_dim, hidden_dim, False))

        self.lin = torch.nn.Linear(2 * hidden_dim, 3)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.rating_pred = torch.nn.Linear(2 * hidden_dim, 1)
        self.mse_loss = nn.MSELoss()

    def forward(self, data_p, data_n):
        pos_edges = data_p.edge_index
        neg_edges = data_n.edge_index
        alpha = 1. / (self.num_layer + 1)

        ego_pos_embeddings = torch.cat((self.user_embedding, self.item_embedding), dim=0)
        ego_neg_embeddings = torch.cat((self.user_neg_embedding, self.item_neg_embedding), dim=0)
        # ego_neg_embeddings = torch.cat((self.user_embedding, self.item_embedding), dim=0)
        # ego_embeddings = (ego_pos_embeddings, ego_neg_embeddings)
        ego_embeddings = (ego_pos_embeddings, ego_pos_embeddings)
        pos_embeddings, neg_embeddings = ego_pos_embeddings * alpha, ego_neg_embeddings * alpha

        for i in range(self.num_layer):
            ego_embeddings = self.conv[i](ego_embeddings, pos_edges, neg_edges)
            pos_embeddings, neg_embeddings = pos_embeddings + ego_embeddings[0] * alpha, neg_embeddings + ego_embeddings[1] * alpha
        return pos_embeddings, neg_embeddings

    def loss(self, users, items, weights, negative_samples, data_p, data_n):
        """
        Args:
            users: batch users id
            items: batch items id
            weights: user-item (ratings-offset)
            negative_samples: negative samples for BPR
            data_p: positive edges
            data_n: negative edges

        Returns: loss to backward

        """
        loss = 0.
        pos_emb, neg_emb = self(data_p, data_n)

        # 1. Positive BPR Loss
        u_p = pos_emb[users]
        i_p = pos_emb[items]
        n_p = pos_emb[negative_samples]
        positive_batch = torch.mul(u_p, i_p)
        negative_batch = torch.mul(u_p.view(len(u_p), 1, self.embed_dim), n_p)

        pos_bpr_loss = F.logsigmoid(
            (-1/2*torch.sign(weights) + 3/2).view(len(u_p), 1) * positive_batch.sum(dim=1).view(len(u_p), 1)
            - negative_batch.sum(dim=2)
        ).sum(dim=1)

        # pos_bpr_loss = F.logsigmoid(
        #     torch.sign(weights).view(len(u_p), 1) * (positive_batch.sum(dim=1).view(len(u_p), 1) - negative_batch.sum(dim=2))
        # )

        pos_bpr_loss = torch.mean(pos_bpr_loss)
        # pos_bpr_loss = torch.sum(pos_bpr_loss)

        loss += -pos_bpr_loss
        reg_loss_1 = 1. / 2 * (u_p ** 2).sum() + 1. / 2 * (i_p ** 2).sum() + 1. / 2 * (n_p ** 2).sum()
        # reg_loss_1 = u_p.norm(dim=1).pow(2).sum() + i_p.norm(dim=1).pow(2).sum() + n_p.norm(dim=2).pow(2).sum()
        loss += self.reg * reg_loss_1

        # 2. Negative BPR Loss 1+2 0.1246 1+2+40.1246
        u_n = neg_emb[users]
        i_n = neg_emb[items]
        n_n = neg_emb[negative_samples]
        positive_batch = torch.mul(u_n, i_n)
        negative_batch = torch.mul(u_n.view(len(u_p), 1, self.embed_dim), n_n)
        neg_bpr_loss = F.logsigmoid(
            negative_batch.sum(dim=2)
            - (1/2*torch.sign(weights) + 3/2).view(len(u_n), 1) * positive_batch.sum(dim=1).view(len(u_n),1)
        ).sum(dim=1)
        # neg_bpr_loss = F.logsigmoid(
        #     (1/2*(-torch.sign(weights)) + 3/2).view(len(u_n), 1) * positive_batch.sum(dim=1).view(len(u_n), 1)
        #     - negative_batch.sum(dim=2)
        # ).sum(dim=1)

        # neg_bpr_loss = F.logsigmoid(
        #     -torch.sign(weights).view(len(u_p), 1) * (positive_batch.sum(dim=1).view(len(u_n), 1) - negative_batch.sum(dim=2))
        # )

        neg_bpr_loss = torch.mean(neg_bpr_loss)
        # neg_bpr_loss = torch.sum(neg_bpr_loss)
        loss += -neg_bpr_loss

        reg_loss_2 = 1. / 2 * (u_n ** 2).sum() + 1. / 2 * (i_n ** 2).sum() + 1. / 2 * (n_n ** 2).sum()
        # reg_loss_2 = u_n.norm(dim=1).pow(2).sum() + i_n.norm(dim=1).pow(2).sum() + n_n.norm(dim=2).pow(2).sum()
        loss += self.reg * reg_loss_2

        # 3. Link Prediction Loss

        # sign_labels = ((torch.sign(weights) + 1) / 2).long()
        # sign_pred = torch.log_softmax(self.lin(torch.cat([u_p, i_p], dim=1)), dim=1)
        # labels_onehot = torch.zeros_like(sign_pred)
        # labels_onehot.scatter_(1, sign_labels.unsqueeze(1), 1)
        # loss += self.bce_loss(sign_pred, labels_onehot)

        # pos_labels = ((torch.sign(weights) + 1) / 2).long()
        # pos_pred = torch.log_softmax(self.lin(torch.cat([u_p, i_p], dim=1)), dim=1)
        # loss += F.nll_loss(pos_pred, pos_labels)
        # u_p_repeat = u_p.unsqueeze(1).repeat(1, 40, 1)
        # neg_pred = torch.log_softmax(self.lin(torch.cat((u_p_repeat, n_p), dim=2)), dim=1).reshape(-1, 3)
        # neg_labels = neg_pred.new_full((neg_pred.size(0),), 2).long()
        # loss += F.nll_loss(neg_pred, neg_labels)

        rating_label = weights
        rating_pred = self.rating_pred(torch.cat([u_p, i_p], dim=1))
        loss += self.mse_loss(rating_label, rating_pred)

        # 4. Cross Loss 1+4 0.1285
        u_cross_loss = torch.mean(torch.sum(torch.mul(u_p, u_n), dim=1) ** 2, dim=0)
        i_cross_loss = torch.mean(torch.sum(torch.mul(i_p, i_n), dim=1) ** 2, dim=0)
        loss += (i_cross_loss + u_cross_loss)

        return loss

    @torch.no_grad()
    def get_ui_embeddings(self, data_p, data_n):
        pos_embeddings, neg_embeddings = self(data_p, data_n)
        u_p_embeddings, i_p_embeddings = torch.split(pos_embeddings, [self.M, self.N], dim=0)
        u_n_embeddings, i_n_embeddings = torch.split(neg_embeddings, [self.M, self.N], dim=0)
        return u_p_embeddings, u_n_embeddings, i_p_embeddings, i_n_embeddings





