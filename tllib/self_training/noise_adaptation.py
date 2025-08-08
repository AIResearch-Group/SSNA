import torch
import torch.nn as nn
import torch.nn.functional as F


class Noise_Adaptation(nn.Module):
    def __init__(self):
        super(Noise_Adaptation, self).__init__()

    def forward(self, y_u_strong, y_u_strong_feature, y_u, y_u_feature, y_l_strong, y_l_strong_feature, y_l, y_l_feature, labels_l, noise, noise_label, m_h, m_n_h, alpha, device, mode='NDS'):

        num_classes = y_l.size(-1)

        all_y_l_features = torch.cat([y_l_feature, y_l_strong_feature])
        all_label_l = torch.cat([labels_l, labels_l]).to("cpu")
        all_y_l_one_hot_labels = torch.eye(num_classes)[all_label_l].to(device)  # (B * 2 , C)

        m_t_l = (all_y_l_one_hot_labels.T @ all_y_l_features) # (C, D)
        count_m_t_l = all_y_l_one_hot_labels.sum(dim=0) # (C)

        all_y_u = torch.cat([y_u, y_u_strong])
        all_y_u_features = torch.cat([y_u_feature, y_u_strong_feature])
        all_y_u_softmax = nn.Softmax(dim=1)(all_y_u) # (B * 2, C)
        pred_labels = torch.argmax(all_y_u_softmax, dim=1)
        all_y_u_one_hot_labels = torch.eye(num_classes)[pred_labels].to(device) # (B * 2 , C)

        m_t_u = (all_y_u_one_hot_labels.T @ all_y_u_features) # (C, D)
        count_m_t_u = all_y_u_one_hot_labels.sum(dim=0) # (C)

        m_t_c = (m_t_u + m_t_l) / (count_m_t_l + count_m_t_u + 1e-8).unsqueeze(1)
        if mode in ('NDS', 'EDD'):
            m_t_c = torch.cat([m_t_c, torch.cat([all_y_l_features, all_y_u_features]).mean(dim=0, keepdim=True)], dim=0)
        
        m_h = alpha * m_t_c + (1. - alpha) * m_h.detach()
        if mode != 'EDD':
            m_t_c = F.normalize(m_h, p=2, dim=1) # Apply L2 normalization to the m_t_c to enable cosine similarity computation (equivalent to L2 distance for unit vectors)

        all_noise_label = noise_label.to("cpu")
        all_noise_one_hot_labels = torch.eye(num_classes)[all_noise_label].to(device)  #(B , C)

        noise_sum = (all_noise_one_hot_labels.T @ noise) # (C, D)
        count_noise = all_noise_one_hot_labels.sum(dim=0) #(C)

        m_s = noise_sum / (count_noise + 1e-8).unsqueeze(1)
        if mode in ('NDS', 'EDD'):
            m_s = torch.cat([m_s, noise.mean(dim=0, keepdim=True)], dim=0)
        
        m_n_h = alpha * m_s + (1. - alpha) * m_n_h.detach()
        if mode != 'EDD':
            m_s = F.normalize(m_n_h, p=2, dim=1)

        if mode == 'NDS':
            self_transfer_loss = torch.norm(m_s - m_t_c, dim=-1)
            self_transfer_loss = self_transfer_loss.mean()
        elif mode == 'EDD':
            self_transfer_loss = torch.norm(m_n_h - m_h, dim=-1)
            self_transfer_loss = self_transfer_loss.mean()
        elif mode == 'NCSS':
            target_f = torch.cat([all_y_l_features, all_y_u_features])
            target_f = F.normalize(target_f, p=2, dim=1)
            noise = F.normalize(noise, p=2, dim=1)
            target_l = torch.cat([all_y_l_one_hot_labels, all_y_u_one_hot_labels])
            logits_mtc = torch.matmul(target_f, noise.t())
            logits_ms = logits_mtc.t()
            labels_mtc = torch.matmul(target_l, all_noise_one_hot_labels.t())
            labels_mtc = 2. * labels_mtc - 1
            labels_ms = labels_mtc.t()
            self_transfer_loss = (F.mse_loss(logits_ms, labels_ms) + F.mse_loss(logits_mtc, labels_mtc)) / (2 * num_classes)
        elif mode == 'NCDS':
            logits_mtc = torch.matmul(m_t_c, m_s.t()) # Similarity matrix, (C, C)
            logits_ms = logits_mtc.t()
            num_classes = m_s.shape[0]
            labels = torch.arange(num_classes, device=m_s.device)
            self_transfer_loss = (F.cross_entropy(logits_ms, labels) + F.cross_entropy(logits_mtc, labels)) / (2 * num_classes)
        else:
            target_f = torch.cat([all_y_l_features, all_y_u_features])
            target_f = F.normalize(target_f, p=2, dim=1)
            noise = F.normalize(noise, p=2, dim=1)
            target_l = torch.cat([all_y_l_one_hot_labels, all_y_u_one_hot_labels])
            labels_mtc = torch.matmul(target_l, all_noise_one_hot_labels.t())
            diff = target_f[:, None, :] - noise[None, :, :]
            l2_dist = torch.norm(diff, dim=2)
            self_transfer_loss = l2_dist[labels_mtc == 1].mean()


        return self_transfer_loss, m_h, m_n_h
    