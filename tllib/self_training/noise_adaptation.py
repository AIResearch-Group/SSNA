import torch
import torch.nn as nn
import torch.nn.functional as F


class Noise_Adaptation(nn.Module):
    def __init__(self):
        super(Noise_Adaptation, self).__init__()

    def forward(self, y_u_strong, y_u_strong_feature, y_u, y_u_feature, y_l_strong, y_l_strong_feature, y_l, y_l_feature, labels_l, m_s, device, PAM=True):
        all_y_l_features = torch.cat([y_l_feature, y_l_strong_feature])
        all_label_l = torch.cat([labels_l, labels_l]).to("cpu")
        num_classes = y_l.size(-1)
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

        m_t_c = m_t_u + m_t_l
        count_m_t_c = count_m_t_l + count_m_t_u

        # Select classes in the mini-batch
        valid_idx = count_m_t_c != 0
        count_m_t_c = count_m_t_c[valid_idx]
        m_t_c = m_t_c[valid_idx]
        m_s = m_s[valid_idx]

        m_t_c = m_t_c / count_m_t_c.unsqueeze(1)
        m_t_c = F.normalize(m_t_c, p=2, dim=1)
        m_s = F.normalize(m_s, p=2, dim=1)

        if PAM:
            self_transfer_loss = torch.norm(m_s - m_t_c, dim=-1)
            self_transfer_loss = self_transfer_loss.mean()
        else:
            logits_mtc = torch.matmul(m_t_c, m_s.t()) # Similarity matrix, (C, C)
            logits_ms = logits_mtc.t()
            num_classes = m_s.shape[0]
            labels = torch.arange(num_classes, device=m_s.device)
            self_transfer_loss = (F.cross_entropy(logits_ms, labels) + F.cross_entropy(logits_mtc, labels)) / 2

        return self_transfer_loss