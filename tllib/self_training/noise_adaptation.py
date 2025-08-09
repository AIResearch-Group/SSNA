import torch
import torch.nn as nn
import torch.nn.functional as F


class Noise_Adaptation(nn.Module):
    def __init__(self):
        super(Noise_Adaptation, self).__init__()

    def forward(self, y_u_strong, y_u_strong_feature, y_u, y_u_feature, y_l_strong, y_l_strong_feature, y_l, y_l_feature, labels_l, noise, noise_label, m_h, m_n_h, alpha, device):

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
        m_t_c = torch.cat([m_t_c, torch.cat([all_y_l_features, all_y_u_features]).mean(dim=0, keepdim=True)], dim=0)
        
        m_h = alpha * m_t_c + (1. - alpha) * m_h.detach()
        m_t_c = F.normalize(m_h, p=2, dim=1) # Apply L2 normalization to the m_t_c to enable cosine similarity computation (equivalent to L2 distance for unit vectors)

        all_noise_label = noise_label.to("cpu")
        all_noise_one_hot_labels = torch.eye(num_classes)[all_noise_label].to(device)  #(B , C)

        noise_sum = (all_noise_one_hot_labels.T @ noise) # (C, D)
        count_noise = all_noise_one_hot_labels.sum(dim=0) #(C)

        m_s = noise_sum / (count_noise + 1e-8).unsqueeze(1)
        m_s = torch.cat([m_s, noise.mean(dim=0, keepdim=True)], dim=0)
        
        m_n_h = alpha * m_s + (1. - alpha) * m_n_h.detach()
        m_s = F.normalize(m_n_h, p=2, dim=1)

        self_transfer_loss = torch.norm(m_s - m_t_c, dim=-1)
        self_transfer_loss = self_transfer_loss.mean()

        return self_transfer_loss, m_h, m_n_h
    