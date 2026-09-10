import torch.nn as nn
import timm
import torch 

class HierarchicalUltrasoundModel(nn.Module):
    def __init__(self, num_anatomies, num_planes, backbone_name="convnext_base", pretrained=False, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, drop_path_rate=dropout, num_classes=0,
        )
        feat_dim = self.backbone.num_features

        self.anatomy_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_anatomies)
        )
        # plane head now takes backbone features + anatomy logits concatenated
        self.plane_head = nn.Sequential(
            nn.LayerNorm(feat_dim + num_anatomies),
            nn.Dropout(dropout),
            nn.Linear(feat_dim + num_anatomies, num_planes)
        )

    def forward(self, x):
        features = self.backbone(x)
        # feature_map = self.backbone.forward_features(x)
        # features = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        anatomy_logits = self.anatomy_head(features)
        # detach so plane-head gradient doesn't corrupt anatomy head's own training signal
        plane_input = torch.cat([features, anatomy_logits.detach()], dim=-1)
        plane_logits = self.plane_head(plane_input)
        return {"anatomy": anatomy_logits, "plane": plane_logits} #, "feature_maps": feature_map, "features": features}
