import torch
import torch.nn as nn


class MultiBoxLayer(nn.Module):
    num_anchors = [4, 6, 6, 6, 4, 4]
    in_planes = [512, 1024, 512, 256, 256, 256]

    def __init__(self):
        super(MultiBoxLayer, self).__init__()

        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for i in range(len(self.in_planes)):
            self.loc_layers.append(
                nn.Conv2d(self.in_planes[i], self.num_anchors[i]*4, kernel_size=3, padding=1)
            )
            self.conf_layers.append(
                nn.Conv2d(self.in_planes[i], self.num_anchors[i]*21, kernel_size=3, padding=1)
            )

    def forward(self, xs):
        """
        Arguments:
            xs: a list of float tensors, intermediate layers' outputs.

        Returns:
            loc_preds: a float tensor of shape [n, 8732, 4], predicted locations.
            conf_preds: a float tensor of shape [n, 8732, 21], predicted class confidences.
        """
        y_locs = []
        y_confs = []
        for i, x in enumerate(xs):
            n = x.size(0)  # batch size

            y_loc = self.loc_layers[i](x)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_locs.append(y_loc.view(n, -1))

            y_conf = self.conf_layers[i](x)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_confs.append(y_conf.view(n, -1))

        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)

        loc_preds = loc_preds.view(n, -1, 4)
        conf_preds = conf_preds.view(n, -1, 21)

        return loc_preds, conf_preds
