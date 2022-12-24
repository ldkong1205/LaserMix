import torch
import torch.nn as nn

from model.range.rangenet.crf import CRF
from model.range.rangenet.module.darknet import Backbone, Decoder


class RangeNet(nn.Module):
    def __init__(
		self,
		num_cls: int,
        if_CRF: bool = False,
        H: int = 64,
        W: int = 1920,
	):
        super().__init__()
        self.num_cls = num_cls
        self.if_CRF = if_CRF
        self.H = H
        self.W = W

		# backbone
        self.backbone = Backbone()

        # do a pass of the backbone to initialize the skip connections
        stub = torch.zeros((1, 5, self.H, self.W))
        if torch.cuda.is_available():
            stub = stub.cuda()
            self.backbone.cuda()
        _, stub_skips = self.backbone(stub)

		# decoder
        self.decoder = Decoder(stub_skips=stub_skips)

		# head
        self.head = nn.Sequential(
			nn.Dropout2d(0.01),
            nn.Conv2d(
				self.decoder.get_last_depth(), self.num_cls,
				kernel_size=3, stride=1, padding=1,
			)
		) 

        # post-proc
        if self.if_CRF:
            self.CRF = CRF()
		
    def forward(self, batch, mask=None):
        scan_rv = batch['scan_rv']
        label_rv = batch['label_rv']
        if len(label_rv.size()) != 3:
            label_rv = label_rv.squeeze(dim=1)  # [bs, H, W]

        y, skips = self.backbone(scan_rv)
        y = self.decoder(y, skips)
        logits = self.head(y)

        if self.if_CRF:
            assert (mask is not None)
            logits = self.CRF(scan_rv, logits, mask)

        return logits		

