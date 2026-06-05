import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_utils import inconv, down_block
from .utils import get_block, get_norm
from .attention_unet_utils import attention_up_block
from .vit_seg_modeling import TRACE

class AttentionUNet(nn.Module):
    def __init__(self, in_ch, num_classes, base_ch=32, block='SingleConv', pool=True):
        super().__init__()

        num_block = 2 
        block = get_block(block)

        self.inc = inconv(in_ch, base_ch, block=block)

        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool)
        self.down4 = down_block(8*base_ch, 16*base_ch, num_block=num_block, block=block, pool=pool)

        self.up1 = attention_up_block(16*base_ch, 8*base_ch, num_block=num_block, block=block)
        self.up2 = attention_up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block)
        self.up3 = attention_up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block)
        self.up4 = attention_up_block(2*base_ch, base_ch, num_block=num_block, block=block)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x): 

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)

        return out


class AttentionUNet_ours(nn.Module):  
    def __init__(self, img_size=224, num_classes=2, refine_iters = 3, in_ch=1,
            detach_between_iters = True):
        super(AttentionUNet_ours, self).__init__()
        self.attention_unet = AttentionUNet(in_ch=in_ch, num_classes=num_classes)
        self.refinement_module = TRACE()
        self.refine_iters = refine_iters
        self.detach_between_iters = detach_between_iters
        # self.refinement_module.load_from(weights=np.load(config_small.pretrained_path))

    def forward(self, x,  x_ref, ref_gt):  #x: bs,1,224,224
        
        logits = self.attention_unet(x)  #bs,2,224,224
        logits_ref = self.attention_unet(x_ref)  #bs,2,224,224

        probs = torch.softmax(logits, dim=1)  # (bs, 2, 224, 224)
        ori_mask = probs[:, 1:2, :, :]        # Extract the 'foreground' class probability; shape stays (bs, 1, 224, 224)

        probs_ref = torch.softmax(logits_ref, dim=1)
        ref_mask = probs_ref[:, 1:2, :, :]  # (bs, 1, 224, 224)


        target_image = torch.cat([x, ori_mask], dim=1)  #bs,2,224,224
        ref_image = torch.cat([x_ref, ref_mask, ref_gt], dim=1)  #bs,3,224,224
        preds_all = []  
        final_pred = self.refinement_module(target_image, ref_image)  #bs,2,224,224
        preds_all.append(final_pred)

        for it in range(1, max(1, self.refine_iters) ):
        # Feed the previous iteration's output as the next iteration's target prediction
            next_mask = torch.softmax(final_pred, dim=1)[:, 1:2]   

            if self.detach_between_iters:
                next_mask = next_mask.detach()

            target_2ch = torch.cat([x, next_mask], dim=1)      # (B,2,H,W)
            # Reference typically stays unchanged; if you want to update ref_pred too, swap ref_mask for the iterated logits_ref
            # Keep ref_3ch unchanged here:
            # ref_3ch = torch.cat([ref_image_raw, ref_mask, ref_gt], dim=1)

            final_pred = self.refinement_module(target_2ch, ref_image)
            preds_all.append(final_pred)
        return {"final": final_pred, "iters": preds_all}