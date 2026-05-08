import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_block
from .medformer_utils import down_block, up_block, inconv, SemanticMapFusion
import pdb
from .vit_seg_modeling import TRACE


class MedFormer(nn.Module):

    def __init__(self, in_chan, num_classes, base_chan=32, map_size=8, conv_block='BasicBlock', conv_num=[2,1,0,0, 0,1,2,2], trans_num=[0,1,2,2, 2,1,0,0], num_heads=[1,4,8,16, 8,4,1,1], fusion_depth=2, fusion_dim=512, fusion_heads=16, expansion=4, attn_drop=0., proj_drop=0., proj_type='depthwise', norm=nn.BatchNorm2d, act=nn.ReLU, aux_loss=False):
        super().__init__()
        
        
        chan_num = [2*base_chan, 4*base_chan, 8*base_chan, 16*base_chan, 
                        8*base_chan, 4*base_chan, 2*base_chan, base_chan]
        dim_head = [chan_num[i]//num_heads[i] for i in range(8)]
        conv_block = get_block(conv_block)

        # self.inc and self.down1 forms the conv stem
        self.inc = inconv(in_chan, base_chan, norm=norm, act=act)
        self.down1 = down_block(base_chan, chan_num[0], conv_num[0], trans_num[0], conv_block, norm=norm, act=act, map_generate=False)
        
        # down2 down3 down4 apply the B-MHA blocks
        self.down2 = down_block(chan_num[0], chan_num[1], conv_num[1], trans_num[1], conv_block, heads=num_heads[1], dim_head=dim_head[1], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
        self.down3 = down_block(chan_num[1], chan_num[2], conv_num[2], trans_num[2], conv_block, heads=num_heads[2], dim_head=dim_head[2], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)
        self.down4 = down_block(chan_num[2], chan_num[3], conv_num[3], trans_num[3], conv_block, heads=num_heads[3], dim_head=dim_head[3], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_generate=True)

        
        self.map_fusion = SemanticMapFusion(chan_num[1:4], fusion_dim, fusion_heads, depth=fusion_depth, norm=norm)


        self.up1 = up_block(chan_num[3], chan_num[4], conv_num[4], trans_num[4], conv_block, heads=num_heads[4], dim_head=dim_head[4], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)
        self.up2 = up_block(chan_num[4], chan_num[5], conv_num[5], trans_num[5], conv_block, heads=num_heads[5], dim_head=dim_head[5], expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, map_shortcut=True)
         
         # up3 up4 form the conv decoder
        self.up3 = up_block(chan_num[5], chan_num[6], conv_num[6], trans_num[6], conv_block, norm=norm, act=act, map_shortcut=False)
        self.up4 = up_block(chan_num[6], chan_num[7], conv_num[7], trans_num[7], conv_block, norm=norm, act=act, map_shortcut=False)
        

        self.outc = nn.Conv2d(chan_num[7], num_classes, kernel_size=1)

        self.aux_loss = aux_loss
        if aux_loss:
            self.aux_out = nn.Conv2d(chan_num[5], num_classes, kernel_size=1)

    def forward(self, x):
        
        x0 = self.inc(x)
        x1, _ = self.down1(x0)
        x2, map2 = self.down2(x1)
        x3, map3 = self.down3(x2)
        x4, map4 = self.down4(x3)
        
        map_list = [map2, map3, map4]
        map_list = self.map_fusion(map_list)
        
        out, semantic_map = self.up1(x4, x3, map_list[2], map_list[1])
        out, semantic_map = self.up2(out, x2, semantic_map, map_list[0])

        if self.aux_loss:
            aux_out = self.aux_out(out)
            aux_out = F.interpolate(aux_out, size=x.shape[-2:], mode='bilinear', align_corners=True)

        out, semantic_map = self.up3(out, x1, semantic_map, None)
        out, semantic_map = self.up4(out, x0, semantic_map, None)

        out = self.outc(out)

        if self.aux_loss:
            return [out, aux_out]
        else:
            return out


class MedFormer_ours(nn.Module):  
    def __init__(self, img_size=224, num_classes=2, refine_iters = 3, in_chan=1,
            detach_between_iters = True):
        super(MedFormer_ours, self).__init__()
        self.medformer = MedFormer(in_chan=in_chan, num_classes=num_classes)
        self.refinement_module = TRACE()
        self.refine_iters = refine_iters
        self.detach_between_iters = detach_between_iters
        # self.refinement_module.load_from(weights=np.load(config_small.pretrained_path))

    def forward(self, x,  x_ref, ref_gt):  #x: bs,1,224,224
        
        logits = self.medformer(x)  #bs,2,224,224
        logits_ref = self.medformer(x_ref)  #bs,2,224,224

        probs = torch.softmax(logits, dim=1)  # (bs, 2, 224, 224)
        ori_mask = probs[:, 1:2, :, :]        # Extract the 'foreground' class probability; shape stays (bs, 1, 224, 224)

        probs_ref = torch.softmax(logits_ref, dim=1)
        ref_mask = probs_ref[:, 1:2, :, :]  # (bs, 1, 224, 224)

        # ori_mask = torch.argmax(logits, dim=1).unsqueeze(1)  #bs,1,224,224
        # ref_mask = torch.argmax(logits_ref, dim=1).unsqueeze(1)  #bs,1,224,224

        # print('ori_mask:',ori_mask.shape,ori_mask.min(),ori_mask.max())
        # print('mask_ref:',mask_ref.shape,mask_ref.min(),mask_ref.max())
        # print('ori_image:',ori_image.shape,ori_image.min(),ori_image.max())
        # raise Exception
        # new_image = torch.cat([ori_image, ori_mask, ref_image, ref_mask, ref_gt], dim=1)  #bs,5,224,224
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
        # print('final_pred:',final_pred.shape)
        # print('logits:',logits.shape)
        # raise Exception
        # return logits
        return {"final": final_pred, "iters": preds_all}