import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_block, get_norm
from .vit_seg_modeling import TRACE


class UNetPlusPlus(nn.Module):
    def __init__(self, in_ch, num_classes, base_ch=32, block='SingleConv'):
        super().__init__()

        num_block = 2
        block = get_block(block)

        n_ch = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16]
    
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.conv0_0 = self.make_layer(in_ch, n_ch[0], num_block, block)
        self.conv1_0 = self.make_layer(n_ch[0], n_ch[1], num_block, block)
        self.conv2_0 = self.make_layer(n_ch[1], n_ch[2], num_block, block)
        self.conv3_0 = self.make_layer(n_ch[2], n_ch[3], num_block, block)
        self.conv4_0 = self.make_layer(n_ch[3], n_ch[4], num_block, block)
        self.conv0_1 = self.make_layer(n_ch[0]+n_ch[1], n_ch[0], num_block, block)
        self.conv1_1 = self.make_layer(n_ch[1]+n_ch[2], n_ch[1], num_block, block)
        self.conv2_1 = self.make_layer(n_ch[2]+n_ch[3], n_ch[2], num_block, block)
        self.conv3_1 = self.make_layer(n_ch[3]+n_ch[4], n_ch[3], num_block, block)

        self.conv0_2 = self.make_layer(n_ch[0]*2+n_ch[1], n_ch[0], num_block, block)
        self.conv1_2 = self.make_layer(n_ch[1]*2+n_ch[2], n_ch[1], num_block, block)
        self.conv2_2 = self.make_layer(n_ch[2]*2+n_ch[3], n_ch[2], num_block, block)

        self.conv0_3 = self.make_layer(n_ch[0]*3+n_ch[1], n_ch[0], num_block, block)
        self.conv1_3 = self.make_layer(n_ch[1]*3+n_ch[2], n_ch[1], num_block, block)


        self.conv0_4 = self.make_layer(n_ch[0]*4+n_ch[1], n_ch[0], num_block, block)


        self.output = nn.Conv2d(n_ch[0], num_classes, kernel_size=1)


    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.output(x0_4)

        return output


    def make_layer(self, in_ch, out_ch, num_block, block):
        blocks = []
        blocks.append(block(in_ch, out_ch))

        for i in range(num_block-1):
            blocks.append(block(out_ch, out_ch))

        return nn.Sequential(*blocks)


class UNetPlusPlus_ours(nn.Module):  
    def __init__(self, img_size=224, num_classes=2, refine_iters = 3, in_ch=1,
            detach_between_iters = True):
        super(UNetPlusPlus_ours, self).__init__()
        self.unetpp = UNetPlusPlus(in_ch=in_ch, num_classes=num_classes)
        self.refinement_module = TRACE()
        self.refine_iters = refine_iters
        self.detach_between_iters = detach_between_iters
        # self.refinement_module.load_from(weights=np.load(config_small.pretrained_path))

    def forward(self, x,  x_ref, ref_gt):  #x: bs,1,224,224
        
        logits = self.unetpp(x)  #bs,2,224,224
        logits_ref = self.unetpp(x_ref)  #bs,2,224,224

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