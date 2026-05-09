# -*- coding: utf-8 -*-
"""
MedSAM2 + TRACE model with adds-on refinement module + iterative refinement + deep supervision
"""

import torch
import torch.nn.functional as F
from training.model.trace import TRACE


class MedSAM2_with_TRACE(torch.nn.Module):
    """
    MedSAM2 with the TRACE add-on module.
    Bundles iterative refinement and deep supervision.
    """
    def __init__(
        self,
        sam2_model,
        refinement,
        refine_iters=3,
        detach_between_iters=True,
    ):
        super().__init__()
        self.sam2_model = sam2_model
        self.refinement = refinement
        self.refine_iters = refine_iters
        self.detach_between_iters = detach_between_iters
        
        # Freeze the prompt encoder (if present)
        if hasattr(sam2_model, 'sam_prompt_encoder'):
            for param in sam2_model.sam_prompt_encoder.parameters():
                param.requires_grad = False

    def forward(self, images, boxes, ref_images, ref_gts, image_size=512):
        """
        Forward pass with iterative refinement and deep supervision
        
        Args:
            images: (B, 3, H, W) current slice image, already normalized
            boxes: (B, 4) box coordinates [x_min, y_min, x_max, y_max] in image_size scale
            ref_images: (B, 3, H, W) reference slice image, already normalized
            ref_gts: (B, 1, H, W) reference-slice GT mask in [0, 1]
            image_size: model input size, default 512
        
        Returns:
            dict with keys:
                'final': (B, 1, H, W) final mask logits
                'iters': list of (B, 1, H, W) mask logits across all iterations (for deep supervision)
        """
        B = images.shape[0]
        device = images.device
        
        # 1. Get the initial prediction from MedSAM2
        # Get image features
        backbone_out = self.sam2_model.forward_image(images)
        _, vision_feats, vision_pos_embeds, feat_sizes = self.sam2_model._prepare_backbone_features(backbone_out)
        
        H_feat, W_feat = feat_sizes[-1]
        C = self.sam2_model.hidden_dim
        backbone_features = vision_feats[-1].permute(1, 2, 0).view(B, C, H_feat, W_feat)  # (B, C, H_feat, W_feat)
        
        # Prepare high-res features (if needed)
        if self.sam2_model.use_high_res_features_in_sam and len(vision_feats) > 1:
            # vision_feats[i]: (HW, B, C) -> (B, C, H, W)
            high_res_features = [
                vision_feats[i].permute(1, 2, 0).view(vision_feats[i].size(1), vision_feats[i].size(2), *feat_sizes[i])
                for i in range(len(vision_feats) - 1)
            ]
        else:
            high_res_features = None
        
        # Prepare the box prompt
        boxes_torch = torch.as_tensor(boxes, dtype=torch.float32, device=device)
        if boxes_torch.dim() == 2 and boxes_torch.shape[1] == 4:
            boxes_torch = boxes_torch.reshape(B, 2, 2)  # (B, 2, 2)
        
        # Run the prompt encoder
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=None,
                boxes=boxes_torch,
                masks=None,
            )
        
        # Run the mask decoder
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        # Select the best mask
        best_iou_inds = torch.argmax(ious, dim=-1)
        batch_inds = torch.arange(B, device=device)
        low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)  # (B, 1, H_low, W_low)
        
        # Upsample to the original image size
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        ori_mask = torch.sigmoid(ori_res_masks)  # (B, 1, image_size, image_size)
        
        # 2. Run the same pipeline on the reference image to obtain the reference mask
        ref_backbone_out = self.sam2_model.forward_image(ref_images)
        _, ref_vision_feats, ref_vision_pos_embeds, ref_feat_sizes = self.sam2_model._prepare_backbone_features(ref_backbone_out)
        
        ref_H_feat, ref_W_feat = ref_feat_sizes[-1]
        ref_backbone_features = ref_vision_feats[-1].permute(1, 2, 0).view(B, C, ref_H_feat, ref_W_feat)
        
        # Prepare reference high-res features (if needed)
        if self.sam2_model.use_high_res_features_in_sam and len(ref_vision_feats) > 1:
            ref_high_res_features = [
                ref_vision_feats[i].permute(1, 2, 0).view(ref_vision_feats[i].size(1), ref_vision_feats[i].size(2), *ref_feat_sizes[i])
                for i in range(len(ref_vision_feats) - 1)
            ]
        else:
            ref_high_res_features = None
        
        with torch.no_grad():
            ref_sparse_embeddings, ref_dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=None,
                boxes=boxes_torch,
                masks=None,
            )
        
        ref_low_res_multimasks, ref_ious, _, _ = self.sam2_model.sam_mask_decoder(
            image_embeddings=ref_backbone_features,
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=ref_sparse_embeddings,
            dense_prompt_embeddings=ref_dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=ref_high_res_features,
        )
        
        ref_best_iou_inds = torch.argmax(ref_ious, dim=-1)
        ref_batch_inds = torch.arange(B, device=device)
        ref_low_res_masks = ref_low_res_multimasks[ref_batch_inds, ref_best_iou_inds].unsqueeze(1)
        
        ref_res_masks = F.interpolate(
            ref_low_res_masks,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        ref_mask = torch.sigmoid(ref_res_masks)  # (B, 1, image_size, image_size)
        
        # 3. Prepare the input for the refinement module
        # target: [image, ori_mask] -> (B, 2, H, W)
        # Note: images are RGB; take the first channel as grayscale (or convert)
        if images.shape[1] == 3:
            # Convert to grayscale (mean across channels or take the first one)
            target_image = images[:, 0:1, :, :]  # use the first channel as grayscale
        else:
            target_image = images
        
        target_2ch = torch.cat([target_image, ori_mask], dim=1)  # (B, 2, H, W)
        
        # reference: [ref_image, ref_mask, ref_gt] -> (B, 3, H, W)
        if ref_images.shape[1] == 3:
            ref_image_gray = ref_images[:, 0:1, :, :]
        else:
            ref_image_gray = ref_images
        
        ref_3ch = torch.cat([ref_image_gray, ref_mask, ref_gts], dim=1)  # (B, 3, H, W)
        
        # 4. Iterative refinement with deep supervision
        preds_all = []
        final_pred = self.refinement(target_2ch, ref_3ch)  # (B, 1, H, W)
        preds_all.append(final_pred)
        
        for it in range(1, max(1, self.refine_iters)):
            # Feed the previous iteration's output as the next iteration's target prediction
            next_mask = torch.sigmoid(final_pred)
            
            if self.detach_between_iters:
                next_mask = next_mask.detach()
            
            target_2ch = torch.cat([target_image, next_mask], dim=1)  # (B, 2, H, W)
            
            # Reference stays unchanged
            final_pred = self.refinement(target_2ch, ref_3ch)
            preds_all.append(final_pred)
        
        return {"final": final_pred, "iters": preds_all}
