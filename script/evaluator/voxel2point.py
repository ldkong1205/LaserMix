# map voxel cells to points

# NOTE: to be updated

# label = batch['point_label']
# point_pred = []
# point_pred_logits = []
# point_labels = []

# for idx in range(int(batch['point_coord'][:, -1].max() + 1)):
#     mask_point = batch['point_coord'][:, -1] == idx
#     mask_logits = up0e.C[:, -1] == idx

#     out_batch_i = logits[mask_logits].argmax(1)
#     out_batch_i_logits = logits[mask_logits]

#     hash_logits = torchsparse.nn.functional.sphash(up0e.C[mask_logits])
#     hash_point = torchsparse.nn.functional.sphash(batch['point_coord'][mask_point].to(up0e.C))
#     idx_query = torchsparse.nn.functional.sphashquery(hash_point, hash_logits)
#     point_pred.append(out_batch_i[idx_query][:batch['num_points'][idx]].cpu().numpy())
#     point_labels.append(label[mask_point][:batch['num_points'][idx]].cpu().numpy())
#     point_pred_logits.append(out_batch_i_logits[idx_query][:batch['num_points'][idx]].cpu().numpy())

# return logits