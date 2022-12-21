import torch
from torch.nn import functional as F

from script.evaluator.avgmeter import AverageMeter
from script.trainer.utils import map_data_to_gpu


def validate(logger, loader_val, evaluator, model, criterion, info, args, cfg, device):

    # set meta info
    meter_loss = AverageMeter()
    meter_acc  = AverageMeter()
    meter_iou  = AverageMeter()
    meter_wce  = AverageMeter()
    meter_jacc = AverageMeter()

    # class id
    if cfg.DATA.DATASET == 'nuscenes':
        class_name = {
            0: "ignore_label", 1: "barrier", 2: "bicycle", 3: "bus", 4: "car",
            5: "construction_vehicle", 6: "motorcycle", 7: "pedestrian", 8: "traffic_cone",
            9: "trailer", 10: "truck", 11: "driveable_surface", 12: "flat_other",
            13: "sidewalk", 14: "terrain", 15: "manmade", 16: "vegetation" 
        }
    elif cfg.DATA.DATASET == 'semantickitti' or cfg.DATA.DATASET == 'scribblekitti':
        class_name = {
            0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck", 5: "other-vehicle",
            6: "person", 7: "bicyclist", 8: "motorcyclist", 9: "road", 10: "parking",
            11: "sidewalk", 12: "other-ground", 13: "building", 14: "fence", 15: "vegetation",
            16: "trunk", 17: "terrain", 18: "pole", 19: "traffic-sign"
        }

    # switch to evaluate mode
    model.eval()
    evaluator.reset()

    # val steps start here
    with torch.no_grad():

        for idx, data in enumerate(loader_val):

            if cfg.MODEL.MODALITY == 'range':
                scan, label = data['scan'].to(device), torch.squeeze(data['label'], dim=1).to(device)

            elif cfg.MODEL.MODALITY == 'voxel':
                map_data_to_gpu(data)

            with torch.cuda.amp.autocast(enabled=args.amp):

                if cfg.MODEL.MODALITY == 'range':
                    logits = model(scan)
                    if logits.size()[-1] != label.size()[-1] and logits.size()[-2] != label.size()[-2]:
                        logits = F.interpolate(logits, size=label.size()[1:], mode='bilinear', align_corners=True)  # [bs, cls, H, W]
                    
                elif cfg.MODEL.MODALITY == 'voxel':
                    logits, label = model(data)  # [uniq, cls], [uniq]
                
                wce  = criterion[0](logits, label).contiguous().view(-1).mean()
                jacc = criterion[1](F.softmax(logits, dim=1), label)
                loss = wce + jacc

            # measure accuracy and record loss
            argmax = F.softmax(logits, dim=1).argmax(dim=1)
            evaluator.addBatch(argmax, label)
            meter_loss.update(loss.mean().item(), cfg.VALID.BATCH_SIZE)
            meter_jacc.update(jacc.mean().item(), cfg.VALID.BATCH_SIZE)
            meter_wce.update(wce.mean().item(), cfg.VALID.BATCH_SIZE)

        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
        meter_acc.update(accuracy.item(), cfg.VALID.BATCH_SIZE)
        meter_iou.update(jaccard.item(), cfg.VALID.BATCH_SIZE)

        logger.info("validation")
        logger.info("  loss avg: {loss.avg:.4f}".format(loss=meter_loss))
        logger.info("  acc avg: {acc.avg:.3f}".format(acc=meter_acc))
        logger.info("  iou avg: {iou.avg:.3f}".format(iou=meter_iou))
        logger.info("  wce avg: {wce.avg:.4f}".format(wce=meter_wce))
        logger.info("  jaccard avg: {jacc.avg:.4f}".format(jacc=meter_jacc))

        logger.info("-"*52)
        logger.info("class-wise iou scores")

        for i, jacc in enumerate(class_jaccard):

            if i != 0:
                logger.info("  class '{i:}' [{class_str:}] = {jacc:.3f}".format(
                    i=i, class_str=class_name[i], jacc=jacc))

            info["valid_classes/" + class_name[i]] = jacc

    return meter_loss.avg, meter_acc.avg, meter_iou.avg

