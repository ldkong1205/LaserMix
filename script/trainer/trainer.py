import os

import torch
from torch.nn import functional as F

from script.trainer.utils import ClassWeightSemikitti, CrossEntropyDiceLoss, Lovasz_softmax, BoundaryLoss
from script.trainer.validator import validate
from script.evaluator.avgmeter import AverageMeter
from script.evaluator.ioueval import iouEval


def train(logger, model, datasets, args, cfg, device):

    # set train loader
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]

    sampler_train = torch.utils.data.RandomSampler(datasets[0])
    loader_train = torch.utils.data.DataLoader(
        datasets[0],
        batch_size=cfg.TRAIN.BATCH_SIZE,
        sampler=sampler_train,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        drop_last=True,
        pin_memory=True
    )
    num_batch_train = len(loader_train)

    # set val loader
    loader_val = torch.utils.data.DataLoader(
        datasets[1],
        batch_size=cfg.VALID.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.VALID.NUM_WORKERS,
        drop_last=True, 
        pin_memory=True
    )
    num_batch_val = len(loader_val)

    # set optimizer
    if cfg.OPTIM.OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.OPTIM.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-5
        )
    elif cfg.OPTIM.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.OPTIM.LEARNING_RATE,
            momentum=0.9,
        )
    else:
        raise NotImplementedError

    # set scheduler
    if cfg.OPTIM.SCHEDULER == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.OPTIM.LEARNING_RATE,
            epochs=cfg.TRAIN.NUM_EPOCHS,
            steps_per_epoch=int(num_batch_train),
            pct_start=0.2,
            anneal_strategy='cos',
            cycle_momentum=True,
            div_factor=25.0,
            final_div_factor=100.0
        )
    elif cfg.OPTIM.SCHEDULER == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(num_batch_train),
            gamma=0.9
        )
    else:
        raise NotImplementedError

    # set amp training
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = 0

    # set loss function
    top_k_percent_pixels = cfg.OPTIM.TOP_K_PERCENT_PIXELS
    
    if cfg.OPTIM.LOSS == 'wce':
        weight = torch.tensor(ClassWeightSemikitti.get_weight()).cuda()
        WCE = torch.nn.CrossEntropyLoss(reduction='none', weight=weight).cuda()
    elif cfg.OPTIM.LOSS == 'dice':
        WCE = CrossEntropyDiceLoss(reduction='none', weight=None).cuda()
    else:
        raise NotImplementedError
    
    if cfg.OPTIM.IF_LS_LOSS:
        LS = Lovasz_softmax(ignore=0).cuda()
    
    if cfg.OPTIM.IF_BD_LOSS:
        BD = BoundaryLoss().cuda()

    # set running info
    info = {"train_loss": 0, "train_acc": 0, "train_iou": 0,
            "valid_loss": 0, "valid_acc": 0, "valid_iou": 0,
            "best_train_iou": 0, "best_val_iou": 0}
    
    # set evaluator
    if cfg.DATA.DATASET == 'nuscenes':
        evaluator = iouEval(device, n_classes=16+1, ignore=0)
    elif cfg.DATA.DATASET == 'semantickitti' or cfg.DATA.DATASET == 'scribblekitti':
        evaluator = iouEval(device, n_classes=19+1, ignore=0)
    else:
        raise NotImplementedError

    # whether to resume
    if args.resume_from:
        logger.info("Loading epoch, model, optimizer, scaler, and scheduler states from '{}' ...".format(args.resume_from))
        start_epoch = resume(args.resume_from, model, optimizer, scaler, scheduler, logger, args.amp)
        start_epoch += 1
    else:
        start_epoch = 0

    model = model.cuda()

    # train/val steps start here
    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS):

        meter_loss = AverageMeter()
        meter_acc  = AverageMeter()
        meter_iou  = AverageMeter()
        meter_wce  = AverageMeter()
        meter_ls   = AverageMeter()
        meter_bd   = AverageMeter()

        model.train()

        for idx, (scan, label, _, name) in enumerate(loader_train):

            optimizer.zero_grad()
            lr = scheduler.get_last_lr()[0]
            bs = scan.size(0)
            
            scan  = scan.cuda()  # [bs, 6, H, W]
            label = torch.squeeze(label, axis=1).cuda()  # [bs, H, W]

            with torch.cuda.amp.autocast(enabled=args.amp):

                if cfg.MODEL.MODALITY == 'range':
                    logits = model(scan)
                    if logits.size()[-1] != label.size()[-1] and logits.size()[-2] != label.size()[-2]:
                        logits = F.interpolate(logits, size=label.size()[1:], mode='bilinear', align_corners=True)  # [bs, cls, H, W]

                elif cfg.MODEL.MODALITY == 'voxel':
                    pass

                pixel_losses = WCE(logits, label)
                pixel_losses = pixel_losses.contiguous().view(-1)

                if top_k_percent_pixels == 1.0:
                    loss_ce = pixel_losses.mean()
                else:
                    top_k_pixels = int(top_k_percent_pixels * pixel_losses.numel())
                    pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
                    loss_ce = pixel_losses.mean()
                
                loss_ls = LS(F.softmax(logits, dim=1), label)

                if args.if_bd_loss:
                    loss_bd = BD(F.softmax(logits, dim=1), label)
                else:
                    loss_bd = 0.

                loss = 1.0 * loss_ce + 3.0 * loss_ls.mean() + 1.0 * loss_bd

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                scheduler.step()

            with torch.no_grad():
                evaluator.reset()
                argmax = F.softmax(logits, dim=1).argmax(dim=1)
                evaluator.addBatch(argmax, label)
                accuracy = evaluator.getacc()
                jaccard, _ = evaluator.getIoU()

            meter_loss.update(loss.item(), bs)
            meter_acc.update(accuracy.item(), bs)
            meter_iou.update(jaccard.item(), bs)
            meter_wce.update(loss_ce.item(), bs)

            if cfg.OPTIM.IF_LS_LOSS:
                meter_ls.update(loss_ls.item(), bs)
            else:
                meter_ls.update(loss_ls, bs)

            if cfg.OPTIM.IF_BD_LOSS:
                meter_bd.update(loss_bd.item(), bs)
            else:
                meter_bd.update(loss_bd, bs)

            if idx % 10 == 0:
                logger.info(
                    'epoch: [{0}][{1}/{2}], ' 'acc: {acc.val:.3f} ({acc.avg:.3f}), ' 'iou: {iou.val:.3f} ({iou.avg:.3f}), '
                    'loss: {loss.val:.3f} ({loss.avg:.3f}), ' 'wce: {wce.val:.3f} ({wce.avg:.3f}), ' 'ls: {ls.val:.3f} ({ls.avg:.3f}), ' 'bd: {bd.val:.3f} ({bd.avg:.3f}), '
                    'lr: {lr:.6f} '.format(
                        epoch, idx, num_batch_train, acc=meter_acc, iou=meter_iou,
                        loss=meter_loss, wce=meter_wce, ls=meter_ls, bd=meter_bd,
                        lr=lr
                    )
                )

            if epoch >= args.epoch_start_val:

                # validation
                loss, acc, iou = validate(
                    logger=logger,
                    loader_val=loader_val,
                    evaluator=evaluator,
                    model=model,
                    criterion=[WCE, LS],
                    info=info,
                    device=device,
                    args=args
                )

                logger.info("*" * 80)

                # update info
                info["valid_loss"] = loss
                info["valid_acc"]  = acc
                info["valid_iou"]  = iou

                logger.info("Current mIoU is '{:.1f}'%, while the previous best mIoU is '{:.1f}'%.".format(
                    info["valid_iou"] * 100, info['best_val_iou'] * 100)
                )

                if info['valid_iou'] > info['best_val_iou']:
                    info['best_val_iou'] = info['valid_iou']

                    # save checkpoint
                    logger.info('Taking snapshot ...')
                    torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_so_far') + '.pth')

                # switch back to train mode
                model.train()

        # update info
        info["train_loss"] = meter_loss.avg
        info["train_acc"]  = meter_acc.avg
        info["train_iou"]  = meter_iou.avg

        if info['train_iou'] > info['best_train_iou']:
            info['best_train_iou'] = info['train_iou']

        # save everything
        logger.info('Saving checkpoint ...')
        save_checkpoint(epoch, model, optimizer, scaler, scheduler, args.log_dir, amp=args.amp)
    

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()
    for key, value in model_state.items():
        model_state_cpu[key] = value.cpu()
    return model_state_cpu


def save_checkpoint(epoch, model, optimizer, scaler, scheduler, log_dir, amp):
    name = os.path.join(log_dir, 'checkpoint' + '.pth')
    
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state = model_state_to_cpu(model.module.state_dict())
    else:
        model_state = model_state_to_cpu(model.state_dict())

    checkpoint_state = {}
    checkpoint_state['epoch_state'] = epoch
    checkpoint_state['model_state'] = model_state
    checkpoint_state['optimizer_state'] = optimizer.state_dict()
    if amp:
        checkpoint_state['scaler_state'] = scaler.state_dict()
    else:
        pass
    checkpoint_state['scheduler_state'] = scheduler.state_dict()

    torch.save(checkpoint_state, name)


def resume(resume_from, model, optimizer, scaler, scheduler, logger, amp):
    if not os.path.isfile(resume_from):
        raise FileNotFoundError

    checkpoint = torch.load(resume_from, map_location='cpu')

    # epoch
    resume_epoch = checkpoint['epoch_state']

    # model
    model_state_dict = checkpoint['model_state']
    model.load_state_dict(model_state_dict)
    model = model.cuda()

    # optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    # scaler
    if amp:
        scaler.load_state_dict(checkpoint['scaler_state'])
    else:
        pass

    # scheduler
    scheduler.load_state_dict(checkpoint['scheduler_state'])

    return resume_epoch

