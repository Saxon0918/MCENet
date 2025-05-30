from torch import nn
import sys
from src.utils import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.eval_metrics import *
from modules import models


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    # get model
    model = getattr(models, hyp_params.model)(hyp_params)
    if hyp_params.use_cuda:
        model = model.cuda()

    # get optimizer and loss function
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model, 'optimizer': optimizer, 'criterion': criterion, 'scheduler': scheduler}

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs + 1):
        if hyp_params.dataset == 'ADNI':
            train(model, optimizer, criterion, train_loader, hyp_params, epoch)
            val_loss, _, _ = evaluate(model, criterion, valid_loader, hyp_params, test=False)
            test_loss, _, _ = evaluate(model, criterion, test_loader, hyp_params, test=True)
        else:
            train_My(model, optimizer, criterion, train_loader, hyp_params, epoch)
            val_loss, _, _ = evaluate_My(model, criterion, valid_loader, hyp_params, test=False)
            test_loss, _, _ = evaluate_My(model, criterion, test_loader, hyp_params, test=True)

        scheduler.step(val_loss)  # Decay learning rate by validation loss

        print("-" * 50)
        print(
            'Epoch {:2d} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, val_loss, test_loss))
        print("-" * 50)

        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    model = load_model(hyp_params, name=hyp_params.name)
    if hyp_params.dataset == 'ADNI':
        _, preds, labels = evaluate(model, criterion, test_loader, hyp_params, test=True)
    else:
        _, preds, labels = evaluate_My(model, criterion, test_loader, hyp_params, test=True)

    if hyp_params.classes == 2:
        calculate_metric(preds, labels)   # two class
    else:
        calculate_three_metric(preds, labels)  # three class
    sys.stdout.flush()
    input('[Press Any Key to start another run]')


def train(model, optimizer, criterion, train_loader, hyp_params, epoch):
    epoch_loss = 0
    model.train()
    num_batches = hyp_params.n_train // hyp_params.batch_size  # 样本数/batch_size
    proc_loss, proc_size = 0, 0
    conloss = models.ConLoss(temperature=0.5)

    for i_batch, data in enumerate(train_loader):
        mri = data['mri']
        av45 = data['av45']
        fdg = data['fdg']
        gene = data['gene']
        label = data['label']
        model.zero_grad()
        if hyp_params.use_cuda:
            with torch.cuda.device(0):
                mri, av45, fdg, gene, label = mri.cuda(), av45.cuda(), fdg.cuda(), gene.cuda(), label.cuda()

        batch_size = mri.size(0)
        net = nn.DataParallel(model) if batch_size > 10 else model
        pred, hiddens = net(mri, av45, fdg, gene)
        pred_loss = criterion(pred, label)
        con_loss = conloss(hiddens, label)
        total_loss = pred_loss + 0.1 * con_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)  # 梯度裁剪
        optimizer.step()
        proc_loss += total_loss.item() * batch_size
        proc_size += batch_size
        epoch_loss += total_loss.item() * batch_size
        if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
            avg_loss = proc_loss / proc_size
            print('Epoch {:2d} | Batch {:3d}/{:3d} | Train Loss {:5.4f}'.
                  format(epoch, i_batch, num_batches, avg_loss))
            proc_loss, proc_size = 0, 0

    return epoch_loss / hyp_params.n_train


def evaluate(model, criterion, loader, hyp_params, test=False):
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for i_batch, data in enumerate(loader):
            mri = data['mri']
            av45 = data['av45']
            fdg = data['fdg']
            gene = data['gene']
            label = data['label']
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    mri, av45, fdg, gene, label = mri.cuda(), av45.cuda(), fdg.cuda(), gene.cuda(), label.cuda()

            batch_size = mri.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            pred, _ = net(mri, av45, fdg, gene)
            total_loss += criterion(pred, label).item() * batch_size
            preds.append(pred)
            labels.append(label)

    avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    return avg_loss, preds, labels


def train_My(model, optimizer, criterion, train_loader, hyp_params, epoch):
    epoch_loss = 0
    model.train()
    num_batches = hyp_params.n_train // hyp_params.batch_size
    proc_loss, proc_size = 0, 0

    for i_batch, data in enumerate(train_loader):
        mri = data['mri']
        ct = data['ct']
        clinical = data['clinical']
        label = data['label']
        model.zero_grad()
        if hyp_params.use_cuda:
            with torch.cuda.device(0):
                mri, ct, clinical, label = mri.cuda(), ct.cuda(), clinical.cuda(), label.cuda()

        batch_size = mri.size(0)
        net = nn.DataParallel(model) if batch_size > 10 else model
        pred, hiddens = net(mri, ct, clinical)
        total_loss = criterion(pred, label)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
        optimizer.step()
        proc_loss += total_loss.item() * batch_size
        proc_size += batch_size
        epoch_loss += total_loss.item() * batch_size
        if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
            avg_loss = proc_loss / proc_size
            print('Epoch {:2d} | Batch {:3d}/{:3d} | Train Loss {:5.4f}'.
                  format(epoch, i_batch, num_batches, avg_loss))
            proc_loss, proc_size = 0, 0

    return epoch_loss / hyp_params.n_train


def evaluate_My(model, criterion, loader, hyp_params, test=False):
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for i_batch, data in enumerate(loader):
            mri = data['mri']
            ct = data['ct']
            clinical = data['clinical']
            label = data['label']
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    mri, ct, clinical, label = mri.cuda(), ct.cuda(), clinical.cuda(), label.cuda()

            batch_size = mri.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            pred, _ = net(mri, ct, clinical)
            total_loss += criterion(pred, label).item() * batch_size
            preds.append(pred)
            labels.append(label)

    avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    return avg_loss, preds, labels