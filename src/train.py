import torch

# training utilities functions


def accuracy(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)



def train(gnn, trainloader, testloader, **kwargs):

    if 'epochs' in kwargs:
        epochs = kwargs['epochs']
    else:
        epochs = 200
    if 'patience' in kwargs:
        patience = kwargs['patience']
    else:
        patience = 20
    if 'lr' in kwargs:
        lr = kwargs['lr']
    else:
        lr = 1e-3
    if 'weight_decay' in kwargs:
        weight_decay = kwargs['weight_decay']
    else:
        weight_decay = 1e-5

    criterion = gnn.criterion
    optimizer = torch.optim.Adam(gnn.parameters(), lr=lr, weight_decay=weight_decay)

    counter = 0
    best_val_loss = torch.inf

    train_losses_log = []
    test_losses_log = []
    test_acc_log = []

    gnn.train()


    for epoch in range(epochs):
        total_loss = torch.tensor(0.0)
        for data in trainloader:
            optimizer.zero_grad()
            out = gnn(data.x, data.edge_index, data.batch)
            curr_l = criterion(out, data.y)
            curr_l.backward()
            optimizer.step()
            total_loss += curr_l.item()

        total_loss /= len(trainloader)
        train_losses_log.append(total_loss)
        

        with torch.no_grad():
            total_test_loss = torch.tensor(0.0)
            for data in testloader:
                out = gnn(data.x, data.edge_index, data.batch)
                curr_l = criterion(out, data.y)
                total_test_loss += curr_l.item()
            total_test_loss /= len(testloader)
            test_losses_log.append(total_test_loss)

            test_acc = accuracy(gnn, testloader)
            test_acc_log.append(test_acc)


            if total_test_loss < best_val_loss:
                best_val_loss = total_test_loss
                counter = 0
            else:
                counter += 1

        print(f'Epoch {epoch} - Training loss: {total_loss}; Test loss: {total_test_loss}; Test accuracy: {test_acc}')

        if counter == patience:
            break

    return train_losses_log, test_losses_log, test_acc_log
