import time
import torch
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(model, criterion, optimizer, scheduler, dics, num_epochs=25):
    since = time.time()

    dataloaders = dics['dataloaders']
    dataset_sizes = dics['dataset_sizes']
    best_loss = 1000000000000000000000

    for epoch in range(1, num_epochs+1):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print('Starting ', phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            batch_nr = 0 
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_nr += 1
                print('Epoch {}/{}. Batch {}/{}'.format(epoch, num_epochs, batch_nr, dataset_sizes[phase]//len(labels)))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.to(device)(inputs)
                    #preds = torch.softmax(outputs, 1)[:,1]
                    loss = criterion(outputs, labels)
                    #Average loss for batch

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) #Needs to de-average batch loss
#            if phase == 'train':
#                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            print('Epoch {} loss : {}'.format(epoch, epoch_loss))
            if best_loss > epoch_loss:
                best_loss = epoch_loss
            torch.save(model.cpu().state_dict(), os.path.join('Models', 'model_' + str(epoch) + '.pth'))
            with open("Models/scores.txt", "a") as myfile:
                myfile.write('Epoch {}. {} loss : {} \n'.format(epoch, phase, epoch_loss))



        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model



def test_model(model, dics):
    dataloaders = dics['dataloaders']
    dataset_sizes = dics['dataset_sizes']
    
    batch_nr = 0
    predictions = []
    all_labels = []
    model.eval()   # Set model to evaluate mode
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_nr += 1
        print('Batch {}/{}'.format(batch_nr, dataset_sizes['test']//len(labels)))
        outputs = model(inputs)
        preds = torch.softmax(outputs, 1)[:,1]
        predictions = predictions + preds.tolist()
        all_labels = all_labels + labels.tolist()
    return predictions, all_labels