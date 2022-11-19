import os
from tqdm import tqdm
import torch

class Trainer():
    def __init__(self, model, trainloader, testloader, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def train(self, num_epochs, checkpoint_dir):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in tqdm(enumerate(self.trainloader), total=len(self.trainloader), desc=f'Epoch(training): {epoch+1}/{num_epochs}'):
                lr_patch, target = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                output = self.model(lr_patch)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            print(f"Training Loss: {running_loss/len(self.trainloader)}")
            
            self.validate(epoch, num_epochs)
            self.scheduler.step()
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict()},
                       os.path.join(checkpoint_dir, f'{self.model.name}.pt'))

    def validate(self, epoch, num_epochs):
        with torch.no_grad():
            test_loss = 0
            correct = 0
            adj_correct = 0
            for i, data in tqdm(enumerate(self.testloader), total=len(self.testloader), desc=f'Epoch(validation): {epoch+1}/{num_epochs}'):
                lr_patch, target = data[0].to(self.device, non_blocking=True), data[1].to(self.device, non_blocking=True)
                output = self.model(lr_patch)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                if pred.eq(target.view_as(pred)+1) or pred.eq(target.view_as(pred)-1) or pred.eq(target.view_as(pred)):
                    adj_correct += 1
            
            print(f"Validation Loss: {test_loss / len(self.testloader)}, Acc: {100. * correct / len(self.testloader)}%, Adj_acc: {100. * adj_correct / len(self.testloader)}%")
