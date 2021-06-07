import torch
import numpy as np
import time
import copy
import matplotlib.pyplot as plt


class Model():
    def __init__(self, network, optimizer, criterion, device):
        super(Model, self).__init__()
        self.net = network.to(device)
        self.optim = optimizer
        self.loss = criterion
        self.device = device
        print("Model initialized succssefully :)\n")

    def train(self, train_data, val_data, epochs, patience, batch_size):
        best_loss = np.inf
        self.patience = patience
        self.train_losses = []
        self.val_losses = []
        self.achieved_epochs = []
        train_inputs, train_outputs = train_data
        val_inputs, val_outputs = val_data
        total_train = train_inputs.size()[0]
        total_val = val_inputs.size()[0]
        print("Train loop:\n")

        for epoch in range(epochs):
            t0 = time.time()
            self.net.train()
            train_loss = 0
            val_loss = 0
            self.achieved_epochs.append(epoch)
            train_permutation = torch.randperm(total_train)
            val_permutation = torch.randperm(total_val)

            for i in range(0,total_train, batch_size):
                self.optim.zero_grad()
                indices = train_permutation[i:i+batch_size]
                batch_x, batch_y = train_inputs[indices], train_outputs[indices]
                outputs = self.net(batch_x)
                loss = self.loss(outputs, batch_y)
                loss.backward()
                self.optim.step()
                train_loss += loss
            train_loss = train_loss.cpu().detach() / total_train
            self.train_losses.append(train_loss)

            for j in range(0, total_val, batch_size):
                self.net.eval()
                indices = val_permutation[j:j+batch_size]
                batch_x, batch_y = val_inputs[indices], val_outputs[indices]
                outputs = self.net(batch_x)
                loss = self.loss(outputs, batch_y)
                val_loss += loss
            val_loss = val_loss.cpu().detach() / total_val
            self.val_losses.append(val_loss)
            tf = time.time()

            if val_loss < best_loss:
                best_loss = val_loss
                cost_patience = patience
                self.state_dict = copy.deepcopy(self.net.state_dict())
                print(f"\tEpoch: {epoch+1}/{epochs}, ",
                      f"Epoch duration: {tf-t0:.3g}s, ",
                      f"Train Loss: {train_loss:.3g}, ",
                      f"Val Loss: {val_loss:.3g}")

            else:
                cost_patience -= 1
                if cost_patience < 0:
                    print(f"\nEarly stopping after {patience} epochs of no improvements")
                    break

                else:
                    print(f"\tEpoch: {epoch+1}/{epochs}, ",
                          f"Epoch duration: {tf-t0:.3g}s, ", 
                          f"Train Loss: {train_loss:.3g}, ", 
                          f"Val Loss: {val_loss:.3g} - No improvement",
                          f"-> Remaining patience: {cost_patience}")

        print("\nTrain finished successfully :)")

    def evaluate(self, test_data):
        test_inputs, test_outputs = test_data
        self.net.load_state_dict(self.state_dict)
        predictions = self.net(test_inputs).cpu().detach().numpy()
        
        correct = 0
        wrong = 0
        for i,(j,k) in enumerate(zip(predictions, test_outputs.cpu().detach())):
          if np.argmax(j) == np.argmax(k):
            correct +=1
          else:
            wrong += 1
        
        score = 100 * correct / test_outputs.shape[0]
        print(f'\nTest accuracy:{score:.3g}%')
        print(f'Correct predictions: {correct}, Wrong predictions: {wrong}')

    def save(self, path, checkpoint_name):
        torch.save(self.state_dict, f"{path}/{checkpoint_name}.pth")
        print("\nCheckpoint saved successfully :)")

    def plot(self):
        f, ax = plt.subplots()
        ax.plot(self.achieved_epochs, self.train_losses, label='train')
        ax.plot(self.achieved_epochs, self.val_losses, label='validation')
        ax.set_title('model loss')
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        no_improvement_line = self.achieved_epochs[-1] - self.patience
        ax.axvline(x=no_improvement_line, color='r')
        ax.legend(loc='upper center', frameon=False)
        plt.show()
