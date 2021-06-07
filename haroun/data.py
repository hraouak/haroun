from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
import random


class Data():
    def __init__(self, loader, classes):
        super(Data, self).__init__()
        self.images, self.labels = loader
        self.classes = classes
        if self.classes is not None:
            self.encode()
        else:
            pass

    def shape(self):
        print(f"Images shape: {self.images.shape}",
              f"Labels shape: {self.labels.shape}")

    def encode(self):
        total = len(self.classes)
        self.labels = np.array([self.classes[item] for item in self.labels])
        self.labels = np.eye(total)[self.labels]

    def decode(self, item):
        keys = list(self.classes.keys())
        index = np.argmax(item)
        label = keys[index]
        return label

    def show(self):
        f, axis = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        for i, ax in enumerate(axis.flat):
            rand = random.randint(0, self.images.shape[0] - 1)
            ax.imshow(self.images[rand])

            if self.classes is not None: 
                title = f"target: {self.decode(self.labels[rand])}"
            else:
                title = f"target: {self.labels[rand]}"

            ax.set_title(title)
        plt.show()

    def dataset(self, split_size, shuffle, random_state, images_format,
                labels_format, permute, device):

        if len(self.images.shape)==3:
            self.images = np.expand_dims(self.images, axis=3)

        elif len(self.images.shape)==4:
            pass

        x_train, x_val, y_train, y_val = train_test_split(self.images,
                                                          self.labels,
                                                          test_size=split_size,
                                                          shuffle=shuffle,
                                                          random_state=random_state)

        x_test, x_val, y_test, y_val = train_test_split(x_val, y_val,
                                                        test_size=0.5,
                                                        shuffle=shuffle,
                                                        random_state=random_state)

        # Free memory
        del self.images, self.labels

        # Convert Numpy arrays to Torch tensors
        self.train_inputs = torch.from_numpy(x_train).to(images_format).to(device)
        self.train_outputs = torch.from_numpy(y_train).to(labels_format).to(device)
        del x_train, y_train

        self.val_inputs = torch.from_numpy(x_val).to(images_format).to(device)
        self.val_outputs = torch.from_numpy(y_val).to(labels_format).to(device)
        del x_val, y_val

        self.test_inputs = torch.from_numpy(x_test).to(images_format).to(device)
        self.test_outputs = torch.from_numpy(y_test).to(labels_format).to(device)
        del x_test, y_test

        if permute:
            self.train_inputs = self.train_inputs.permute(0, 3, 1, 2)
            self.val_inputs = self.val_inputs.permute(0, 3, 1, 2)
            self.test_inputs = self.test_inputs.permute(0, 3, 1, 2)

        # Verify datasets shapes
        print(f"Train tensor shape: {self.train_inputs.shape}, {self.train_outputs.shape}")
        print(f"Test tensor shape: {self.test_inputs.shape}, {self.test_outputs.shape}")
        print(f"Validation tensor shape: {self.val_inputs.shape}, {self.val_outputs.shape}")

        print("\nDataset generated successfully :)")