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
        self.shapes = (self.images.shape, self.labels.shape)
        print("Data loaded successfully :)")

    def stat(self):
        keys = self.classes.keys()
        stat = {}
        for key in keys:
            n = 0
            for label in self.labels:
                if label == key:
                    n += 1
                else:
                    pass
            stat[key] = n
        plt.figure(figsize=(4, 4))
        plt.pie(stat.values(), labels=stat.keys(), normalize=True)
        plt.show()

    def shape(self):
        print(f"Images shape: {self.shapes[0]}",
              f"Labels shape: {self.shapes[1]}\n")

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
            rand = random.randint(0, self.shapes[0][0] - 1)
            if len(self.shapes[0]) == 4:
                ax.imshow(self.images[rand])
            elif len(self.shapes[0]) == 3:
                ax.imshow(self.images[rand], cmap="gray")

            title = f"target: {self.labels[rand]}"
            ax.set_title(title)
        plt.show()

    def dataset(self, split_size, shuffle, random_state, images_format,
                labels_format, permute, one_hot, device):

        if len(self.shapes[0])==3:
            self.images = np.expand_dims(self.images, axis=3)

        elif len(self.shapes[0])==4:
            pass

        if one_hot:
            self.encode()
        else:
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
