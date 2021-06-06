from sklearn.model_selection import train_test_split
import torch


class Data():
    def __init__(self, load):
        super(Data, self).__init__()
        self.images, self.labels = load

    def dataset(self, split_size, shuffle, random_state, images_format,
                labels_format, device):

        x_train, x_val, y_train, y_val = train_test_split(self.images,
                                                          self.labels,
                                                          test_size=split_size, 
                                                          shuffle=shuffle, 
                                                          random_state=random_state)

        x_test, x_val, y_test, y_val = train_test_split(x_val, y_val,
                                                        test_size=0.5,
                                                        shuffle=shuffle,
                                                        random_state=random_state)

        # Verify datasets shapes
        print(f"Train data shape: {x_train.shape}, {y_train.shape}")
        print(f"Test data shape: {x_test.shape}, {y_test.shape}")
        print(f"Validation data shape: {x_val.shape}, {y_val.shape}")

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

        print("\nDataset generated successfully :)")