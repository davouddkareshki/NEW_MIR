import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
import tqdm 

from data_loader import ReviewLoader
from basic_classifier import BasicClassifier
from sklearn.model_selection import train_test_split


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.val_loader = None
        #self.device = 'mps' if torch.backends.mps.is_available else 'cpu'
        #self.device = 'cuda' if torch.cuda.is_available() else self.device
        #self.model.to(self.device)
        #print(f"Using device: {self.device}")

    def fit(self, X, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        dataset = ReviewDataSet(X,y)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in tqdm.tqdm(range(self.num_epochs)):

            self.model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.long())
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                #print(self._eval_epoch(self.test_loader, self.model))
        #    train_losses.append(train_loss / len(train_loader))
        eval_epoch = self._eval_epoch(self.val_loader, self.model)
        print('evaluation on validation data :')
        print('loss', eval_epoch[0])
        print('f1_score',eval_epoch[3])
    print('--------------------------------------')
    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        feature = torch.tensor([x], dtype=torch.float)
        output = self.model(feature)
        _, predicted = torch.max(output, 1)
        return predicted.numpy()[0]

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, model):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        self.model.eval()
        all_loss = 0.0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, labels.long())
                #print(len(outputs))
                _,predicted = torch.max(outputs, 1)
                for pred in predicted : 
                    all_outputs.append(pred)
                for label in labels.long() : 
                    all_labels.append(label)
                all_loss += loss.item()
        return all_loss, all_outputs, all_labels, f1_score(all_labels, all_outputs)
    
    def set_val_dataloader(self, X_val, y_val):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        val_dataset = ReviewDataSet(X_val,y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = val_loader
        pass

    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        test_dataset = ReviewDataSet(X_test,y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = test_loader
        pass

    def prediction_report(self, X, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        y_pred = [self.predict(x) for x in tqdm.tqdm(X)]
        #print(y_pred)
        return classification_report(y, y_pred)


# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    RL = ReviewLoader() 
    data = pd.read_csv('vectorized_IMDB_reviews.csv') 
    #data = data[0:1000]
    y = np.array(data['label'])
    for i in range(len(y)) : 
        if y[i] == -1 : 
            y[i] = 0
    #print(np.unique(y))
    #print('------------')
    data['label'] = y
    X_train, X_test, y_train, y_test = RL.split_data(data)
    test_size = int(len(X_test)/2)
    val_size = len(X_test)-test_size 
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_size, train_size=val_size, shuffle=True)  
        
    #print(np.unique(y_test))
    batch_size = 1000
    model = DeepModelClassifier(X_train.shape[1],len(np.unique(data['label'])), batch_size)
    #print('************')
    model.set_val_dataloader(X_val, y_val)
    model.set_test_dataloader(X_test, y_test)
    model.fit(X_train, y_train)
    print('--------------------------------------')
    print(model.prediction_report(X_test, y_test))
    pass
# F1 score between 84% and 86%