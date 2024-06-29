from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import json 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from huggingface_hub import login, create_repo
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """        
        
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=top_n_genres, problem_type="multi_label_classification")
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None 
        

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        self.data = data
        pass 

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        number = {}
        for movie in self.data :
            genres = movie['genres'] 
            for genre in genres : 
                if genre not in number.keys() : 
                    number[genre] = 0 
                number[genre] += 1
        arr = []
        for genre in number.keys() :
            arr.append((genre,number[genre]))
        arr = sorted(arr, key=lambda x:x[1], reverse=True)[0:self.top_n_genres] 
        top_genres = [x[0] for x in arr]
        indexing_top_genres = {}
        for i, genre in enumerate(top_genres) : 
            indexing_top_genres[genre] = i
            
        new_data = []
        all_genres = []
        for i,movie in enumerate(self.data) : 
            genres = movie['genres'] 
            new_genres = []
            for genre in genres : 
                if genre in top_genres : 
                    new_genres.append(genre) 
            
            if len(new_genres) == 0 : continue
            new_data.append(movie)
            all_genres.append(new_genres)

        mlb = MultiLabelBinarizer(classes=top_genres)
        all_genres = mlb.fit_transform(all_genres)
        for i,movie in enumerate(new_data):
            new_data[i]['genres'] = all_genres[i]
        self.data = new_data

        pass 

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        N = len(self.data) 
        test_size = int(test_size * N) 
        train_size = N - test_size  
        train_data, test_data = train_test_split(self.data, test_size=test_size, train_size=train_size, shuffle=True)
        
        new_test_size = int(test_size / 2)
        val_size = test_size - new_test_size
        val_data, test_data = train_test_split(test_data, test_size=new_test_size, train_size=val_size, shuffle=True)
        self.train_data = train_data
        self.val_data = val_data 
        self.test_data = test_data
        return train_data, val_data, test_data

    def create_dataset(self, texts, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        return IMDbDataset(self.tokenizer(list(map(str, texts)), truncation=True, padding=True, max_length=400), labels)
    def convert_data_to_dataset(self) : 
        self.train_data = self.create_dataset([self.train_data[i]['first_page_summary'] for i in range(len(self.train_data))], 
                                              [self.train_data[i]['genres']             for i in range(len(self.train_data))])
        self.val_data   = self.create_dataset([self.val_data[i]['first_page_summary']   for i in range(len(self.val_data))],   
                                              [self.val_data[i]['genres']               for i in range(len(self.val_data))])
        self.test_data  = self.create_dataset([self.test_data[i]['first_page_summary']  for i in range(len(self.test_data))],  
                                              [self.test_data[i]['genres']              for i in range(len(self.test_data))])
        pass 

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        self.model = trainer
        pass 

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        y_true = pred.label_ids
        y_pred = torch.sigmoid(torch.tensor(pred.predictions)).round().numpy()
        
        out = {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, average='samples'), 'recall': recall_score(y_true, y_pred, average='samples'), 'f1': f1_score(y_true, y_pred, average='samples')}
        return out

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        return self.model.evaluate(self.test_data)
    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        self.model.save_model(model_name)
        self.tokenizer.save_pretrained(model_name)
        access_token = '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'
        login(access_token)
        create_repo(repo_id=model_name)
        self.model.push_to_hub(model_name)
        self.tokenizer.push_to_hub(model_name)


class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)