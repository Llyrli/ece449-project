import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os
import argparse
import pickle
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


"""--------------------- Data Processing ---------------------"""

# dataset loading
def dataloader(data_path, process_data_path, save_json=True):
    x = []
    y = []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        ham_path = os.path.join(folder_path, 'ham')
        spam_path = os.path.join(folder_path, 'spam')
        for file in os.listdir(ham_path):
            with open(os.path.join(ham_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                x.append(f.read())
                y.append(0)
        for file in os.listdir(spam_path):
            with open(os.path.join(spam_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                x.append(f.read())
                y.append(1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
    print(f"Train size: {len(x_train)}, Test size: {len(x_test)}")
    
    os.makedirs(process_data_path, exist_ok=True)
    text_data_path = os.path.join(process_data_path, "text_data.pkl")
    with open(text_data_path, 'wb') as f:
        pickle.dump((x_train, x_test, y_train, y_test), f)
    
    
    if save_json:
        import json
        data_sizes = {
            "Train size": len(x_train),
            "Test size": len(x_test)
        }
        json_file_path = os.path.join(process_data_path, "data_description.json")
        with open(json_file_path, "w") as json_file:
            json.dump(data_sizes, json_file, indent=4)
        print(f"Data sizes saved to {json_file_path}")
    
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    pickle_path = os.path.join(process_data_path, "train_test_data.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump((x_train, x_test, y_train, y_test, vectorizer), f)

    
    return x_train, x_test, y_train, y_test, vectorizer

# data retransform using TfidfVectorizer
def classify_new_email(vectorizer, classifier, email_text, tokenizer_=None):
    if tokenizer_ is None:
        email_sample = vectorizer.transform([email_text])
    else:
        email_dataset = SpamDataset([email_text], [2], tokenizer_)
        email_samples = DataLoader(email_dataset, batch_size=1, shuffle=False)
        email_sample = None
        for sample in email_samples:
            email_sample = sample
    prediction = classifier.predict(email_sample)
    return "Spam" if prediction[0] == 1 else "Ham"

"""--------------------- Models ---------------------"""
def naive_bayes_exp(x_train, x_test, y_train, y_test):
    classifier = MultinomialNB()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return classifier, accuracy


def random_forest_exp(x_train, x_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    # rescale the data
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return classifier, accuracy

class SpamClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SpamClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return self.softmax(x)
        
        def predict(self, x):
            x = x.toarray()
            x = torch.tensor(x, dtype=torch.float32)
            outputs = self.forward(x)
            return torch.argmax(outputs, dim=1)

def mlp_exp(x_train, x_test, y_train, y_test):

    def train_model(model, dataloader, criterion, optimizer, epochs=10, save_model=True):
        model_path = './models/pytorch_mlp_model.pth'
        if os.path.exists(model_path):
            try:
                model = torch.load(model_path)
                print(f"Model loaded from {model_path}")
                return model
            except:
                print("Model loading failed, training new model...")
        
        model.train()
        loss_list = []
        for epoch in range(epochs):
            total_loss = 0
            for inputs, labels in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
            loss_list.append(total_loss)
        
        if save_model:
            model_dir = './models'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "pytorch_mlp_model.pth")
            torch.save(model, model_path)
            print(f"Model saved to {model_path}")

        return model
    
    def evaluate_model(model, test_loader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()

        # avg_loss = total_loss / len(test_loader)
        accuracy = correct / len(test_loader.dataset)
        return accuracy
        
    # sparse-matrix to pytorch tensor
    x_train = torch.tensor(x_train.toarray(), dtype=torch.float32) 
    y_train = torch.tensor(y_train, dtype=torch.long) 
    x_test = torch.tensor(x_test.toarray(), dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # reduce data size for faster training
    miner = 50
    x_train = x_train[: int(x_train.shape[0]/miner), :]
    y_train = y_train[: int(y_train.shape[0]/miner)]
    x_test = x_test[: int(x_test.shape[0]/miner), :]
    y_test = y_test[: int(y_test.shape[0]/miner)]

    # dataloader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # initialize model
    input_size = x_train.shape[1]
    hidden_size = 128
    output_size = 2
    model = SpamClassifier(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model = train_model(model, train_loader, criterion, optimizer, epochs=20)

    # evaluate model
    accuracy = evaluate_model(model, test_loader, criterion)
    return model, accuracy


class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)
    
class BertSpamClassifier(nn.Module):
    def __init__(self, model_name):
        super(BertSpamClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)  # 二分类任务

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 输出
        return self.fc(cls_output)
    
    def predict(self, x):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        outputs = self.forward(input_ids, attention_mask)
        return torch.argmax(outputs, dim=1)


def bert_exp(x_train, x_test, y_train, y_test):

    def train_model(model, train_loader, optimizer, criterion, epochs=40, save_model=True):
        if os.path.exists('./models/bert_model.pth'):
            try:
                model = torch.load('./models/bert_model.pth')
                print("Model loaded from ./models/bert_model.pth")
                return model
            except:
                print("Model loading failed, training new model...")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            loss_list = []
            for batch in train_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                loss_list.append(loss.item())
        
        if save_model:
            model_dir = './models'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "bert_model.pth")
            torch.save(model, model_path)
            print(f"Model saved to {model_path}")
        
        return model
    

    def evaluate_model(model, test_loader, criterion):
        model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()

        accuracy = correct / len(test_loader.dataset)
        return accuracy


    # model_name = "prajjwal1/bert-tiny"  # or "distilbert-base-uncased"
    model_name = "distilbert-base-uncased"  # or "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # sparse-matrix to pytorch tensor
    miner = 100
    x_train = x_train[: int(len(x_train)/miner)]
    y_train = y_train[: int(len(y_train)/miner)]
    x_test = x_test[: int(len(x_test)/miner)]
    y_test = y_test[: int(len(y_test)/miner)]
    
    train_dataset = SpamDataset(x_train, y_train, tokenizer)
    test_dataset = SpamDataset(x_test, y_test, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = BertSpamClassifier(model_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    model = train_model(model, train_loader, optimizer, criterion, epochs=10, save_model=True)
    accuracy = evaluate_model(model, test_loader, criterion)
    return model, accuracy, tokenizer



"""--------------------- Main Function ---------------------"""

def plot_accuracy(accuracy_x, accuracy_y):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))  
    plt.bar(accuracy_x, accuracy_y, width=0.4)

    plt.title("Accuracy of Different Models")
    plt.xlabel("Model Name")
    plt.ylabel("Accuracy (%)")

    for i, value in enumerate(accuracy_y):
        plt.text(i, value + 0.01, f"{value * 100:.2f}%", ha='center', va='bottom')  # 转换为百分数显示

    pic_path = './results/accuracy_plot.png'
    plt.savefig(pic_path)

    plt.show()


def read_test_files(data_path):
    ham_path = os.path.join(data_path, 'ham')
    spam_path = os.path.join(data_path, 'spam')
    ham_list = []
    spam_list = []
    for file in os.listdir(ham_path):
        with open(os.path.join(ham_path, file), 'r', encoding='utf-8', errors='ignore') as f:
            ham_list.append(f.read())
    for file in os.listdir(spam_path):
        with open(os.path.join(spam_path, file), 'r', encoding='utf-8', errors='ignore') as f:
            spam_list.append(f.read())
    return ham_list, spam_list


def classify_dataset(model_name, vectorizer, classifier, ham_list, spam_list, tokenizer_=None):
    ham_dict = {}
    for ham in ham_list:
        if tokenizer_ is None:
            email_sample = vectorizer.transform([ham])
        else:
            email_dataset = SpamDataset([ham], [2], tokenizer_)
            email_samples = DataLoader(email_dataset, batch_size=1, shuffle=False)
            email_sample = None
            for sample in email_samples:
                email_sample = sample
        prediction = classifier.predict(email_sample)
        ham_dict[ham] = "Spam" if prediction[0] == 1 else "Ham"
    spam_dict = {}
    for spam in spam_list:
        if tokenizer_ is None:
            email_sample = vectorizer.transform([spam])
        else:
            email_dataset = SpamDataset([spam], [2], tokenizer_)
            email_samples = DataLoader(email_dataset, batch_size=1, shuffle=False)
            email_sample = None
            for sample in email_samples:
                email_sample = sample
        prediction = classifier.predict(email_sample)
        spam_dict[spam] = "Spam" if prediction[0] == 1 else "Ham"

    import json
    ham_json_file_path = os.path.join('./results', f"{model_name}-ham_predictions.json")
    with open(ham_json_file_path, "w") as ham_json_file:
        json.dump(ham_dict, ham_json_file, indent=4)

    # Save spam_dict to a separate JSON file
    spam_json_file_path = os.path.join('./results', f"{model_name}-spam_predictions.json")
    with open(spam_json_file_path, "w") as spam_json_file:
        json.dump(spam_dict, spam_json_file, indent=4)

    print(f"Ham predictions saved to {ham_json_file_path}")
    print(f"Spam predictions saved to {spam_json_file_path}")
    
    return 


def main(raw_data_path, process_data_path, model_list, predict_bool, visual_data, test_dataset, test_dataset_path, plot=True):
    # get model
    if not os.path.exists(process_data_path):
        x_train, x_test, y_train, y_test, vectorizer = dataloader(raw_data_path, process_data_path)
    else:  
        pickle_path = os.path.join(process_data_path, "train_test_data.pkl")
        with open(pickle_path, "rb") as f:
            x_train, x_test, y_train, y_test, vectorizer = pickle.load(f)
    # get models to be tested
    model_list = model_list.split(',')
    accuracy_x = []
    accuracy_y = []
    for model_name in model_list:
        # bert needs raw text data and its own tokenizer
        if model_name == 'bert':
            pickle_path = os.path.join(process_data_path, "text_data.pkl")
            with open(pickle_path, "rb") as f:
                x_train, x_test, y_train, y_test = pickle.load(f)
        if model_name in model_dict.keys():
            print(f"Testing {model_name} model...")
            if model_name == 'bert':
                classifier, accuracy, tokenizer = model_dict[model_name](x_train, x_test, y_train, y_test)
                accuracy_x.append(model_name)
                accuracy_y.append(accuracy)
            else:   
                classifier, accuracy = model_dict[model_name](x_train, x_test, y_train, y_test)
                accuracy_x.append(model_name)
                accuracy_y.append(accuracy)
            print(f"Accuracy: {accuracy:.4f}")
            if predict_bool:
                with open(visual_data, 'r', encoding='utf-8') as file:
                    for line in file:
                        print(f'Test Message: {line.strip()}')
                        if model_name == 'bert':
                            result = classify_new_email(vectorizer, classifier, line.strip(), tokenizer)
                        else:
                            result = classify_new_email(vectorizer, classifier, line.strip())
                        print(f"Prediction: {result}")
            if test_dataset:
                ham_list, spam_list = read_test_files(test_dataset_path)
                if model_name == 'bert':
                    classify_dataset(model_name, vectorizer, classifier, ham_list, spam_list, tokenizer)
                else:
                    classify_dataset(model_name, vectorizer, classifier, ham_list, spam_list)
        else:
            print(f"Model {model_name} not found in model_dict")

    if plot:
        plot_accuracy(accuracy_x, accuracy_y)
    



model_dict = {'bayes': naive_bayes_exp, 'random_forest': random_forest_exp, 'pytorch_mlp': mlp_exp, 'bert': bert_exp}

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default=None)
    parser.add_argument("--process_data_path", type=str, default=None)
    parser.add_argument("--model_list", type=str, default='models need to use')
    parser.add_argument("--predict_bool", action='store_true', help="Enable prediction for new email")
    parser.add_argument("--visual_data", type=str, default='./visual_data.txt')
    parser.add_argument("--test_dataset", action='store_true', help="if use ECE_449_dataset for model evaluation")
    parser.add_argument("--test_dataset_path", type=str, default=None)
    args = parser.parse_args()
    
    raw_data_path = args.raw_data_path 
    process_data_path = args.process_data_path
    model_list = args.model_list
    predict_bool = args.predict_bool
    visual_data = args.visual_data
    test_dataset = args.test_dataset
    test_dataset_path = args.test_dataset_path

    main(raw_data_path, process_data_path, model_list, predict_bool, visual_data, test_dataset, test_dataset_path)

