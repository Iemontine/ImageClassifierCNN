import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from PIL import Image


# Load and preprocess the dataset
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.0,), (1.0,))
])

# Load training and test datasets
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Split training data into training and validation sets
validation_size = 12000
train_size = len(train_dataset) - validation_size
train_data, val_data = random_split(train_dataset, [train_size, validation_size])

# Dataloaders for training, validation, and test sets, 32 observations per batch
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		# 2D convolutional layers
		self.layer1 = nn.Sequential(	
			nn.Conv2d(1, 28, 3),		# 1 input channel, 28 filters, 3x3 window size
			nn.ReLU(),					# ReLU activation
			nn.MaxPool2d(2, 2) 			# 2x2 max pooling
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(28, 56, 3),		# 28 input channels, 56 filters, 3x3 window size
			nn.ReLU(),					# ReLU activation
		)
		# Fully connected layers
		self.fc1 = nn.Linear(56 * 11 * 11, 56)  	# 56x5x5 input nodes, 56 output nodes
		self.relu = nn.ReLU() 					# ReLU activation
		self.fc2 = nn.Linear(56, 10)			# 56 input nodes, 10 output nodes
		self.softmax = nn.Softmax(dim=1)		# Softmax activation

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = x.view(-1, 56 * 11 * 11)
		x = self.relu(self.fc1(x))
		x = self.softmax(self.fc2(x))
		return x

# Initialize the model, sparse categorical cross-entropy loss function, and Adam optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

if os.path.exists('model.pth'):
    model.load_state_dict(torch.load('model.pth'))
    print("Model loaded, skipping training.")
else:
	# Training the model for 10 epochs
	train_accuracies = []
	val_accuracies = []
	for epoch in range(10):
		model.train()
		running_loss = 0.0
		correct_train = 0
		total_train = 0
		for inputs, labels in train_loader:
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			_, predicted = torch.max(outputs.data, 1)
			total_train += labels.size(0)
			correct_train += (predicted == labels).sum().item()
			running_loss += loss.item()

		train_accuracy = 100 * correct_train / total_train

		# Validation loop
		model.eval()
		correct_val = 0
		total_val = 0
		for inputs, labels in val_loader:
			outputs = model(inputs)
			_, predicted = torch.max(outputs.data, 1)
			total_val += labels.size(0)
			correct_val += (predicted == labels).sum().item()

		val_accuracy = 100 * correct_val / total_val

		train_accuracies.append(train_accuracy)
		val_accuracies.append(val_accuracy)
		print(f"Epoch {epoch+1}: Loss = {running_loss:.3f}, Train Accuracy = {train_accuracy:.2f}%, Val Accuracy = {val_accuracy:.2f}%")
	torch.save(model.state_dict(), 'model.pth')

#Evaluate on test set
correct_test = 0
total_test = 0
test_examples = {}

model.eval()
with torch.no_grad():
	for inputs, labels in test_loader:
		outputs = model(inputs)
		_, predicted = torch.max(outputs.data, 1)
		total_test += labels.size(0)
		correct_test += (predicted == labels).sum().item()

		for i, pred in enumerate(predicted):
			if len(test_examples) < 10 and labels[i] not in test_examples and pred != labels[i]:
				test_examples[labels[i].item()] = (inputs[i], pred, labels[i])
test_accuracy = 100 * correct_test / total_test

# Save misclassified examples
os.makedirs("misclassified", exist_ok=True)
for label, (img, pred, true_label) in test_examples.items():
    if pred != true_label:
        img_path = f"misclassified/True{true_label}_Pred{pred}_{label}.png"
        Image.fromarray(img.squeeze().numpy() * 255).convert('L').save(img_path)

# Model performance evaluation
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("============Model Performance===========")
print(f"Number of Trainable Parameters: {trainable_params}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Number of Misclassified Examples: {len(test_examples)}")

try:
	print(f"Final Train Accuracy: {train_accuracy:.2f}%")
	print(f"Final Validation Accuracy: {val_accuracy:.2f}%")
	# Evaluate training and validation accuracy at the end of each epoch, and plot them as line plots
	plt.figure(figsize=(10, 5))
	plt.plot(range(10), train_accuracies, label='Training Accuracy')
	plt.plot(range(10), val_accuracies, label='Validation Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('Training and Validation Accuracy per Epoch')
	plt.legend()
	plt.show()
except:
	pass