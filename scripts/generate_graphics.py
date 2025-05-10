import matplotlib.pyplot as plt
import numpy as np
import os

def read_data_from_txt(file_path):
    epochs = []
    loss = []
    train_accuracy = []
    test_accuracy = []
    time_taken = []
    
    with open(file_path, 'r') as file:
        next(file)
        
        for line in file:
            columns = line.strip().split(",")
            if len(columns) == 5:
                try:
                    epochs.append(int(columns[0]))
                    loss.append(float(columns[1]))
                    train_accuracy.append(float(columns[2].replace('%', '')))
                    test_accuracy.append(float(columns[3].replace('%', '')))
                    time_taken.append(float(columns[4]))
                except ValueError:
                    continue
    
    return epochs, loss, train_accuracy, test_accuracy, time_taken

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

base_folder = 'C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/output_data'
subfolders = [f'w{i}' for i in range(1, 6)]  

w_values = {
    'w1': -2,
    'w2': 0,
    'w3': 0.01,
    'w4': -0.02,
    'w5': -0.002
}

for subfolder in subfolders:
    folder_path = os.path.join(base_folder, subfolder)
    
    gd_path = os.path.join(folder_path, f'gradient_descent{subfolder[-1]}.txt')
    sgd_path = os.path.join(folder_path, f'stochastic_gradient_descent{subfolder[-1]}.txt')
    adam_path = os.path.join(folder_path, f'adam{subfolder[-1]}.txt')
    
    gd_epochs, gd_loss, gd_train_accuracy, gd_test_accuracy, gd_time = read_data_from_txt(gd_path)
    sgd_epochs, sgd_loss, sgd_train_accuracy, sgd_test_accuracy, sgd_time = read_data_from_txt(sgd_path)
    adam_epochs, adam_loss, adam_train_accuracy, adam_test_accuracy, adam_time = read_data_from_txt(adam_path)
    
    gd_loss_smooth = moving_average(gd_loss)
    sgd_loss_smooth = moving_average(sgd_loss)
    adam_loss_smooth = moving_average(adam_loss)

    gd_test_accuracy_smooth = moving_average(gd_test_accuracy)
    sgd_test_accuracy_smooth = moving_average(sgd_test_accuracy)
    adam_test_accuracy_smooth = moving_average(adam_test_accuracy)

    gd_train_accuracy_smooth = moving_average(gd_train_accuracy)
    sgd_train_accuracy_smooth = moving_average(sgd_train_accuracy)
    adam_train_accuracy_smooth = moving_average(adam_train_accuracy)
    
    w_value = w_values[subfolder]
    decimal_places = len(str(w_value).split('.')[-1]) if '.' in str(w_value) else 0
    w_title_format = f'Results for w = {w_value:.{decimal_places}f}'

    plt.figure(figsize=(16, 12))
    plt.suptitle(w_title_format, fontsize=16)

    # Epoch vs Loss
    plt.subplot(3, 2, 1)
    plt.plot(gd_epochs[4:], gd_loss_smooth, label='GD', color='blue', linestyle='-', linewidth=2)
    plt.plot(sgd_epochs[4:], sgd_loss_smooth, label='SGD', color='orange', linestyle='-', linewidth=2)
    plt.plot(adam_epochs[4:], adam_loss_smooth, label='Adam', color='green', linestyle='-', linewidth=2)
    plt.grid(alpha=0.8, linestyle='--', linewidth=0.7)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend()

    # Time vs Loss
    plt.subplot(3, 2, 2)
    plt.plot(gd_time[4:], gd_loss_smooth, label='GD', color='blue', linestyle='-', linewidth=2)
    plt.plot(sgd_time[4:], sgd_loss_smooth, label='SGD', color='orange', linestyle='-', linewidth=2)
    plt.plot(adam_time[4:], adam_loss_smooth, label='Adam', color='green', linestyle='-', linewidth=2)
    plt.grid(alpha=0.8, linestyle='--', linewidth=0.7)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend()

    # Epoch vs Test Accuracy
    plt.subplot(3, 2, 3)
    plt.plot(gd_epochs[4:], gd_test_accuracy_smooth, label='GD', color='blue', linestyle='-', linewidth=2)
    plt.plot(sgd_epochs[4:], sgd_test_accuracy_smooth, label='SGD', color='orange', linestyle='-', linewidth=2)
    plt.plot(adam_epochs[4:], adam_test_accuracy_smooth, label='Adam', color='green', linestyle='-', linewidth=2)
    plt.grid(alpha=0.8, linestyle='--', linewidth=0.7)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Test Accuracy', fontsize=10)
    plt.legend()

    # Time vs Test Accuracy
    plt.subplot(3, 2, 4)
    plt.plot(gd_time[4:], gd_test_accuracy_smooth, label='GD', color='blue', linestyle='-', linewidth=2)
    plt.plot(sgd_time[4:], sgd_test_accuracy_smooth, label='SGD', color='orange', linestyle='-', linewidth=2)
    plt.plot(adam_time[4:], adam_test_accuracy_smooth, label='Adam', color='green', linestyle='-', linewidth=2)
    plt.grid(alpha=0.8, linestyle='--', linewidth=0.7)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Test Accuracy', fontsize=10)
    plt.legend()

    # Epoch vs Train Accuracy
    plt.subplot(3, 2, 5)
    plt.plot(gd_epochs[4:], gd_train_accuracy_smooth, label='GD', color='blue', linestyle='-', linewidth=2)
    plt.plot(sgd_epochs[4:], sgd_train_accuracy_smooth, label='SGD', color='orange', linestyle='-', linewidth=2)
    plt.plot(adam_epochs[4:], adam_train_accuracy_smooth, label='Adam', color='green', linestyle='-', linewidth=2)
    plt.grid(alpha=0.8, linestyle='--', linewidth=0.7)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Train Accuracy', fontsize=10)
    plt.legend()

    # Time vs Train Accuracy
    plt.subplot(3, 2, 6)
    plt.plot(gd_time[4:], gd_train_accuracy_smooth, label='GD', color='blue', linestyle='-', linewidth=2)
    plt.plot(sgd_time[4:], sgd_train_accuracy_smooth, label='SGD', color='orange', linestyle='-', linewidth=2)
    plt.plot(adam_time[4:], adam_train_accuracy_smooth, label='Adam', color='green', linestyle='-', linewidth=2)
    plt.grid(alpha=0.8, linestyle='--', linewidth=0.7)
    plt.xlabel('Time (s)', fontsize=10)
    plt.ylabel('Train Accuracy', fontsize=10)
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()
