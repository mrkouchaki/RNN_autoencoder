#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow_addons.layers import MultiHeadAttention
import matplotlib.pyplot as plt
import os
import struct

totalMagnitude = 0
totalnumberofsamples = 0
def load_data(filepath):
    _, file_extension = os.path.splitext(filepath)
    
    if file_extension == '.csv':
        df = pd.read_csv(filepath)
        samples = df['your_column_name'].values
    elif file_extension == '.dat':
        #Looks like DAT is binary encoded added a b in r as rb
        with open(filepath, 'rb') as f:
            samples = f.readlines()
        samples = np.array([sample.strip() for sample in samples])
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return samples

def count_lines(filepath):
    with open(filepath, 'rb') as f:
        return sum(1 for _ in f)

class DataGenerator:        
    def __init__(self, filepath, batch_size, sequence_length, max_samples=None, for_training=True):
        self.filepath = filepath
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_samples = max_samples
        self.for_training = for_training
        self.samples = []
        self.binary_file = open(self.filepath, 'rb')  # Initialize the binary_file here
        self.reset()

    def reset(self):
        self.total_samples_processed = 0
        _, self.file_extension = os.path.splitext(self.filepath)


    def __iter__(self):
        self.binary_file.seek(0)  # reset file pointer
        self.samples = []
        return self
    
    def close(self):
        if not self.binary_file.closed:
            self.binary_file.close()

    def process_data(self, samples):
        real_parts = []
        imag_parts = []
        for sample in samples:
            try:
                cnum = complex(sample.replace('j', 'j'))
                real_parts.append(np.real(cnum))
                imag_parts.append(np.imag(cnum))
            except ValueError:
                continue

        real_parts = (real_parts - np.mean(real_parts)) / np.std(real_parts)
        imag_parts = (imag_parts - np.mean(imag_parts)) / np.std(imag_parts)

        X = [list(zip(real_parts[i:i+self.sequence_length], imag_parts[i:i+self.sequence_length])) for i in range(len(real_parts) - self.sequence_length)]
        return np.array(X)

    def __next__(self):
        chunksize = self.batch_size * self.sequence_length
        global totalMagnitude  # Access the global variable
        global totalnumberofsamples  # Access the global variable
        
        if self.file_extension == '.dat':
            samples = []
            while True:
                binary_data = self.binary_file.read(8)
                if not binary_data:
                    break 
                decoded_data = struct.unpack('ff', binary_data)
                if decoded_data[0] == 0 and decoded_data[1] == 0:
                    decoded_line = f"0j\n"
                    #Calculates the mangitude of the complex number
                    totalMagnitude += abs(complex(decoded_line)) 
                    totalnumberofsamples +=1
                else:
                    if decoded_data[1] >= 0:
                        decoded_line = f"{decoded_data[0]}+{decoded_data[1]}j\n"
                        #Calculates the mangitude of the complex number
                        totalMagnitude += abs(complex(decoded_line)) 
                        totalnumberofsamples +=1                        
                    else:
                        decoded_line = f"{decoded_data[0]}{decoded_data[1]}j\n"
                        #Calculates the mangitude of the complex number
                        totalMagnitude += abs(complex(decoded_line)) 
                        totalnumberofsamples +=1                       
                samples.append(decoded_line)

                if self.max_samples and self.total_samples_processed >= self.max_samples:
                    raise StopIteration
                self.total_samples_processed += 1
                #print('samples:', samples)
                if len(samples) == chunksize:
                    X_chunk = self.process_data(samples)
                    #print('X_chunk:', X_chunk)
                    if self.for_training:
                        return X_chunk, X_chunk
                    else:
                        return X_chunk
                    samples = []

        
        else:
            raise StopIteration

def plot_with_intrusions4(X_chunk_test, X_chunk_pred, jamming_detected, sequence_length):
    intrusion_indices = np.where(jamming_detected)[0]  

    if len(intrusion_indices) == 0:
        print("No intrusions detected for plotting.")
        return  # Exit the function if no intrusions detected

    plt.figure(figsize=(14, 10))
    legend_added = False  # Control variable for legend

    for idx, intrusion in enumerate(jamming_detected):
        if intrusion:  # Check if intrusion is detected for the sequence
            start_idx = idx * sequence_length
            end_idx = start_idx + sequence_length
            time_steps = np.arange(start_idx, end_idx)

            # Plot original and reconstructed data for the real and imaginary parts
            plt.plot(time_steps, X_chunk_test[idx, :, 0], 'b-', label='Original Real' if not legend_added else "", linewidth=2)
            plt.plot(time_steps, X_chunk_pred[idx, :, 0], 'r--', label='Reconstructed Real' if not legend_added else "", linewidth=2)
            plt.plot(time_steps, X_chunk_test[idx, :, 1], 'g-', label='Original Imag' if not legend_added else "", linewidth=2)
            plt.plot(time_steps, X_chunk_pred[idx, :, 1], 'y--', label='Reconstructed Imag' if not legend_added else "", linewidth=2)
            
            # Highlight the area of intrusion
            plt.fill_between(time_steps, -3, 3, where=intrusion, color=(1, 0.5, 0.5), alpha=0.3, label='Intrusion Detected' if not legend_added else "")
            legend_added = True  # Set the flag to True after adding legend

    # Add legend
    if legend_added:
        plt.legend()
    plt.title('4-IQ Data: Original vs Reconstructed with Intrusion Highlighted')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Value')
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    plt.show()    
#-----------------------------6----------------------------------------
def plot_with_intrusions6(all_X_chunk_test, all_X_chunk_pred, all_intrusion_flags, sequence_length):
    plt.figure(figsize=(14, 10))

    # Plotting one set of lines for legend purposes
    plt.plot([], [], 'b-', label='Original Real', linewidth=2)
    plt.plot([], [], 'r--', label='Reconstructed Real', linewidth=2)
    plt.plot([], [], 'g-', label='Original Imag', linewidth=2)
    plt.plot([], [], 'y--', label='Reconstructed Imag', linewidth=2)
    plt.fill_between([], [], [], color=(1, 0.5, 0.5), alpha=0.3, label='Intrusion Detected')

    for idx in range(0, len(all_X_chunk_test), sequence_length):
        sequence_idx = idx // sequence_length  # Calculate the actual sequence index
        if all_intrusion_flags[sequence_idx]:  # Check intrusion for the specific sequence
            time_steps = np.arange(idx, idx + sequence_length)

            # Plotting without labels for actual data
            plt.plot(time_steps, all_X_chunk_test[idx:idx + sequence_length, :, 0], 'b-', linewidth=2)
            plt.plot(time_steps, all_X_chunk_pred[idx:idx + sequence_length, :, 0], 'r--', linewidth=2)
            plt.plot(time_steps, all_X_chunk_test[idx:idx + sequence_length, :, 1], 'g-', linewidth=2)
            plt.plot(time_steps, all_X_chunk_pred[idx:idx + sequence_length, :, 1], 'y--', linewidth=2)
            
            # Highlighting the area of intrusion without adding to legend
            plt.fill_between(time_steps, -3, 3, where=True, color=(1, 0.5, 0.5), alpha=0.3)

    plt.legend()
    plt.title('6-IQ Data: Original vs Reconstructed with Intrusion Highlighted')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Value')
    plt.tight_layout()
    plt.show()
#--------------------------------------7-------------------------------------
def plot_with_intrusions7(all_X_chunk_test, all_X_chunk_pred, all_intrusion_flags, sequence_length):
    plt.figure(figsize=(14, 10))
    legend_added = False

    for idx in range(0, len(all_X_chunk_test), sequence_length):
        sequence_idx = idx // sequence_length
        if all_intrusion_flags[sequence_idx]:
            time_steps = np.arange(sequence_length)

            real_part_test = all_X_chunk_test[idx, :, 0].reshape(-1)
            imag_part_test = all_X_chunk_test[idx, :, 1].reshape(-1)
            real_part_pred = all_X_chunk_pred[idx, :, 0].reshape(-1)
            imag_part_pred = all_X_chunk_pred[idx, :, 1].reshape(-1)

            # Printing the sliced data
            #print(f"Real Part Test (Sequence {sequence_idx}):", real_part_test)
            #print(f"Imag Part Test (Sequence {sequence_idx}):", imag_part_test)
            #print(f"Real Part Pred (Sequence {sequence_idx}):", real_part_pred)
            #print(f"Imag Part Pred (Sequence {sequence_idx}):", imag_part_pred)

            plt.plot(time_steps, real_part_test, 'b-', label='Original Real' if not legend_added else "", linewidth=2)
            plt.plot(time_steps, real_part_pred, 'r--', label='Reconstructed Real' if not legend_added else "", linewidth=2)
            plt.plot(time_steps, imag_part_test, 'g-', label='Original Imag' if not legend_added else "", linewidth=2)
            plt.plot(time_steps, imag_part_pred, 'y--', label='Reconstructed Imag' if not legend_added else "", linewidth=2)
            
            plt.fill_between(time_steps, -3, 3, where=True, color=(1, 0.5, 0.5), alpha=0.3, label='Intrusion Detected' if not legend_added else "")
            legend_added = True

    if legend_added:
        plt.legend()
    plt.title('7-IQ Data: Original vs Reconstructed with Intrusion Highlighted')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Value')
    plt.tight_layout()
    plt.show()
#---------------------------------8--------------------------------------------
def plot_with_intrusions8(all_X_chunk_test, all_X_chunk_pred, all_intrusion_flags, sequence_length, save_path):
    for idx in range(0, len(all_X_chunk_test), sequence_length):
        sequence_idx = idx // sequence_length
        if all_intrusion_flags[sequence_idx]:
            plt.figure(figsize=(14, 6))
            time_steps = np.arange(idx * sequence_length, (idx + 1) * sequence_length)

            real_part_test = all_X_chunk_test[idx, :, 0].reshape(-1)
            imag_part_test = all_X_chunk_test[idx, :, 1].reshape(-1)
            real_part_pred = all_X_chunk_pred[idx, :, 0].reshape(-1)
            imag_part_pred = all_X_chunk_pred[idx, :, 1].reshape(-1)

            # Printing the sliced data
            #print(f"Real Part Test (Sequence {sequence_idx}):", real_part_test)
            #print(f"Imag Part Test (Sequence {sequence_idx}):", imag_part_test)
            #print(f"Real Part Pred (Sequence {sequence_idx}):", real_part_pred)
            #print(f"Imag Part Pred (Sequence {sequence_idx}):", imag_part_pred)

            plt.plot(time_steps, real_part_test, 'b-', label='Original Real', linewidth=2)
            plt.plot(time_steps, real_part_pred, 'r--', label='Reconstructed Real', linewidth=2)
            plt.plot(time_steps, imag_part_test, 'g-', label='Original Imag', linewidth=2)
            plt.plot(time_steps, imag_part_pred, 'y--', label='Reconstructed Imag', linewidth=2)
            
            plt.fill_between(time_steps, -3, 3, where=True, color=(1, 0.5, 0.5), alpha=0.3, label='Intrusion Detected')
            plt.legend(loc='lower right')
            plt.title(f'8-IQ Data: Original vs Reconstructed with Intrusion Highlighted (Sequence {sequence_idx})')
            plt.xlabel('Sample Index')
            plt.ylabel('Normalized Value')
            plt.tight_layout()

            filename = os.path.join(save_path, f'intrusion_sequence_{sequence_idx}.png')
            plt.savefig(filename)
            plt.close()

#in RNN: we should determine the number of consecutive samples grouped together as a single input 
#sequence for the RNN, so the model will take the first N samples as input 
#and try to reconstruct them.
#-------------------------------SelfAttention-RNN---------------------------------------
model = Sequential()
# Input LSTM layer
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 2), return_sequences=True))
# Self-attention layer
model.add(MultiHeadAttention(head_size=64, num_heads=4, output_size=50))
# Rest of the model remains the same
model.add(LSTM(25, activation='relu', return_sequences=False))
model.add(RepeatVector(sequence_length))
model.add(LSTM(25, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(2)))

model.summary()
model.compile(optimizer='adam', loss='mse')
# first I need to train pure data batch by batch
batch_size = 100
num_pure_samples = count_lines('/home/mreza/5G accelerator/models/5G_DL_IQ_no_jamming_0924.dat')
#print('num_pure_samples:', num_pure_samples)

max_train_samples = 2000000  # I limit the train or can put None for whole data
train_steps = (min(num_pure_samples, max_train_samples) if 
               max_train_samples else num_pure_samples) // (batch_size * sequence_length)

train_gen_instance = DataGenerator('/home/mreza/5G accelerator/models/5G_DL_IQ_no_jamming_0924.dat', 
                                   batch_size=batch_size, sequence_length=sequence_length, 
                                   max_samples=max_train_samples, for_training=True)

# Modify training loop
num_epochs = 7  # You can adjust the number of epochs as needed
steps_per_epoch = train_steps  # Assuming one epoch processes all the data
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_gen_instance.reset()  # Reset the generator at the beginning of each epoch
    for step in range(steps_per_epoch):
        try:
            X_chunk, Y_chunk = next(train_gen_instance)
        except StopIteration:
            train_gen_instance.reset()  # Reset the generator when it runs out of data
            X_chunk, Y_chunk = next(train_gen_instance)

        model.train_on_batch(X_chunk, Y_chunk)
        print(f"Step {step + 1}/{steps_per_epoch}", end='\r')
    print()
combined_gen_instance = DataGenerator('/home/mreza/5G accelerator/models/5G_DL_IQ_with_periodic_jamming_0928_02.dat', 
                                      batch_size=batch_size, sequence_length=sequence_length, 
                                      for_training=False)


num_samples = count_lines('/home/mreza/5G accelerator/models/5G_DL_IQ_with_periodic_jamming_0928_02.dat')
print('num_samples = count_lines:', num_samples)
max_predictions = num_samples // (batch_size * sequence_length)
print('max_predictions=num_samples//(batch_size*sequence_length):', max_predictions)

num_predictions = 2000  # or any other large number
num_predictions = min(num_predictions, max_predictions)

print(f"Maximum number of predictions available: {max_predictions}")
print(f"Number of predictions to be performed: {num_predictions}")

reconstruction_errors = []
all_X_chunk_test = []
all_X_chunk_pred = []
all_intrusion_flags = []
try:
    for _ in range(num_predictions):
        print('prediction number:', _)
        X_chunk_test = next(combined_gen_instance)
        X_chunk_pred = model.predict(X_chunk_test)
        chunk_errors = np.mean(np.square(X_chunk_test - X_chunk_pred), axis=1)
        reconstruction_errors.extend(chunk_errors)        
        all_X_chunk_test.append(X_chunk_test)
        all_X_chunk_pred.append(X_chunk_pred)
except StopIteration:
    print("All samples processed.")
    
    
reconstruction_error = np.array(reconstruction_errors)
#---------------------------------------111-----------------------------------
#reconstruction_error is already flat, size (num_predictions * batch_size * sequence_length * 2)
# Calculate the max error for each sequence across all time steps, considering real and imaginary parts separately
max_error_per_sequence = reconstruction_error.reshape(-1, 2).max(axis=1)  # Shape (num_predictions * batch_size * sequence_length,)
# Now, get the error per sequence (not per step) by taking the mean of max errors in chunks of sequence_length
error_per_sequence = max_error_per_sequence.reshape(-1, sequence_length).mean(axis=1)  # Shape (num_predictions * batch_size,)
# Determine the threshold for intrusion
threshold1 = np.percentile(error_per_sequence, 99.6)
print('threshold1:', threshold1)
threshold2 = np.percentile(reconstruction_error, 99.6)
print('threshold percentile:', threshold2)
threshold3 =totalMagnitude /totalnumberofsamples
# print("Total number of samples", totalnumberofsamples)
print('threshold magnitude:', threshold3)
# Detect sequences where the error exceeds the threshold
is_intrusion_detected = error_per_sequence > threshold1  # Boolean array for sequences, shape (num_predictions * batch_size,)
num_total_sequences = num_predictions * batch_size - num_predictions
print('len(is_intrusion_detected):', len(is_intrusion_detected))
print('num_total_sequences:', num_total_sequences)

# Here the code is for plotting only the last batch, we take the last 'batch_size' detected intrusions
if len(is_intrusion_detected) == num_total_sequences:
    print("Correct number of sequences for intrusion detection.")
    last_batch_intrusions = is_intrusion_detected[-batch_size:]  # Get the last batch_size elements
    print('len(last_batch_intrusions):', len(last_batch_intrusions))
    plot_with_intrusions4(X_chunk_test, X_chunk_pred, last_batch_intrusions, sequence_length)
else:
    print("Incorrect number of sequences for intrusion detection.")
#---------------------------------------finish 111-----------------------------------
#-------------------------------------new block------------------------------
flat_error_per_sequence = error_per_sequence.flatten()
# Determine if intrusion detected for each sequence
for error in flat_error_per_sequence:
    all_intrusion_flags.append(error > threshold1)    
#is_intrusion_detected2 = is_intrusion_detected.reshape((num_predictions-1), batch_size)
# Flatten the accumulated test and prediction data
all_X_chunk_test = np.concatenate(all_X_chunk_test, axis=0)
all_X_chunk_pred = np.concatenate(all_X_chunk_pred, axis=0)
#all_intrusion_flags = is_intrusion_detected2.flatten()
print('len(all_X_chunk_test):',len(all_X_chunk_test))
print('len(all_X_chunk_pred):',len(all_X_chunk_pred))
print('len(all_intrusion_flags):',len(all_intrusion_flags))
# Plot all batches where intrusion is detected
plot_with_intrusions6(all_X_chunk_test, all_X_chunk_pred, all_intrusion_flags, sequence_length)

plot_with_intrusions7(all_X_chunk_test, all_X_chunk_pred, all_intrusion_flags, sequence_length)

#-------------------------------finish new block-------------------------------------------------
#----------------------------------save plots separately----------------------------
save_path = '/home/mreza/5G accelerator/models/results/intrusion_detected'

# Plot all batches where intrusion is detected and save them as separate PNG files
plot_with_intrusions8(all_X_chunk_test, all_X_chunk_pred, all_intrusion_flags, sequence_length, save_path)
#---------------------------------------------------------------------------
jamming_detected = reconstruction_error > threshold2
#print('jamming_detected:', jamming_detected)

train_gen_instance.close()
combined_gen_instance.close()

### visualization #######
#Table to get insight
flattened_jamming_detected = jamming_detected.flatten()

real_part_detected = jamming_detected[:, 0]
imag_part_detected = jamming_detected[:, 1]

real_true_count = np.sum(real_part_detected)
real_false_count = len(real_part_detected) - real_true_count

imag_true_count = np.sum(imag_part_detected)
imag_false_count = len(imag_part_detected) - imag_true_count

# Overall
overall_true_count = np.sum(flattened_jamming_detected)
overall_false_count = len(flattened_jamming_detected) - overall_true_count

# Table-DataFrame
df = pd.DataFrame({
    'Part': ['Real', 'Imaginary', 'Overall'],
    'True Count': [real_true_count, imag_true_count, overall_true_count],
    'False Count': [real_false_count, imag_false_count, overall_false_count]
})

print(df)

num_jamming_detected = np.sum(jamming_detected)
print(f"Number of jamming sequences detected: {num_jamming_detected} out of {len(flattened_jamming_detected)} sequences")

# reconstruction error
plt.figure(figsize=(14, 6))
plt.plot(reconstruction_error, label='Reconstruction Error')
plt.axhline(y=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error with Threshold')
plt.xlabel('Sequence Number')
plt.ylabel('Reconstruction Error')
plt.legend()
# plt.savefig('1-Reconstruction Error with Threshold.png')
# plt.close()
plt.show()

# reconstruction error
reconstruction_error_real = reconstruction_error[:, 0]
reconstruction_error_imag = reconstruction_error[:, 1]

# Plot for Real Part
plt.figure(figsize=(14, 6))
plt.plot(reconstruction_error_real, label='Reconstruction Error - Real Part', color='blue')
plt.axhline(y=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error for Real Part with Threshold')
plt.xlabel('Sequence Number')
plt.ylabel('Reconstruction Error')
plt.legend()
# plt.savefig('2-Reconstruction Error for Real Part with Threshold.png')
# plt.close()
plt.show()

# Plot for Imaginary Part
plt.figure(figsize=(14, 6))
plt.plot(reconstruction_error_imag, label='Reconstruction Error - Imaginary Part', color='orange')
plt.axhline(y=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error for Imaginary Part with Threshold')
plt.xlabel('Sequence Number')
plt.ylabel('Reconstruction Error')
plt.legend()
# plt.savefig('3-Reconstruction Error for Imaginary Part with Threshold.png')
# plt.close()
plt.show()


#Histogram of Reconstruction Errors:
plt.figure(figsize=(14, 6))
plt.hist(reconstruction_error, bins=50, alpha=0.75)
plt.axvline(x=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Histogram of Reconstruction Errors')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend()
# plt.savefig('4-Histogram of Reconstruction Errors.png')
# plt.close()
plt.show()


#Time Series Plot of IQ Samples:
sample_index = np.random.choice(len(X_chunk_test))
original_sample = X_chunk_test[sample_index]
reconstructed_sample = X_chunk_pred[sample_index]

plt.figure(figsize=(14, 6))
plt.plot(original_sample[:, 0], 'b-', label='Original Real Part')
plt.plot(reconstructed_sample[:, 0], 'r--', label='Reconstructed Real Part')
plt.plot(original_sample[:, 1], 'g-', label='Original Imaginary Part')
plt.plot(reconstructed_sample[:, 1], 'y--', label='Reconstructed Imaginary Part')
plt.title('Original vs Reconstructed IQ Data')
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.legend()
# plt.savefig('5-Original vs Reconstructed IQ Data.png')
# plt.close()
plt.show()



#Scatter Plot of Reconstruction Errors vs. Real and Imaginary Parts:
avg_real = np.mean(X_chunk_test, axis=1)[:, 0]
avg_imag = np.mean(X_chunk_test, axis=1)[:, 1]

last_errors = np.mean(reconstruction_errors[-len(X_chunk_test):], axis=1)

print("Shape of avg_real:", avg_real.shape)
print("Shape of avg_imag:", avg_imag.shape)
print("Shape of last_errors:", len(last_errors))


plt.figure(figsize=(14, 6))
plt.scatter(avg_real, last_errors, label='Real Part', alpha=0.5)
plt.axhline(y=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error vs. Average Real Part')
plt.xlabel('Average Amplitude')
plt.ylabel('Reconstruction Error')
plt.legend()
# plt.savefig('6-Reconstruction Error vs. Average Real Part.png')
# plt.close()
plt.show()

plt.figure(figsize=(14, 6))
plt.scatter(avg_imag, last_errors, label='Imaginary Part', alpha=0.5)
plt.axhline(y=threshold2, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error vs. Average Imaginary Part')
plt.xlabel('Average Amplitude')
plt.ylabel('Reconstruction Error')
plt.legend()
# plt.savefig('7-Reconstruction Error vs. Average Imaginary Part.png')
# plt.close()
plt.show()

# # Define the number of sequences to plot together
n = 5  # Change this to desired number of sequences
sample_length = sequence_length * n

# Select a random starting sequence for plotting
sequence_index = np.random.choice(len(X_chunk_test) - n + 1)

# Extract and concatenate the original and reconstructed samples
original_sample = np.concatenate(X_chunk_test[sequence_index:sequence_index + n])
reconstructed_sample = np.concatenate(X_chunk_pred[sequence_index:sequence_index + n])

# Plot concatenated sequences
plt.figure(figsize=(14, 6))
plt.plot(original_sample[:, 0], 'b-', label='Original Real Part')
plt.plot(reconstructed_sample[:, 0], 'r--', label='Reconstructed Real Part')
plt.plot(original_sample[:, 1], 'g-', label='Original Imaginary Part')
plt.plot(reconstructed_sample[:, 1], 'y--', label='Reconstructed Imaginary Part')
plt.title(f'Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}')
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.legend()
# plt.savefig('9-Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}.png')
# plt.close()
plt.show()

# Repeat for n = 9
n = 9  # Change this to desired number of sequences
sequence_index = np.random.choice(len(X_chunk_test) - n + 1)
original_sample = np.concatenate(X_chunk_test[sequence_index:sequence_index + n])
reconstructed_sample = np.concatenate(X_chunk_pred[sequence_index:sequence_index + n])

plt.figure(figsize=(14, 6))
plt.plot(original_sample[:, 0], 'b-', label='Original Real Part')
plt.plot(reconstructed_sample[:, 0], 'r--', label='Reconstructed Real Part')
plt.plot(original_sample[:, 1], 'g-', label='Original Imaginary Part')
plt.plot(reconstructed_sample[:, 1], 'y--', label='Reconstructed Imaginary Part')
plt.title(f'Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}')
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.legend()
# plt.savefig('11-Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}.png')
# plt.close()
plt.show()



# In[ ]:


# Repeat for n = 9
n = 9  # Change this to desired number of sequences
sequence_index = np.random.choice(len(X_chunk_test) - n + 1)
original_sample = np.concatenate(X_chunk_test[sequence_index:sequence_index + n])
reconstructed_sample = np.concatenate(X_chunk_pred[sequence_index:sequence_index + n])

plt.figure(figsize=(14, 6))
plt.plot(original_sample[:, 0], 'b-', label='Original Real Part')
plt.plot(reconstructed_sample[:, 0], 'r--', label='Reconstructed Real Part')
plt.plot(original_sample[:, 1], 'g-', label='Original Imaginary Part')
plt.plot(reconstructed_sample[:, 1], 'y--', label='Reconstructed Imaginary Part')
plt.title(f'Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}')
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.legend()
# plt.savefig('11-Original vs Reconstructed IQ Data for {n} Sequences of Length {sequence_length}.png')
# plt.close()
plt.show()


# In[ ]:




