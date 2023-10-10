#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
import matplotlib.pyplot as plt
import os

#To unback binary encoded file
import struct

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


def process_data(samples):
    real_parts = []
    imag_parts = []
    for sample in samples:
        try:
            cnum = complex(sample.replace('j', 'j'))
            real_parts.append(np.real(cnum))
            imag_parts.append(np.imag(cnum))
        except ValueError:
            #print(f"Malformed complex number string: {sample}")
            continue  # I'll pass error values, not important

    real_parts = (real_parts - np.mean(real_parts)) / np.std(real_parts)
    imag_parts = (imag_parts - np.mean(imag_parts)) / np.std(imag_parts)
    
    sequence_length = 10
    X = [list(zip(real_parts[i:i+sequence_length], imag_parts[i:i+sequence_length])) for i in range(len(real_parts) - sequence_length)]
    return np.array(X)


def data_generator(filepath, batch_size=1024, max_samples=None, for_training=True):
    chunksize = batch_size * sequence_length
    _, file_extension = os.path.splitext(filepath)

    total_samples_processed = 0

    # I read data in chunks both csv and dat
    if file_extension == '.csv':
        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            if max_samples and total_samples_processed >= max_samples:
                break
            samples = chunk['IQ Data'].values
            X_chunk = process_data(samples)
            total_samples_processed += len(samples)
            yield X_chunk, X_chunk

    
    elif file_extension == '.dat':
        #DAT file is encoded as
        #application/octet-stream; charset=binary

        #Hence we needed to decode the binary file using the struct module
        
        samples = []
        skip_zeros = True  # Use this flag to check if we should still skip zero lines
        # f was change to binary_file. f is reserved for 32 bit floating point in binary decoding
        with open(filepath, 'rb') as binary_file:  # reading in text mode, should handle decoding
            while True:
                #print('line:', line)
                    binary_data = binary_file.read(8)
                    #decoded_line = line.decode('utf-8').strip()
                    #decoded_line = line.strip()
                    if not binary_data:
                        break 
                    #decoded_line = line.strip()
                    decoded_data = struct.unpack('ff', binary_data)
                    if decoded_data[0] == 0 and decoded_data[1] == 0:
                        decoded_line = f"0j\n"
                    else :
                        if decoded_data[1] >= 0:
                             decoded_line = f"{decoded_data[0]}+{decoded_data[1]}j\n"
                        else:
                            decoded_line = f"{decoded_data[0]}{decoded_data[1]}j\n"
                
                    #print( decoded_line)
                    samples.append(decoded_line)
               
                
                    if max_samples and total_samples_processed >= max_samples:
                        break
                    samples.append(decoded_line)
                    total_samples_processed += 1
                    if len(samples) == chunksize:
                        X_chunk = process_data(samples)
                        #print("Samples:", samples)
                        #print("X_chunk:", X_chunk)
                        print("Shape of X_chunk:", X_chunk.shape)
                        if for_training:
                            yield X_chunk, X_chunk  # Yields input data and target data for training
                        else:
                            yield X_chunk
                        samples = []



#in RNN: we should determine the number of consecutive samples grouped together as a single input 
#sequence for the RNN, so the model will take the first N samples as input 
#and try to reconstruct them.

sequence_length = 10


model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 2), return_sequences=True))
model.add(LSTM(25, activation='relu', return_sequences=False))
model.add(RepeatVector(sequence_length))
model.add(LSTM(25, activation='relu', return_sequences=True))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(2)))

model.compile(optimizer='adam', loss='mse')

# first I need to train pure data batch by batch

batch_size = 1024
num_pure_samples = count_lines('/home/mreza/5G accelerator/models/5G_DL_IQ_no_jamming_0924.dat')
#Testing with csv file
#num_pure_samples = count_lines('/home/oai/Desktop/iq_samples/5G_DL_IQ_no_jamming_0924.dat')

print('num_pure_samples:', num_pure_samples)

max_train_samples = 100000  # I limit the train or can put None for whole data
train_steps = (min(num_pure_samples, max_train_samples) if 
               max_train_samples else num_pure_samples) // (batch_size * sequence_length)


train_gen = data_generator('/home/mreza/5G accelerator/models/5G_DL_IQ_no_jamming_0924.dat', batch_size, max_train_samples, for_training=True)
model.fit(train_gen, steps_per_epoch=train_steps, epochs=10, verbose=1)


# Now reconstructing error by trained model and infected data
#combined_gen = data_generator('/home/mreza/5G accelerator/models/5G_DL_IQ_with_periodic_jamming_0928_02.dat', batch_size)
combined_gen = data_generator('/home/mreza/5G accelerator/models/5G_DL_IQ_with_periodic_jamming_0928_02.dat', batch_size, for_training=False)

reconstruction_errors = []
for X_chunk_test in combined_gen:
    X_chunk_pred = model.predict(X_chunk_test)
    chunk_errors = np.mean(np.square(X_chunk_test - X_chunk_pred), axis=1)
    reconstruction_errors.extend(chunk_errors)

reconstruction_error = np.array(reconstruction_errors)

# set threshold
threshold = np.percentile(reconstruction_error, 95)

jamming_detected = reconstruction_error > threshold




num_jamming_detected = np.sum(jamming_detected)
print(f"Number of jamming sequences detected: {num_jamming_detected} out of {len(X_test)} sequences")

# reconstruction error
plt.figure(figsize=(14, 6))
plt.plot(reconstruction_error, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.title('Reconstruction Error with Threshold')
plt.xlabel('Sequence Number')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.show()

