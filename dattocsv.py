import numpy as np
import pandas as pd

with open('5G_DL_IQ_with_periodic_jamming_0928_02.dat', 'rb') as file:
    iq_data = np.fromfile(file, dtype=np.complex64)  # complex IQ data

# adjust the chunk size
chunk_size = 1048576  # number of rows per file
num_chunks = len(iq_data) // chunk_size + 1  # number of chunks

# eparate excel files for each chunk
for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(iq_data))
    
    chunk = iq_data[start_idx:end_idx]
    df = pd.DataFrame({'IQ Data': chunk})
    
    output_file = f'output_chunk_{i+1}.xlsx'
    
    df.to_excel(output_file, index=False)
