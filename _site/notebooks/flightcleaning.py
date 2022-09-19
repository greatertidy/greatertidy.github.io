import pandas as pd
import numpy as np
import csv

#####    Make Flight Delays CSV
# col 15 = cancellation indicator
cols_to_keep = [0, 1, 2, 3, 4, 7, 14, 15, 18, 19, 21]
# only use data from 2017-2018 due to tableau public restriction
csv_file_list = [f"M:/data/flights/{i}.csv" for i in range(2009, 2019)]
output_file = r"M:/data/flights/flight_delays.csv"
col_names = ("row_id,date,airline,flight_number,origin,destination,dep_delay,"+
                     "arr_delay,crs_elapsed_time,actual_elapsed_time\n")
csv_flights = open(output_file, "a")
csv_flights.write(col_names)
csv_flights.close()


csize = 50000 # process in chunks
# row id = row index of current csv + number of rows in previous csv files
num_rows = 0 # store number of rows to properly label row ids

for csv_file_name in csv_file_list:
    chunk_container = pd.read_csv(csv_file_name, 
                                  chunksize=csize,
                                  skiprows=[0], 
                                  usecols=cols_to_keep,
                                  header=None) # don't use header
    for chunk in chunk_container:
        # only include non cancelled flights
        chunk = chunk[chunk.iloc[:, -4]==0.]
        # remove nan entries
        chunk = chunk[~chunk.isna().any(axis=1)]
        # remove canncellation indicator
        chunk = chunk.drop(chunk.columns[-4], axis=1)
        # pandas interprets as float if there are nulls
        # convert to int
        chunk.iloc[:, -5:] = chunk.iloc[:, -5:].astype('int')
        # process row id
        n_chunk = len(chunk)
        chunk.index = np.arange(num_rows, num_rows+n_chunk, 
                                dtype=np.int32)
        num_rows += n_chunk
        # append to output, use header if first one
        chunk.to_csv(output_file, mode="a", index=True,
                   header=False)


##### Make Airlines csv
airlines = pd.read_csv(r"M:/data/flights/airlines.csv")
# Original airlines csv has quotes on airline code
# remove them to match flight delay csv
# Need quotes around Description in case of commas in desc
descs = airlines.Description.values
for i in range(len(descs)):
    descs[i] = '"'+descs[i]+'"'

airlines.to_csv(r"M:/data/flights/airlines_min_quote.csv",
                quoting=csv.QUOTE_NONE, escapechar=",", index=False)
