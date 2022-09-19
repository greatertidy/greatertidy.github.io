---
layout: default
usemathjax: false
---

## Introduction to Data
The [dataset]((https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018)) consists of rows each corresponding to a flight and each column
some feature of that flight (I chose to focus on a subset of these columns).
Let's take a look at an entry from this dataset and the columns of interest.
The first flight looks like this:


<!---
Table
-->
<table border="0" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Column Name</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>FL_DATE</th>
      <td>2009-01-01</td>
    </tr>
    <tr>
      <th>OP_CARRIER</th>
      <td>XE</td>
    </tr>
    <tr>
      <th>OP_CARRIER_FL_NUM</th>
      <td>1204</td>
    </tr>
    <tr>
      <th>ORIGIN</th>
      <td>DCA</td>
    </tr>
    <tr>
      <th>DEST</th>
      <td>EWR</td>
    </tr>
    <tr>
      <th>DEP_DELAY</th>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>ARR_DELAY</th>
      <td>4.0</td>
    </tr>
    <tr>
      <th>CRS_ELAPSED_TIME</th>
      <td>62.0</td>
    </tr>
    <tr>
      <th>ACTUAL_ELAPSED_TIME</th>
      <td>68.0</td>
    </tr>
    <tr>
      <th>DISTANCE</th>
      <td>199.0</td>
    </tr>
  </tbody>
</table>

which gives us some basic information on the flight time, the departure and arrival delay
(negative delay = early departure/arrival) as well as where the plane landed and departed from.

## Cleaning in python
Now that we have an idea of what the data looks like the next step would be to
get our data ready for analysis. First, when reading our data we should only take
the columns that we are interested in. In this case it looks like this

```python
cols_to_keep = [0, 1, 2, 3, 4, 7, 14, 15, 18, 19, 21]
```
with an extra column (column 15) which signifies if a flight has been cancelled
or not. Now since the data is quite large we will read the data in 'chunks'
in which pandas iterates over sequential partitions of the csv file. To be
conservative I chose to read the data in chunks of 50,000.

```python
csize = 50000
```

### Flight ID
Every plane that flies in the sky for an airline has what's called a
"flight designator". This is a code consisting of two characters, which specify
the airline, and a 1 to 4 digit number (this is different from a tail number
which specifies a specific plane in service). For example, DL401 is flight
operated by Delta Airlines which services transport from Anchorage to Minneapolis.
Airlines will commonly keep the same flight number for the same service but there
is no rule saying they have to or that they will (for example Malaysia Airlines
Flight 370 was changed to MAH318). The one rule an airline must follow is that
each flight in the air must have a unique flight ID. This means there can only be
one DL401 in the air at any time, however once this flight has finished it's course
there is nothing stopping Delta from immediately flying another plane with this
exact same flight ID. So to uniquely identify a flight we will need it's
flight number, departure date and departure time. Our dataset however, only
contains the flight number and departure date.

### SQL and Uniquely Identifying Flights
First, let's quickly talk about why we would need SQL and unique identifiers.
The [data source]((https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018))
stores the data in 10 csv files each corresponding to flights in a particular year
from 2009 to 2018. To analyze the data it will be useful to combine these csv files
into a singular source. This will make it easier for us to query data easier,
especially when we are interested in analyzing things like year to year trends
or seasonal trends common to each year. The only problem is that the data is quite
large (60M rows) which is not easy to analyze using python or tableau directly.
It would be easier if instead of importing the entire dataset into a python or
tableau workspace, we could summarize the data into parts we are interested in,
then send these much smaller easier to manage datasets into our visualization
tool.


For this reason I chose to store the data in a local SQL server from which I
could query and group data. The problem however, is that SQL requires a primary
key that identifies unique rows which is clear is not directly possible with our
dataset. For this reason I chose to assign to each row a unique id called the
row id according to the order it is listed in the dataset. The assumption I made was
* flights in the dataset were listed chronologically with respect to
the airline
    - for example, if there are two flights on January 1st 2009 from airline AA, the flight listed second did not depart before the first one

With this assumption, if we were to in the future collect data on datetimes of
flights with their flight ID, we could match a flight in our dataset with a flight
in the new dataset by
1. selecting all flights from the day with the same flight id from both datasets
2. arrange flights in our dataset according to their row id
3. arrange flights in the new dataset according to their datetime
4. match flights one-to-one according to their position in this arrangement

## Getting Data Ready for SQL
Before we can insert our data into SQL we will need to properly process it. I chose
to process the data in chunks of 50K, and within each chunk, I chose only the
non-cancelled flights with no missing values. So the first thing I did was create
the csv file with the proper column headers
```python
col_names = ("row_id,date,airline,flight_number,origin,destination,dep_delay,"+
                     "arr_delay,crs_elapsed_time,actual_elapsed_time\n")
csv_flights = open(output_file, "a")
csv_flights.write(col_names)
csv_flights.close()
```
Then for each csv file I processed data accordingly in chunks
```python
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
```

## Moving to SQL
Now we have our output csv file that we can bulk insert into a SQL table.
Along with our original dataset, I collected some data on [airlines](https://github.com/beanumber/airlines/blob/master/data-raw/airlines.csv)
and [airports](https://www.kaggle.com/datasets/usdot/flight-delays?select=airports.csv)
to better visualize our flights dataset. First I had to create the tables

```sql
/*** Airlines Table ***/
USE flight_delays
CREATE TABLE [dbo].Airlines(
	[code] [nvarchar](10) NOT NULL,
	[description] [nvarchar](150),
	PRIMARY KEY (code)
)

/*** Airports Table ***/
CREATE TABLE [dbo].Airports(
	[code][nchar](3) NOT NULL,
	[name][nvarchar](200) NOT NULL,
	[city][nvarchar](150),
	[state][nvarchar](15) NOT NULL,
	[country][nvarchar](3) NOT NULL,
	[latitude][decimal](8,6),
	[longtitude][decimal](9,6),
	PRIMARY KEY (code)
)
GO

/*** Flights Table ***/
CREATE TABLE [dbo].Flights(
	[row_id] [int] IDENTITY(0,1) NOT NULL,
	[date] [date] NOT NULL,
	[airline] [nvarchar](10) NOT NULL,
	[flight_number] [smallint] NOT NULL,
	[origin] [nchar](3) NOT NULL,
	[destination] [nchar](3) NOT NULL,
	[dep_delay] [smallint],
	[arr_delay] [smallint],
	[crs_elapsed_time] [smallint],
	[actual_elapsed_time] [smallint],
	[distance][smallint],
	PRIMARY KEY (row_id),
    FOREIGN KEY (airline) REFERENCES Airlines(code),
	FOREIGN KEY (origin) REFERENCES Airports(code),
	FOREIGN KEY (destination) REFERENCES Airports(code)
)
GO
```

Then, I inserted data from the csv's using bulk insert.

```sql
USE flight_delays
/*** Import from CSV ***/
BULK INSERT Airlines
FROM 'M:\data\flights\airlines_min_quote.csv'
WITH
(
    FIRSTROW = 2, -- ignore header
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    TABLOCK
)
GO


/*** Import from CSV ***/
BULK INSERT Airports
FROM 'M:\data\flights\airports.csv'
WITH
(
    FIRSTROW = 2, -- ignore header
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '0x0a',
    TABLOCK
)
GO


/*** Import from CSV ***/
BULK INSERT Flights
FROM 'M:\data\flights\flight_delays.csv'
WITH
(
    FIRSTROW = 2, -- ignore header
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    TABLOCK
)
GO
```

## Querying Data and Tableau
Finally we summarize the data using SQL queries to make the data more manageable
for Tableau. I chose to do this by grouping data by month according to three
factors, the airline, departure airport and the arrival airport.

```sql
USE flight_delays

/*** Monthly Airline Delays ***/
SELECT DATEPART(Year, date) AS year,
	DATEPART(Month, date) AS month,
	airline,
	Airlines.description AS name,
	COUNT(airline) as num_flights,
	AVG(dep_delay) as avg_dep_delay,
	AVG(arr_delay) AS avg_arr_delay,
	AVG(distance) as avg_distance,
	STDEV(distance) as stdev_distance
FROM Flights
LEFT JOIN Airlines
ON Flights.airline=Airlines.code
GROUP BY DATEPART(Year, date), DATEPART(Month, date), airline, Airlines.description
ORDER BY DATEPART(Year, date), DATEPART(Month, date), airline
GO

/*** Monthly Airport Origin ***/
SELECT DATEPART(Year, date) as year,
	DATEPART(Month, date) as month,
	origin as origin_airport,
	Airports.city as origin_city,
	Airports.state as origin_state,
	COUNT(origin) as num_flights,
	AVG(dep_delay) as avg_dep_delay,
	AVG(arr_delay) as avg_arr_delay,
	AVG(crs_elapsed_time) as avg_crs_elapsed_time,
	AVG(actual_elapsed_time) as avg_elapsed_time
FROM Flights
LEFT JOIN Airports
ON Flights.origin=Airports.code
GROUP BY DATEPART(Year, date), DATEPART(Month, date), origin, Airports.city, Airports.state
ORDER BY DATEPART(Year, date), DATEPART(Month, date), origin
GO

/*** Monthly Airport Destination ***/
SELECT DATEPART(Year, date) as year,
	DATEPART(Month, date) as month,
	destination as destination_airport,
	Airports.city as destination_city,
	Airports.state as destination_state,
	count(destination) as num_flights,
	AVG(dep_delay) as avg_dep_delay,
	AVG(arr_delay) as avg_arr_delay,
	AVG(crs_elapsed_time) as avg_crs_elapsed_time,
	AVG(actual_elapsed_time) as avg_elapsed_time
FROM Flights
LEFT JOIN Airports
ON Flights.destination=Airports.code
GROUP BY DATEPART(Year, date), DATEPART(Month, date), destination, destination, Airports.city, Airports.state
ORDER BY DATEPART(Year, date), DATEPART(Month, date), destination
GO
```
Now we can visualize data using Tableau (images open link to Tableau Public).
## Flight Overview
[![Flight Overview](/assets/img/flights/flights.JPG)](https://public.tableau.com/app/profile/mohammed.ali6348/viz/Flights_16625711978660/Overview)
## Departure Delays
[![Departure Delays](/assets/img/flights/depdelays.JPG)](https://public.tableau.com/app/profile/mohammed.ali6348/viz/FlightDelays_16626653065240/DepartureDelays)
