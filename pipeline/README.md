# Pipeline Usage

This pipeline is for going from PDF / CSV files without geolocation or classification to
a final .csv file that contains the classifications and geolocations.

## Conversion to CSV

If the file is initially a .pdf then it needs to be converted into a .csv file.
This is done using the Pdf2csv.py script. Usage is here:

```python pdf2csv.py {input_file}```

This will output to wherever the input was with the same file name but as a csv.

## Geolocation

This process will take a long time with a large dataset, each row is about 0.7-1 seconds.
Meaning a large batch will take a long time. 

input_file needs to be the path to the file (full path)

```python geocode.py {input_file}```

This will output to the same location but geocoded will be appended to the file name.

## Classification

This classifies the deaths in the file. This works with files where the cause of death
is in columns either "CauseA, CauseB" type or in "Primary Cause, Secondary Cause" type.

location_of_file is simply the path to the file that needs to be classified.

model_type is the model trained this is either BERT or bioclinicalbert, the best performing
is bioclinicalbert so should be called like so 

```python classify.py {location_of_file} {model_type}```

```python classify.py ./pipeline_steps/toclassify_geocoded.csv bioclinicalbert```

This should output out our final result.