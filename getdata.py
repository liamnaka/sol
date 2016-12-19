import csv
import os
import pandas

#Expects cepdb.csv file in data folder, with SMILES in column[1] and the PCE in column[4]
#SMILES should be labeled as 'structure' and PCE as 'pce'

#Extracts SMILES string and PCE data from CEPDB csv and returns csv and h5 file
def percent_to_num(string):
   numstring = ""
   try:
      numstring = str(float(string)/100.0)
   except ValueError:
      numstring = string

   return numstring

with open("data/cepdb.csv","rb") as source:
   rdr = csv.reader(source)
   with open("data/cepdbsmiles.csv","wb") as result:
      wtr = csv.writer(result)
      for r in rdr:
         wtr.writerow((r[1],percent_to_num(r[4]))) #in format(smiles,pce)


#CSV to h5
infile 'data/cepdbsmiles.csv'
outfile = 'cepdb.h5'#h5 for training
df = pandas.read_csv(infile)
df.to_hdf(outfile, 'table', format = 'table', data_columns = True)
