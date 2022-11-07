# Data

Description of data used for this project:
- orig_elements.csv
  - first four columns contain the town's code in the Survey of English Dialects (SED), its name, latitude, and longitude
  - columns five through 13 contain miscellaneous data
  - columns 14 onwards contain linguistic elements and their code (0 or 1)
- cmbnd_feautes_only.csv
  - each row contains a town's feaures as indicated in the header (INV means inverse; e.g. h-dropping = 1 means no h-dropping)
- SED_PoBI_locations.csv
  - each row contains the town's associated PoBI sample-location code
- standardized_rates_list.pickle
  - a python pickled file. It is used to create final results and plots, and it contains a list with 3 elements:
    - the mean linguistic rate of change for each linguistic location
    - for the same locations, a list of the genetic rates of change per cluster pair
    - the names of the cluster pairs used in the previous rates
