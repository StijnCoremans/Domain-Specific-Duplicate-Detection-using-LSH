# Domain Specific Duplicate Detection using LSH

The code in this folder can be used to obtain the results from the "Domain Specific Duplicate Detection using LSH" paper. 
It allows for running the MSMP model, which can be extended to the MSMP+ and MSMP+domain models with or without strict domain blocking.

## How to use the code

The data can be downloaded from: https://personal.eur.nl/frasincar/datasets/TVs-all-merged.zip.

"domain_info.py" is used to determine which features to use as domain features and provides a count a all keys
as well as an overview of which keys are present across multiple webshops.

The results can be obtained by running "domain_MSMP.py" for a specifc model setup. This code file uses the "encoder.py", "minhashing.py" and "lsh.py" files
to determine the candidate pairs for the MSM step.

    Line 212: specify here whether or not to use strict domain blocking
    Line 396-398: specify which LSH to use, the choice is between MSMP, MSMP+ and MSMP+domain. 

Finally, plots of the results can be created using "plots.R".
