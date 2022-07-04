# Master-Popularity-Bias-In-JobRecs
This repo contains the code for the master thesis: "Popularity Bias in Job Recommender Systems"

The computations for the experiments were performed on the finn.no data cluster.
It is possible to run the code on a graphics processing unit (GPU), but due to technical limitations, GPUs were not used in the experiment.
The source code is available on GitHub at
\newline {\url{https://github.com/Pernilleww/Master-Population-Bias-In-JobRecs/url}}.
The following list describes where and how to execute the code:


* Due to privacy concerns, the data from Finn could not be made public before the deadline for this thesis.If the data is required, however, please contact perniww@gmail.com.
* The xQuAD code is located within the 'xQuAD' folder.  The $ Model.py$ and $xQuAD.py$ is taken from \cite{notebook2} and the belonging github can be found: \url{https://github.com/biasinrecsys/wsdm2021}
* To run the pyspark ALS experiment run the file $ALS\_pipeline.ipynb$. The pyspark ALS code is in $pyspark\_ALS.py$ and the popularity metrics for pyspark ALS is in $ metrics.py$ 
* To run the implicit experiment, run the file $implicit\_ALS.ipynb$
* To run the Lenskit experiment, run the file $lenskitALS.ipynb$
* The data.py file contains all the functions for preprocessing and feature engineering of the data.
