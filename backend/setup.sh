pip install -r requirements.txt

conda config --add channels conda-forge
conda install montreal-forced-aligner
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa