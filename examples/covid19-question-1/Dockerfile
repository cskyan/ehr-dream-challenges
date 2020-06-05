FROM python:3.5

# Do not change
LABEL challenge="covid19"

# Description of the model (max 500 characters)
LABEL description="COVID diagnosis pre-selected-features-Clustering prediction"

# Comma-separated list of arbitrary feature names used by the model in desc
# order of importance.
LABEL ranked_features="cough,body temperature,hematocrit,diastolic blood pressure,systolic blood pressure"

# Comma-separated list of references (URL)
LABEL references=""

# Install dependencies
RUN pip install pandas \
  numpy \
  sklearn \
  datetime \
  Cython \
  "dask[complete]"

# Move to the app folder
RUN mkdir /app
WORKDIR /app

# Copy our python program for training and inference
COPY src/infer.py .

# Build Cython extension module
COPY src/extension.pyx .
COPY src/setup.py .
RUN python setup.py build_ext --inplace

# Copy Bash scripts expected by the IT infrastructure of the EHR DREAM Challenge
COPY infer.sh .

# Add executable permission to Bash scripts
RUN chmod +x infer.sh
