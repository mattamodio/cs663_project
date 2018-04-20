# cs663_project

## Getting started

**Prerequisites**: Install anaconda and python3

1. Pull in dependencies and create the cs663-discogan environment:
   ```bash
   # Install dependencies and create environment
   $ conda env create -f environment.yml
   # Activate newly created environment
   $ source activate cs663_project
   # Install the project in development mode
   $ python setup.py develop
   ```

2. Download additional data. Right now, the only dataset integration is coil-100, but to download that, issue the following command
   ```bash
   $ download_data coil
   ```

3. Train DiscoGAN. For details on training, please see:
   ```bash
   $ train --help
   ```
