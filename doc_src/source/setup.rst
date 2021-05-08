Setup
==================================

The first step to using ehr_ml is performing an ETL between your data and the ehr_ml native format.
ehr_ml is designed to primarily work with OMOP data, but if necessary it's possible to write a custom ETL.

In this tutorial we will go over how to perform an ETL with an OMOP formatted dataset.
As a shortcut, you can also simply use the preprocessed extract "synpuf_extract.zip" within the example folder.

*********************************************
Installation
*********************************************

ehr_ml can be installed by cloning the repository and using poetry install. As of time of writing, we have only confirmed installation on Linux (Ubuntu, Debian, CentOS).

Note that there are several main build depencies for ehr_ml:
   1. Python 3.8
   2. Poetry (https://github.com/python-poetry/poetry)
   3. Bazel 3.x (https://bazel.build/)
   4. A C++ compiler that supports C++14
   5. CMake 3 or greater
   6. PyTorch (https://pytorch.org/get-started/locally/)
   7. The `embedding_dot` package (https://github.com/Lalaland/embedding_dot)

If you are using Anaconda, the following commands should install those dependencies:

.. code-block:: console

   conda create -n env_name python=3.8
   conda activate env_name
   conda install cmake poetry bazel=3.1 gxx_linux-64 -c conda-forge
   conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
   git clone https://github.com/Lalaland/embedding_dot.git
   cd embedding_dot
   pip install -e .
   cd ..
   git clone https://github.com/som-shahlab/ehr_ml
   cd ehr_ml
   poetry install

Note that your installation command for PyTorch may differ depending on your version of CUDA; please check https://pytorch.org/get-started/locally/ for the appropriate installation command. Additionally, :code:`embedding_dot` (and CLMBR as a whole) will still work even if you do not have CUDA installed, so long as you install the CPU-only version of PyTorch first.

*********************************************
Existing Extracts
*********************************************
The repository contains a pre-extracted version of the OMOP SynPUF dataset that you can work with out of the box under :code:`example/synpuf_extract.zip`. The following should allow you to get started:

.. code-block:: console

   cd ehr_ml/example
   unzip synpuf_extract.zip

You should see at least three files: :code:`extract.db`, :code:`ontology.db` and :code:`index.db`. You can then follow the instructions in the tutorial to get started training models.

*********************************************
Creating New Extracts
*********************************************

There are three important necessary datasets for performing ehr_ml extractions:

1. You must first have a clinical dataset in OMOP form.

   - One example accessible synthetic clinical dataset is the OMOP SynPUF dataset available at http://www.ltscomputingllc.com/downloads/. Once unzipped, the location of this directory is referred to as `SYNPUF_LOCATION` below.

2. The clinical dataset must have an attached OMOP vocabulary.

     - Normally this comes with the dataset itself, but in the case of SynPUF this must be downloaded seperately from https://athena.ohdsi.org/. You will need to create an account, which may take a few hours to get approval. The vocabulary can be downloaded from the "Download" tab in the top right of the webpage.
     - **IMPORTANT:** Make sure to perform the CPT4 postprocessing step after downloading (on Linux, this involves running `cpt.sh` with your Athena API key, more detailed instructions in the `readme.txt` included in the SynPUF download).
     - After performing the CPT4 postprocessing, move the concept files into the same directory as the SynPUF dataset. In the instructions below, the directory of this vocabulary is referred to as `VOCAB_LOCATION`.

3. You must have a recent copy of UMLS. This can be obtained from https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html. In the instructions below, the unzipped directory is referred to as `UMLS_FOLDER_LOCATION`.

4. You must have a copy of the latest General Equivalence Mappings for both ICD9 diagnoses and procedures. These can be downloaded from https://www.cms.gov/Medicare/Coding/ICD10/2018-ICD-10-CM-and-GEMs and https://www.cms.gov/Medicare/Coding/ICD10/2018-ICD-10-PCS-and-GEMs. Place the contents in in a single directory, referred to as `GEM_FOLDER_LOCATION` below.

*********************************************
Fixing SynPUF Data
*********************************************

The SynPUF dataset isn't correctly formatted for direct use with ehr_ml tools. In particular, we require that the csv files have headers and that everything is compressed with gzip.

In order to deal with these issues, ehr_ml contains a tool for fixing the SynPUF dataset.

.. code-block:: console

   # Need to add vocabulary to synpuf first
   cp VOCAB_LOCATION/* SYNPUF_LOCATION

   # Clean up the synpuf extract
   ehr_ml_clean_synpuf SYNPUF_LOCATION FIXED_SYNPUF_LOCATION

*********************************************
Running The Extraction
*********************************************

The extractor can now be run on the properly formatted SynPUF dataset. If you are using another OMOP dataset, replace `FIXED_SYNPUF_LOCATION` with the path to that dataset. You may need to change the delimiter to ',' depending on the format of your concept files.

.. code-block:: console

   ehr_ml_extract_omop FIXED_SYNPUF_LOCATION UMLS_FOLDER_LOCATION GEM_FOLDER_LOCATION TARGET_EXTRACT_FOLDER_LOCATION --delimiter $'\t' --ignore_quotes

*********************************************
Verifying The Extraction
*********************************************

The extraction results can be manually inspected using the inspect_timelines tool.

.. code-block:: console

   inspect_timelines TARGET_EXTRACT_FOLDER_LOCATION 0

You should see a simple patient timeline as a result.
