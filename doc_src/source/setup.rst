Setup
==================================

The first step to using ehr_ml is performing an ETL between your data and the ehr_ml native format.
ehr_ml is designed to primarily work with OMOP data, but if necessary it's possible to write a custom ETL.

In this tutorial we will go over how to perform an ETL with an OMOP formatted dataset.
As a shortcut, you can also simply use the preprocessed extract "synpuf_extract.zip" within the example folder.

*********************************************
Installation
*********************************************

ehr_ml can be installed by cloning the repository and using poetry install.

Note that there are several main build depencies for ehr_ml:
   1. Poetry (https://github.com/python-poetry/poetry)
   2. Bazel 3 or greater (https://bazel.build/)
   3. A C++ compiler that supports C++14
   4. CMake 3 or greater

If you are using conda, the following command should install those dependencies:

.. code-block:: console

   conda install cmake poetry bazel gxx_linux-64 -c conda-forge


.. code-block:: console

   git clone https://github.com/som-shahlab/ehr_ml
   cd ehr_ml
   poetry install

*********************************************
Downloading Data
*********************************************

There are three important necessary datasets for performing ehr_ml extractions:

1. You must first have a clinical dataset in OMOP form. One example accessible synthetic clinical dataset is the OMOP SynPUF dataset available at http://www.ltscomputingllc.com/downloads/.

2. The clinical dataset must have an attached OMOP vocabulary. Normally this comes with the dataset itself, but in the case of SynPUF this must be downloaded seperately from https://athena.ohdsi.org/. Make sure to perform the CPT4 postprocessing step after downloading.

3. You must have a recent copy of UMLS. This can be obtained from https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html.

4. You must have a copy of the latest General Equivalence Mappings for both ICD9 diagnoses and procedures. These can be downloaded from https://www.cms.gov/Medicare/Coding/ICD10/2018-ICD-10-CM-and-GEMs and https://www.cms.gov/Medicare/Coding/ICD10/2018-ICD-10-PCS-and-GEMs. (Merge the two folders.)

*********************************************
Fixing SynPUF Data
*********************************************

The SynPUF dataset isn't correctly formatted for direct use with ehr_ml tools. In particular, we require that the csv files have headers and that everything is compressed with gzip.

In order to deal with these issues, ehr_ml contains a tool for fixing the SynPUF dataset.

.. code-block:: console

   ehr_ml_clean_synpuf SYNPUF_LOCATION FIXED_SYNPUF_LOCATION

*********************************************
Running The Extraction
*********************************************

The extractor can now be run on the properly formatted SynPUF dataset.

.. code-block:: console

   ehr_ml_extract_omop FIXED_SYNPUF_LOCATION UMLS_FOLDER_LOCATION GEM_FOLDER_LOCATION TARGET_EXTRACT_FOLDER_LOCATION --delimiter "  " --ignore_quotes

*********************************************
Verifying The Extraction
*********************************************

The extraction results can be manually inspected using the inspect_timelines tool.

.. code-block:: console

   inspect_timelines TARGET_EXTRACT_FOLDER_LOCATION 0

You should see a simple patient timeline as a result.