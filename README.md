This Python script is a bibliometric analysis tool designed to process, filter, and visualize scientific article data. To operate it, you must configure the file paths and search parameters.

1. File Path Configuration (Input & Output)
First, specify the locations for your data sources and the directory for the generated results.

SCOPUS_PATH: Provide the full file path for your data export from Scopus. This file must have a .csv extension.

Example: r"C:\Research\data\scopus_data.csv"

WOS_PATH: Set the path for your Web of Science (WoS) data, which must be an Excel file with an .xlsx extension.

Example: r"C:\Research\data\wos_data.xlsx"

OUTPUT_DIR: Define the directory where all script outputs, such as plots, Word reports, and the BibTeX file, will be saved. The script will create this folder if it does not already exist.

Example: r"C:\Research\Analysis_Results"

2. Search Term and Filter Specification
The script employs a two-stage filtering process to identify the most relevant publications.

BASE_TERMS: This list should contain the most specific and critical keywords for your research. The script uses these terms to compute a relevance score (Score) for each article, prioritizing those that feature these terms in their titles or keywords. It also programmatically finds synonyms to enhance coverage.

SEMANTIC_QUERY: This is a broader phrase that describes your central research topic. It is used for an initial "smart" semantic filter that identifies conceptually similar articles.

SEMANTIC_THRESHOLD: This parameter sets the similarity cutoff (from 0 to 1). A value of 0.24 retains only articles with a semantic similarity of 24% or greater to the SEMANTIC_QUERY.

TIME_FILTER: This defines the publication year range for the analysis. The example is set to analyze articles published between 1900 and 2025.
