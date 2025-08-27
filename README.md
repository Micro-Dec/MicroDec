# MicroDec
MicroDec currently supports **Java applications** due to its use of the Java Parser library.

## Usage Instructions

### Prerequisites
Before running the tool, ensure you have the following:
- All required libraries installed.
- The `symbolsolver-1.0.jar` file downloaded.

### Setting Up Your Application
1. Add your application to the `/app` folder.
2. Update the `data/final_results/all_clean_apps_scoh.csv` file by including the name of your application.  
   This allows the tool to process multiple applications.

### Running the Tool
- Execute the main script located at `/app/main_for_all.py` to begin processing.
- Once the processing is complete, run the Jupyter notebook `/select_best_results.ipynb` to identify the best results.

### Results
- The `data` folder contains:
  - All results reported in the paper.
  - Results of baseline comparisons.
  - OpenAI embedding data.
- The ground truth data and predicted results are located in the `MicroDec/results` folder.

## Contact
For any further assistance regarding the duplication of this repository, please contact me directly.

---

Thank you for using MicroDec!
