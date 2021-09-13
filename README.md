# IDMC Internal Displacement

Contains all the useful resources created or used for the project on Internal Displacement in collaboration with IDMC (https://www.internal-displacement.org/).

## MAIN CONTENTS

### Notebooks
All the notebooks used for:
- analyzing the annotated data from Kili and assess annotation consensus (`consensus/*.ipynb`)
- train and evaluate machine learning models for classification tasks (`Classification.ipynb`)

### Datasets
Including:
- urls from articles provided by IDMC or scraped from GDELT urls
- processed output from Kili projects 

Only content for articles that either changed drastically or cannot be scraped are available. Content for other artilces need to be scraped using the code in `Data-Preparation.ipynb'.

### Tasks
Including:
- [Information Exaction](https://github.com/ISIFoundation/Internal-Displacement-Monitoring/tree/master/extraction/idetect/code/Information%20Extraction): extract location, date, quantity information which relavant to displacement
- [Cause Classification](https://github.com/ISIFoundation/Internal-Displacement-Monitoring/tree/master/extraction/idetect/code/Cause%20Classification): train machine learning models to predict the casue of displacement
- [Type Classification](https://github.com/ISIFoundation/Internal-Displacement-Monitoring/tree/master/extraction/idetect/code/Type%20Classification): train machine learning models to predict whether an article is relevant to displacement




