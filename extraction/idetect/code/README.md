## Information Extraction

#### Update Spacy Version (Old version is not available in Spacy library)

* Change ```spacy.load('en')``` to ```spacy.load("en_core_web_sm")```

* Change ```from textacy.spacy_utils import get_main_verbs_of_sent, get_objects_of_verb, get_subjects_of_verb``` 
 to ```from textacy.spacier.utils import get_main_verbs_of_sent, get_objects_of_verb, get_subjects_of_verb```


#### Class Interpreter: 


* ```basic_number```: Add two relevant number strings
 
* ```check_if_collection_contains_token```: Loosen the condition of matching, the sequence doesn't matter. 

* ```get_quantity_from_phrase```: Add two corner cases: 1. if the quantity words are consecutive. 2. if the token is the last token in a sentence and is quantity word

* ```get_quantity_test```: Check a single sentence whether there exist a quantity word or phrase
* 
* ```get_quantity2```: 1. Align with the update function ```get_quantity_from_phrase``` 2. Allow storing multiply quantity words/phrase within a sentence, where the original 
* function ```get_quantity``` can only store one quantity word/phrase given one sentence. 

* ```extract_all_dates2```: Loosen the condition of date by removing the ```date_likelihood``` check to catch the missing date. 


#### Class Information_extraction: 

The method ensembled the location, date and quantity extration functions

* ```get_location_span```: extract the location given an article (string) as input

* ```get_date_span```: extract the date given an article (string) as input.

* ```get_quantity_span```: extract the quantity words or phrases given an article (string) as input

