import re
import string
import pandas as pd
from datetime import datetime, timedelta
from collections import OrderedDict
import parsedatetime
from spacy.tokens import Token, Span
from spacy.symbols import ORTH, LEMMA, POS
from textacy import extract
from textacy.extract import matches
import spacy


# from textacy.extract import matches

# from textacy.extract import pos_regex_matches
from textacy.spacier.utils import get_main_verbs_of_sent, get_objects_of_verb, get_subjects_of_verb

from idetect.model import FactUnit, FactTerm, KeywordType, FactKeyword

def get_absolute_date(relative_date_string, publication_date=None):
    """
    Turn relative dates into absolute datetimes.
    Currently uses API of parsedatetime
    https://bear.im/code/parsedatetime/docs/index.html
    Parameters:
    -----------
    relative_date_string        the relative date in an article (e.g. 'Last week'): String
    publication_date            the publication_date of the article: datetime
    Returns:
    --------
    One of: 
        - a datetime that represents the absolute date of the relative date based on 
            the publication_date
        - None, if parse is not successful
    """

    cal = parsedatetime.Calendar()
    parsed_result = cal.nlp(relative_date_string, publication_date)
    if parsed_result is not None:
        # Parse is successful
        parsed_absolute_date = parsed_result[0][0]

        # Assumption: input date string is in the past
        # If parsed date is in the future (relative to publication_date),
        #   we roll it back to the past

        if publication_date and parsed_absolute_date > publication_date:
            # parsedatetime returns a date in the future
            # likely because year isn't specified or date_string is relative

            # Check a specific date is included
            contains_month = re.search(
                'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec', relative_date_string.lower())

            if contains_month:
                # TODO: Is it enough to just check for month names to determine if a
                #       date_string specifies a particular date?

                # If date is specified explicity, and year is not
                # roll back 1 year
                return datetime(parsed_absolute_date.year - 1,
                                parsed_absolute_date.month, parsed_absolute_date.day)
            else:
                # Use the relative datetime delta and roll back
                delta = parsed_absolute_date - publication_date
                num_weeks = int(delta.days / 7)
                and_num_days_after = 7 if delta.days % 7 == 0 else delta.days % 7
                return publication_date - timedelta(weeks=num_weeks) - \
                    timedelta(7 - and_num_days_after)
        else:
            # Return if date is in the past already or no publication_date is
            # provided
            return parsed_absolute_date
    else:
        # Parse unsucessful
        return None


def convert_quantity(value):
    '''Convert an extracted quantity to an integer.
    Solution forked from
    https://github.com/ghewgill/text2num/blob/master/text2num.py
    and enhanced with numerical and array input
    '''
    value = value.replace(",", "")
    try:
        return (int(value), None)
    except:
        return (None, value)


def convert_tokens_to_strings(value):
    if isinstance(value, Token):
        return value.text
    if isinstance(value, Span):
        return value.text
    else:
        return str(value)


def minimum_loc(spans):
    '''Find the first character location in text for each report
    '''
    locs = []
    for s in spans:
        if s['type'] != 'loc':
            locs.append(s['start'])
    return min(locs)




class Interpreter(object):

    def __init__(self, nlp, dic):
        self.nlp = nlp
        self.dic = dic
        # load_custom_tokenizer_cases(self.nlp)


    def check_if_collection_contains_token(self, token, collection):
        for c in collection:
            if token.i == c.i:
                return True
        return False

    def check_if_collection_contains_token2(self, token, collection):
        if token.text in collection.text:
            return True
        return False



    def get_descendents(self, sentence, root=None):
        """
        Retrieves all tokens that are descended from the specified root token.
        param: root: the root token
        param: sentence: a span from which to retrieve tokens.
        returns: a list of tokens
        """
        if not root:
            root = sentence.root
        ### For the new version change is_ancestor_of -> is_ancestor
        return [t for t in sentence if root.is_ancestor(t)]

    def check_if_entity_contains_token(self, tokens, entity):
        """
        Function to test if a given entity contains at least one of a list of tokens.
        param: tokens: A list of tokens
        param: entity: A span
        returns: Boolean
        """
        tokens_ = [t.text for t in tokens]
        for token in entity:
            if token.text in tokens_:
                return True
        return False

    def get_distance_from_root(self, token, root):
        """
        Gets the parse tree distance between a token and the sentence root.
        :param token: a token
        :param root: the root token of the sentence
        returns: an integer distance
        """
        if token == root:
            return 0
        d = 1
        p = token.head
        while p is not root:
            d += 1
            p = p.head
        return d

    def get_common_ancestors(self, tokens):
        ancestors = [set(t.ancestors) for t in tokens]
        if len(ancestors) == 0:
            return []
        common_ancestors = ancestors[0].intersection(*ancestors)
        return common_ancestors

    def get_distance_between_tokens(self, token_a, token_b):

        if token_b in token_a.subtree:
            distance = self.get_distance_from_root(token_b, token_a)
        elif token_a in token_b.subtree:
            distance = self.get_distance_from_root(token_a, token_b)
        else:
            common_ancestors = self.get_common_ancestors([token_a, token_b])
            distance = 10000
            for ca in common_ancestors:
                distance_a = self.get_distance_from_root(ca, token_a)
                distance_b = self.get_distance_from_root(ca, token_b)
                distance_ab = distance_a + distance_b
                if distance_ab < distance:
                    distance = distance_ab
        return distance

    def get_closest_contiguous_location_block(self, entity_list, root_node):
        location_entity_tokens = [[token for token in sentence]
                                  for sentence in entity_list]
        token_list = [
            item for sublist in location_entity_tokens for item in sublist]
        location_tokens_by_distance = sorted([(token, self.get_distance_between_tokens(token, root_node))
                                              for token in token_list], key=lambda x: x[1])
        closest_location = location_tokens_by_distance[0]
        contiguous_block = [closest_location]
        added_tokens = 1
        while added_tokens > 0:
            contiguous_block_ancestors = [[token for token in token_list if token.is_ancestor(toke)] for toke in
                                          contiguous_block]
            contiguous_block_subtrees = [
                token.subtree for token in contiguous_block]
            contiguous_block_neighbours = contiguous_block_ancestors + contiguous_block_subtrees
            contiguous_block_neighbours = [
                item for sublist in contiguous_block_neighbours for item in sublist]
            added_tokens = 0
            for toke in token_list:
                if not self.check_if_collection_contains_token(toke, contiguous_block):
                    if toke in contiguous_block_neighbours:
                        added_tokens += 1
                        contiguous_block.append(toke)
        return contiguous_block

    def get_contiguous_tokens(self, token_list):
        common_ancestor_tokens = self.get_common_ancestors(token_list)
        highest_contiguous_block = []
        for toke in token_list:
            if self.check_if_collection_contains_token(toke.head, common_ancestor_tokens):
                highest_contiguous_block.append(toke)
        added_tokens = 1
        while added_tokens > 0:
            added_tokens = 0
            for toke in token_list:
                if self.check_if_collection_contains_token(toke.head, highest_contiguous_block):
                    if not self.check_if_collection_contains_token(toke, highest_contiguous_block):
                        highest_contiguous_block.append(toke)
                        added_tokens += 1
        return highest_contiguous_block

    def match_entities_in_block(self, entities, token_block):
        matched = []
        # For some reason comparing identity on tokens does not always work.
        text_block = [t.text for t in token_block]
        for e in entities:
            et = [t.text for t in e]
            et_in_b = [t for t in et if t in text_block]
            if len(et_in_b) == len(et):
                matched.append(e)
        return matched

    def convert_to_facts(self, fact_array, fact_type, start_offset=0):
        """
        Convert extracted Spacy Tokens and Spans to Fact objects.
        param: fact_array       array of Spacy Tokens or Spans
        param: fact_type        type of Fact, i.e. Unit, Term, Quantity
        param: start_offset     Start offset (index) from beginning of article
        returns: A list of Facts
        """
        facts = []
        for fact in fact_array:
            if isinstance(fact, Token):
                
                facts.append(Fact(fact, fact, fact.lemma_,
                                  fact_type, start_offset))
            elif isinstance(fact, Span):
                
                facts.append(Fact(fact[0], fact, fact.lemma_, fact_type, start_offset))
        return facts

    def extract_locations(self, sentence, root=None):
        """
        Examine a sentence and identifies if any of its constituent tokens describe a location.
        If a root token is specified, only location tokens below the level of this token in the tree will be examined.
        If no root is specified, location tokens will be drawn from the entirety of the span.
        param: sentence       a span
        param: root           a token
        returns: A list of strings, or None
        """

        if not root:
            root = sentence.root
        descendents = self.get_descendents(sentence, root)
        
        location_entities = [e for e in self.nlp(
            sentence.text).ents if e.label_ == "GPE"]
        
        if len(location_entities) > 1:
            descendent_location_tokens = []
            for location_ent in location_entities:
                if self.check_if_entity_contains_token(location_ent, descendents):
                    descendent_location_tokens.extend(
                        [token for token in location_ent])
            contiguous_token_block = self.get_contiguous_tokens(
                descendent_location_tokens)

            block_locations = self.match_entities_in_block(
                location_entities, contiguous_token_block)
            if len(block_locations) > 0:
                return self.convert_to_facts(block_locations, "loc", sentence[0].idx)
            else:
                # If we cannot decide which one is correct, choose them all
                return self.convert_to_facts(location_entities, "loc", sentence[0].idx)
                # and figure it out at the report merging stage.
        elif len(location_entities) == 1:
            
            return self.convert_to_facts(location_entities, "loc", sentence[0].idx)
        else:
            return []

    def date_likelihood(self, possible_date, publication_date=None):
        '''
        Excludes dates that are unlikely to represent the event date in the story.
        Parameters
        ----------
        possible_dates_tokens:    a datetime - the date to be evaluated
        publication_date:   a datetime - the publication date of the story (the Article)
        Returns
        -------
        True or False depending on whether date likely represents the dates of events in the story
        '''
        # 1. Check date is not in future compared to publication date:
        if publication_date and possible_date > publication_date:
            return False
        # 2. Check date is not too far in the past vs. publication date:
        if publication_date and (publication_date - possible_date).days > 366:
            return False
        # Otherwise return True; function can be expanded to consider other
        # cases
        return True

    def basic_number(self, token):
        """
        Test if a token is equivalent to a relevant number using
        Spacy like_num method and comparing to list of strings
        param: Token    A Spacy Token
        returns: True or False
        """
        if token.lemma_ in ("dozen", "hundred", "thousand", "fifty", "percent", "percentage"):
            return True
        if token.like_num:
            return True
        if token.pos_ == u'NUM':
            return True
        else:
            return False


    def process_sentence_new(self, sentence, locations_memory, story):
        """
        Extracts the main verbs from a sentence as a starting point
        for report extraction.
        """
        sentence_reports = []
        # Find the verbs
        main_verbs = get_main_verbs_of_sent(sentence)
        for v in main_verbs:
            unit_type, verb = self.verb_relevance(v, story)                     
            if unit_type:               
                reports = self.branch_search_new(verb, unit_type, locations_memory, sentence,
                                                 story)
                if reports:
                    sentence_reports.extend(reports)
        return sentence_reports

    def article_relevance(self, article):
        """
        Tesintet article for relevance based on certain pre-defined terms.
        param: article      An instance of Spacy doc class
        return: True if article is relevant
        """
        relevant_article_lemmas = self.dic['relevant_article_lemmas']
        for token in article: 
            if token.lemma_ in relevant_article_lemmas:
                return True

    

    def verb_relevance(self, verb, article):
        """
        Checks a verb for relevance by:
        1. Comparing to structure term lemmas
        2. Comparing to person term lemmas
        3. Looking for special cases such as 'leave homeless'
        """
        # case for eviction first because we have 'forced eviction' case which would be picked by the 'elif' below
        
        if 'eviction' in [obj.lemma_ for obj in get_objects_of_verb(verb)]:
            verb_objects = get_objects_of_verb(verb)
            for verb_object in verb_objects:
                if verb_object.text == 'eviction' or verb_object.text == 'evictions':
                    return self.dic['structure_unit_lemmas'] + self.dic['person_unit_lemmas'], Fact(verb, article[verb.i: verb_object.i + 1],
                                                        verb.lemma_ + " " + "eviction", "term")
        elif verb.lemma_ in self.dic['joint_term_lemmas']:
            return self.dic['structure_unit_lemmas'] + self.dic['person_unit_lemmas'], Fact(verb, verb, verb.lemma_, "term").__str__()
        elif verb.lemma_ in self.dic['structure_term_lemmas']:
            return self.dic['structure_unit_lemmas'], Fact(verb, verb, verb.lemma_, "term")
        elif verb.lemma_ in self.dic['person_term_lemmas']:
            return self.dic['person_unit_lemmas'], Fact(verb, verb, verb.lemma_, "term")

        elif verb.lemma_ in ('leave', 'render', 'become'):
            children = verb.children
            obj_predicate = None
            for child in children:
                if child.dep_ in ('oprd', 'dobj', 'acomp'):
                    obj_predicate = child
            if obj_predicate:
                if obj_predicate.lemma_ in self.dic['structure_term_lemmas']:
                    return self.dic['structure_unit_lemmas'], Fact(verb, article[verb.i: obj_predicate.i + 1],
                                                            'leave ' + obj_predicate.lemma_, "term")

                elif obj_predicate.lemma_ in self.dic['person_term_lemmas']:
                    return self.dic['person_unit_lemmas'], Fact(verb, article[verb.i: obj_predicate.i + 1],
                                                         'leave ' + obj_predicate.lemma_, "term")

        elif verb.lemma_ == 'affect' and self.article_relevance(article):
            return self.dic['structure_unit_lemmas'] + self.dic['person_unit_lemmas'], Fact(verb, verb, verb.lemma_, "term")

        elif verb.lemma_ in ('fear', 'assume'):
            verb_objects = get_objects_of_verb(verb)
            if verb_objects:
                verb_object = verb_objects[0]
                if verb_object.lemma_ in self.dic['person_term_lemmas']:
                    return self.dic['person_unit_lemmas'], Fact(verb, article[verb.i: verb_object.i + 1],
                                                         verb.lemma_ + " " + verb_object.text, "term")

                elif verb_object.lemma_ in self.dic['structure_term_lemmas']:
                    return self.dic['structure_unit_lemmas'], Fact(verb, article[verb.i: verb_object.i + 1],
                                                            verb.lemma_ + " " + verb_object.text, "term")

        elif verb.lemma_ == 'claim':
            verb_objects = get_objects_of_verb(verb)
            for verb_object in verb_objects:
                if verb_object.text == 'lives':
                    return self.dic['person_unit_lemmas'], Fact(verb, article[verb.i: verb_object.i + 1],
                                                         verb.lemma_ + " " + "lives", "term")

        return None, None

    def get_quantity_from_phrase(self, phrase, offset = 0):
        """
        Look for number-like tokens within noun phrase.
        param: phrase   A sequence of Spacy Tokens
        param: offset   The index offset from beginning of article, an int
        return: Fact instance
        """
        for i in range(len(phrase)):
            ### Find the starting number-like tokens
            if self.basic_number(phrase[i]):
                ### Corner case 1: if the token i is the last token in the given noun phrase
                if i == len(phrase) - 1:
                    return Fact(phrase[i], phrase[i], phrase[i].lemma_, "quantity", start_offset=offset)
                ### Corner case 2: if the token i and i+ 1 are the last token in the given noun phrase and they are both number-like tokens. 
                if (i == len(phrase) -2) and self.basic_number(phrase[i+1]):
                    return Fact(phrase[i:], phrase[i:], None, "quantity", start_offset=offset)
                ### Given the token i is the num-like token, find the consecutive token j until it fails. 
                for j in range(i+1, len(phrase)):
                    if self.basic_number(phrase[j]) == False:
                        return Fact(phrase[i:j], phrase[i:j], None, "quantity", start_offset=offset)
                return Fact(phrase[i:], phrase[i:], None, "quantity", start_offset=offset)
        return Fact(None)

            
  
    def get_quantity2(self, sentence, unit):
        """
        Split a sentence into noun phrases.
        Search for quantities within each noun phrase.
        If the noun phrase is part of a conjunction, then
        search for quantity within preceding noun phrase
        """
        quantity = Fact(None)
        quantities = []
        noun_phrases = list(self.nlp(sentence.text).noun_chunks)
        
        # Case one - if the unit is a conjugated noun phrase,
        # look for numeric tokens descending from the root of the phrase.
        for i, np in enumerate(noun_phrases):
            if self.check_if_collection_contains_token2(unit, np):
                ## Try getting quantity from current noun phrase

                quantity = self.get_quantity_from_phrase(
                        np, offset=sentence[0].idx)
                
                if not quantity.token:
                    quantity = self.get_quantity_from_phrase(
                        noun_phrases[i - 1], offset=sentence[0].idx)
                    
                quantities.append(quantity)

        test_quantity = [x for x in quantities if x.text != ''] 
        if test_quantity:
            return test_quantity
        else:
            quantities = []
            for child in unit.children:
                if self.basic_number(child):
                    quantities.append(Fact(child, child, child.lemma_, "quantity"))
        return quantities
                
            
        # Case two - get any numeric child of the unit noun.
        if quantity.token:
            quantities.append(quantity)
        else:
            for child in unit.children:
                if self.basic_number(child):                  
                    quantities.append(Fact(child, child, child.lemma_, "quantity"))
        return quantities

    def get_quantity_test(self, sentence, unit):
        '''
        Check a single sentence whether there exist a quantity word or phrase
        @sentence (string): given sentence that we wanna extract
        @unit (Spacy Span): Target word accociated with quantity words
        '''
        test_sentence = [x for x in self.get_quantity2(sentence, unit) if x.text != '']
        if test_sentence:
            return True
        return False


    def get_all_quantity(self, story, unit):
        sentences = self.nlp(story).sents
        result = []
        for x in sentences:
            if self.get_quantity(x, unit).text != '':
                result.append(self.get_quantity(x, unit))
        return result


    def get_all_quantity2(self, story, unit):
        '''
        Get all quantity words or phrase given the story (article) string
        @story(string): input string
        @unit(Spacy span): target word accociated to the quantity
        '''
        sentences = self.nlp(story).sents
        result = []
        for x in sentences:
            if self.get_quantity_test(x, unit):
                result += self.get_quantity2(x, unit)
        return result

    def simple_subjects_and_objects(self, verb):
        """
        Extract all simple subjects and objects for a given verb.
        Uses Textacy get_objects_of_verb and get_subjects_of_verb methods
        param: verb     A Spacy Token
        return: A list of verb subjects and objects (Spacy Tokens or Spans)
        """
        
        verb_objects = get_objects_of_verb(verb)
        
        verb_subjects = get_subjects_of_verb(verb)
        
        verb_objects.extend(verb_subjects)
        return verb_objects

    def nouns_from_relative_clause(self, sentence, verb):
        """
        Given a sentence and verb, look for relative clauses and 
        identify nouns  
        param: sentence     A Spacy Span
        param: verb     A SPacy Token
        return: A Spacy token (the extracted noun)
        """
        possible_clauses = list(
              pos_regex_matches(sentence, r'<NOUN>+<VERB>'))
        for clause in possible_clauses:
            if verb in clause:
                for token in clause:
                    if token.tag_ == 'NNS':
                        return token

    def get_subjects_and_objects(self, story, sentence, verb):
        """
        Identify subjects and objects for a verb
        Also check if a reporting unit directly precedes
        a verb and is a direct or prepositional object
        """
        # Get simple or standard subjects and objects
        # print('The verb is ', verb)
        verb_objects = self.simple_subjects_and_objects(verb)
        # print('The value in get_subjects and objects is ', verb_objects)
        # Special Cases
        # see if unit directly precedes verb
        
        if verb.i > 0:
            preceding = story[verb.i - 1]
            if preceding.dep_ in ('pobj', 'dobj', 'nsubj', 'conj') and preceding not in verb_objects:
                verb_objects.append(preceding)

        # see if unit directly follows verb
        if verb.i < len(story) - 1:
            following = story[verb.i + 1]
            if following.dep_ in ('pobj', 'dobj', 'ROOT') and following not in verb_objects:
                verb_objects.append(following)

        # See if verb is part of a conjunction
        if verb.dep_ == 'conj':
            lefts = list(verb.lefts)
            if len(lefts) > 0:
                for token in lefts:
                    if token.dep_ in ('nsubj', 'nsubjpass'):
                        verb_objects.append(token)
            else:
                ancestors = verb.ancestors
                for anc in ancestors:
                    verb_objects.extend(self.simple_subjects_and_objects(anc))

        #
        if verb.dep_ in ("xcomp", "acomp", "ccomp"):
            ancestors = verb.ancestors
            for anc in ancestors:
                verb_objects.extend(self.simple_subjects_and_objects(anc))

        # Look for 'pobj' in sentence
        if verb.dep_ in ['ROOT','xcomp']:
            for token in sentence:
                if token.dep_ == 'pobj':
                    verb_objects.append(token)

        # Look for nouns in relative clauses
        if verb.dep_ == 'relcl':
            relcl_noun = self.nouns_from_relative_clause(sentence, verb)
            if relcl_noun:
                verb_objects.append(relcl_noun)

        return list(set(verb_objects))

    def test_noun_conj(self, sentence, noun):
        """
        Given a sentence and verb, look for conjunctions containing
        that noun
        param: sentence     A Spacy Span
        param: noun     A Spacy Token
        return: A Spacy span (the conjunction containing the noun)
        """
        possible_conjs = list(extract.pos_regex_matches(
            sentence, r'<NOUN><CONJ><NOUN>'))
        for conj in possible_conjs:
            if noun in conj:
                return conj

    def next_word(self, story, token):
        """
        Get the next word in a given story based on a passed token.
        param: story     A Spacy doc instance
        param: token     A Spacy Token
        return: A Spacy token
        """
        if token.i == len(story) - 1:
            return None
        else:
            return story[token.i + 1]

    def set_report_span(self, facts):
        """
        Convert a list of facts into their corresponding
        marker spans for visualizing with Displacy
        param facts: a list of Facts
        return: A list of fact_spans (dictionaries)
        """
        report_span = []
        for f in facts:
            if isinstance(f, list):
                sub_spans = self.set_report_span(f)
                report_span.extend(sub_spans)
            # Make sure that the fact is not None (specifically for the case of
            # Quantities)
            elif f.token:
                span = {'type': f.type_, 'start': f.start_idx, 'end': f.end_idx}
                report_span.append(span)
    
        return report_span

    def branch_search_new(self, verb, search_type, locations_memory, sentence, story):
        """
        Extract reports based upon an identified verb (reporting term).
        Extract possible locations or use most recent locations
        Extract possible dates or use most recent dates
        Identify reporting unit by looking in objects and subjects of reporting term (verb)
        Identify quantity by looking in noun phrases.
        """
        # print('The search_type is', search_type)
        possible_locations = self.extract_locations(sentence, verb.token)
        if not possible_locations:
            possible_locations = locations_memory
        reports = []
        quantity = Fact(None)
        verb_objects = self.get_subjects_and_objects(
            story, sentence, verb.token)
        # If there are multiple possible nouns and it is unclear which is the correct one
        # choose the one with the fewest descendents. A verb object with many descendents is more likely to
        # have its own verb as a descendent.
        verb_descendent_counts = [(v, len(list(v.subtree)))
                                  for v in verb_objects]
        
        verb_objects = [x[0] for x in sorted(
            verb_descendent_counts, key=lambda x: x[1])]
        for o in verb_objects:
            if self.basic_number(o):
                # Test if the following word is either the verb in question
                # Or if it is of the construction 'leave ____', then ____ is
                # the following word
                next_word = self.next_word(story, o)
                if next_word:
                    if (next_word.i == verb.token.i or next_word.text == verb.lemma_.split(" ")[-1]
                            or (next_word.dep_ == 'auxpass' and self.next_word(story, next_word).i == verb.token.i)
                            or o.idx < verb.end_idx):
                        if search_type == self.dic['structure_term_lemmas']:
                            unit = FactUnit.HOUSEHOLDS
                        else:
                            unit = FactUnit.PEOPLE
                        quantity = Fact(o, o, o.lemma_, 'quantity')
                        report = Report(unit, self.convert_term(verb.text), [p.text for p in possible_locations],
                                        sentence.start_char, sentence.end_char,
                                        self.set_report_span([verb, quantity, possible_locations]), quantity)
                        reports.append(report)
                        # print('The report in function branch_search_new is ', reports)
                        break
            # print('The search_type is', search_type)
            elif o.lemma_ in search_type:
                reporting_unit = o
                noun_conj = self.test_noun_conj(sentence, o)
                if noun_conj:
                    reporting_unit = noun_conj
                    # Try and get a number - begin search from noun conjunction
                    # root.
                    quantity = self.get_quantity(sentence, reporting_unit.root)
                else:
                    # Try and get a number - begin search from noun.
                    quantity = self.get_quantity(sentence, o)
                
                reporting_unit = Fact(
                    reporting_unit, reporting_unit, reporting_unit.lemma_, "unit")
                report = Report(self.convert_unit(reporting_unit), self.convert_term(verb.text, reporting_unit.text),
                                [p.text for p in possible_locations],
                                sentence.start_char, sentence.end_char,
                                self.set_report_span([verb, quantity, possible_locations]), quantity)
                
                reports.append(report)
                break
        return reports
        


    def extract_all_dates(self, story, publication_date=None):
        """
        Extract all dates from an article.
        param: story     A string
        param: publication_date     A datetime
        return: A list of dates
        """
        date_times = []
        story = self.nlp(story)
        date_entities = [e for e in story.ents if e.label_ == "DATE"]
        for ent in date_entities:
            abs_date = get_absolute_date(ent.text, publication_date)
            if abs_date and self.date_likelihood(abs_date, publication_date):
                date_times.append(abs_date)
        return date_times


    def extract_all_dates2(self, story, publication_date=None):
        """
        Extract all dates from an article. Loosen the condition by removing the checking function date_likelihood because checking in practice, some of the legit date might be missed by         the condition date_likelihood
        param: story     A string
        param: publication_date     A datetime
        return: A list of dates
        """
        story = self.nlp(story)
        date_entities = [e for e in story.ents if e.label_ == "DATE"]
        return self.convert_to_facts(date_entities, "date")


    def convert_unit(self, reporting_unit):
        """
        Convert extracted reporting units to predefined terms.
        param: reporting_unit   A Fact
        return: An attribute of ReportUnit
        """
        if reporting_unit.lemma_ in self.dic['structure_unit_lemmas']:
            return FactUnit.HOUSEHOLDS
        elif reporting_unit.lemma_ in self.dic['household_lemmas']:
            return FactUnit.HOUSEHOLDS
        else:
            return FactUnit.PEOPLE

    def convert_term(self, reporting_term, reporting_unit="None"):
        """
        Convert extracted reporting terms to predefined terms.
        param: reporting_unit   A Fact
        return: An attribute of ReportTerm
        """
        reporting_term = reporting_term.split(" ")
        reporting_term = [self.nlp(t)[0].lemma_ for t in reporting_term]
        reporting_unit = reporting_unit.split(" ")
        reporting_unit = [self.nlp(t)[0].lemma_ for t in reporting_unit]
        if "refugee" in reporting_unit:
            return FactTerm.REFUGEE
        elif "asylum" in reporting_unit:
            return FactTerm.ASYLUM_SEEKER
        elif "refugee" in reporting_term:
            return FactTerm.REFUGEE
        elif "cross" in reporting_term:
            return FactTerm.REFUGEE
        elif "arrive" in reporting_term and "refugee" in reporting_unit:
            return FactTerm.REFUGEE
        elif "enter" in reporting_term and "refugee" in reporting_unit:
            return FactTerm.REFUGEE
        elif "arrive" in reporting_term and "asylum" in reporting_unit:
            return FactTerm.ASYLUM_SEEKER
        elif "enter" in reporting_term and "asylum" in reporting_unit:
            return FactTerm.ASYLUM_SEEKER
        elif "displace" in reporting_term:
            return FactTerm.DISPLACED
        elif "evacuate" in reporting_term:
            return FactTerm.EVACUATED
        elif "flee" in reporting_term:
            return FactTerm.FLED
        elif "homeless" in reporting_term:
            return FactTerm.HOMELESS
        elif "camp" in reporting_term:
            return FactTerm.CAMP
        elif len(set(reporting_term) & {"shelter", "accommodate"}) > 0:
            return FactTerm.SHELTERED
        elif "relocate" in reporting_term:
            return FactTerm.RELOCATED
        elif "destroy" in reporting_term:
            return FactTerm.DESTROYED
        elif "damage" in reporting_term:
            return FactTerm.DAMAGED
        elif "uninhabitable" in reporting_term:
            return FactTerm.UNINHABITABLE
        elif "evict" in reporting_term:
            return FactTerm.EVICTED
        elif any(evterm in reporting_term for evterm in ["eviction","evictions"]):
            return FactTerm.EVICTED
        elif "sack" in reporting_term:
            return FactTerm.SACKED
        else:
            return FactTerm.DISPLACED

    def process_article_new(self, story):
        """
        Process a story one sentence at a time
        Returns a list of reports in the story
        Parameters
        ----------
        story:      the article content:String
        """
        processed_reports = []
        story = self.nlp(story)
        sentences = list(story.sents)  # Split into sentences
        # Keep a running track of the most recent locations found in articles
        
        locations_memory = []
        for i, sentence in enumerate(sentences):  # Process sentence
            reports = []
            
            reports = self.process_sentence_new(
                sentence, locations_memory, story)
            current_locations = self.extract_locations(sentence)
            if current_locations:
                locations_memory = current_locations           
            processed_reports.extend(reports)
        return list(set(processed_reports))
        
class Fact(object):
    '''Wrapper for individual facts found within articles
    '''

    def __init__(self, token, full_span=None, lemma_=None, fact_type=None, start_offset=0):
        self.token = token
        self.type_ = fact_type
        if full_span:         
            self.text = full_span.text
        else:
            self.text = ''
        self.lemma_ = lemma_
        # Set the start index
        if isinstance(token, Token):
            self.start_idx = token.idx + start_offset
        elif isinstance(token, Span):
            self.start_idx = token[0].idx + start_offset         
        else:           
            self.start_idx = 0
        # Set the end index
        token_length = len(self.text)
        self.end_idx = self.start_idx + token_length

    def __str__(self):
        return self.text

class Report(object):
    '''Wrapper for reports extracted using rules'''

    def __init__(self, reporting_unit, reporting_term, locations, sentence_start, sentence_end, tag_spans=None,
                 quantity=None):
        if tag_spans is None:
            tag_spans = []
        self.reporting_unit = convert_tokens_to_strings(reporting_unit)
        self.reporting_term = convert_tokens_to_strings(reporting_term)
        if quantity:
            self.quantity = convert_quantity(
                convert_tokens_to_strings(quantity))
        else:
            self.quantity = (None, None)
        if locations:
            self.locations = [convert_tokens_to_strings(l) for l in locations]
        else:
            self.locations = []
        self.sentence_start = sentence_start
        self.sentence_end = sentence_end
        self.tag_spans = tag_spans
        self.sentence_idx = None

    def __repr__(self):
        locations = ",".join(self.locations)
        rep = "Locations:{} Unit:{} Term:{} Quantity:{}".format(
            locations, self.reporting_unit, self.reporting_term, self.quantity)
        return rep

def convert_quantity(value):
    '''Convert an extracted quantity to an integer.
    Solution forked from
    https://github.com/ghewgill/text2num/blob/master/text2num.py
    and enhanced with numerical and array input
    '''
    value = value.replace(",", "")
    try:
        return (int(value), None)
    except:
        return (None, value)


def convert_tokens_to_strings(value):
    if isinstance(value, Token):
        return value.text
    if isinstance(value, Span):
        return value.text
    else:
        return str(value)


def minimum_loc(spans):
    '''Find the first character location in text for each report
    '''
    locs = []
    for s in spans:
        if s['type'] != 'loc':
            locs.append(s['start'])
    return min(locs)

class FactUnit:
    PEOPLE = 'Person'
    HOUSEHOLDS = 'Household'


class FactTerm:
    DISPLACED = 'Displaced'
    EVACUATED = 'Evacuated'
    FLED = 'Forced to Flee'
    HOMELESS = 'Homeless'
    CAMP = 'In Relief Camp'
    SHELTERED = 'Sheltered'
    RELOCATED = 'Relocated'
    DESTROYED = 'Destroyed Housing'
    
    DAMAGED = 'Partially Destroyed Housing'
    UNINHABITABLE = 'Uninhabitable Housing'
    OTHER = 'Multiple/Other'
    REFUGEE = 'Refugee'
    ASYLUM_SEEKER = 'Asylum Seeker'
    EVICTED = 'Evicted'
    SACKED = 'Sacked'


class Information_extraction(object): 
    '''
    The method that ensembled the location, date and quantity extration functions
    '''
    def __init__(self, story, dic):
        '''
        @story(string): Input article which is the source for information extraction
        @dic (dictionary): Words or phrase which is relavant to displacement
        @nlp: spacy library
        '''
        self.story = story
        self.nlp = spacy.load("en_core_web_lg")
        self.dic = dic
    
    def get_location_span(self, fact_type = 'loc'):
        '''
        The function is to extract the location given an article (string) as input. 
        It will return a list of dictionary indicating the location and the position of the words in the article
        '''
        
        interpreter = Interpreter(self.nlp, self.dic)
        location = interpreter.extract_locations(self.nlp(self.story)[:])
        location_str = [x.text for x in location]
        report_span = interpreter.set_report_span(location)
        result = list(zip(location_str, report_span))
        return result

    def get_date_span(self, fact_type = 'date'):
        '''
        The function is to extract the date given an article (string) as input. 
        It will return a list of dictionary indicating the date and the position of the words in the article.
        '''
        interpreter = Interpreter(self.nlp, self.dic)
        dates = interpreter.extract_all_dates2(self.story)   
        date_str = [x.text for x in dates]
        report_span = interpreter.set_report_span(dates)
        result = list(zip(date_str, report_span))
        return result

    def get_quantity_span(self, unit):
        '''
        The function is to extract the quantity words or phrases given an article (string) as input. 
        It will return a list of dictionary indicating the quantity words or phrase and the position of the words in the article. 
        @unit (string): target word associated to quantity words, e.g. one million people. Here "people" is the target word 
        '''
        unit = self.nlp(unit)[0] ## Convert string to Spacy span 
        interpreter = Interpreter(self.nlp, self.dic)
        all_quantity = interpreter.get_all_quantity2(self.story, unit)
        quantity_str = [x.text for x in all_quantity]    
        report_span = interpreter.set_report_span(all_quantity)
        result = list(zip(quantity_str, report_span))
        return result