to_import = ['utils']

import importer
importer.import_modules(__name__, __file__, to_import)

################################################################################

import os
import re
import string
from symspellpy.symspellpy import SymSpell, Verbosity
import logging
logger = logging.getLogger('pipeline')

class DocFixer:
    def __init__(self, **kwargs):
        
        def_params = dict(
            special_words = [],
            symspell_params = dict(
                dict_path = "turkish_dict_txt.txt",
                term_index=0,
                count_index=1,
                encoding = 'utf-8',
                separator=','
            )
        )
        
        for k, def_val in def_args.items():
            self.__dict__.update({k: kwargs.get(k, def_val)})
        self.load_symspellpy(**self.symspell_params)
        
    @utils.catch('FIXDOC_FIXTEXTERROR')
    def fix_text(self, text, debug = False):
        def debug_log(msg):
            if debug:
                logger.info(msg)
        
        debug_log('REPLACE NUMBERS')
        text = self.replace_numbers(text)
        
        debug_log('ADD SPACE BEFORE & AFTER SPECIAL WORDS')
        text = self.special_words_add_space(text)
        
        text = self.fix_ibans(text)
        
        #text = self.mask_numbers(text)
        
        tokens = text.split(' ')
        
        debug_log('JOIN BIGRAMS')
        tokens = self.join_bigrams(tokens, debug)
        
        debug_log('FIX SINGLE WORDS')
        tokens = self.fix_words(tokens, debug)
        
        debug_log('FIX COMPOUNDS')
        tokens = self.fix_compounds(tokens, debug)
        
        debug_log('FIX FIX COMPOUNDS ERRORS')
        tokens = self.fix_fix_compounds(tokens, debug)
        
        text = ' '.join(tokens)
        return text
    
    @utils.catch('FIXDOC_FIXIBANSERROR')
    def fix_ibans(self, t, debug = False):
        t = re.sub(r'\s?(tr)\s?[0-9]+', r' \1 ', t)
        return t
        
    
    @utils.catch('FIXDOC_PROCESSBIGRAMSERROR')
    def join_bigrams(self, tokens, debug = False):
        # Joins bigrams
        
        # Copy tokens
        result_tokens = [t for t in tokens]
        skip_bigram = False
        
        i = 0
        for t1, t2 in zip(tokens[:-1], tokens[1:]):
            if skip_bigram\
                or '#' in t1\
                or '#' in t2\
                or t1 in [';', ':','.']\
                or t2 in [';', ':', '.']:

                skip_bigram = False
                i += 1
            else:
                old_token = f'{t1} {t2}'
                suggestion = self.lookup(old_token)
                
                # A correct word was found
                if suggestion is not None:
                    result_tokens[i] = suggestion
                    #self.tokens_nofix.append(suggestion)
                    del result_tokens[i + 1]
                    i -= 1
                    
                    if debug:
                        logger.info(f'{old_token} => {suggestion}')
                    skip_bigram = True
                i += 1
        return result_tokens
    
    @utils.catch('FIXDOC_FIXWORDSERROR')
    def fix_words(self, tokens, debug = False):
        # Copy tokens
        result_tokens = [t for t in tokens]
        
        # Replace - word based
        for i, token in enumerate(tokens):
            if not (token.count('#') > 1 or not token.isalnum()\
                    or len(token) < 3):
                suggestion = self.lookup(token)
                
                # if word will be fixed
                if suggestion is not None and suggestion != token:
                    result_tokens[i] = suggestion
                    #self.tokens_nofix.append(suggestion)
                    
                    if debug:
                        logger.info(f'{token} => {suggestion}')
                        
        return result_tokens
    
    @utils.catch('FIXDOC_FIXCOMPOUNDSERROR')
    def fix_compounds(self, tokens, debug = False):
        result_tokens = [t for t in tokens]
        
        i = 0
        for token in tokens:
            if token.count('#') > 1 or not token.isalnum() or len(token) < 3:
                # Skip this token
                i += 1
                
            else:
                suggestion = self.lookup_compound(token)
                
                if suggestion is not None and suggestion != token:
                    new_text = suggestion.split(' ')
                    if len(new_text) == 1:
                        new_tokens[i] = new_text[0]
                        #self.tokens_nofix.append(new_text[0])
                    else:
                        result_tokens = result_tokens[:i] + new_text + result_tokens[i+1:]
                        i+= len(new_text) - 1
                        
                    if debug:
                        logger.info(f'{token} => {new_text}')
        return result_tokens
    
    @utils.catch('FIXDOC_MASKNUMBERSERROR')
    def mask_numbers(self, text, debug = False):
        # Replace each numeric char with #
        
        def repl(m):
            return f" {'#' * len(m.group())} "
        text = re.sub(r'[0-9]+', repl, text)
        return text
    
    @utils.catch('FIXDOC_FIXFIXCOMPUNDSERROR')
    def fix_fix_compounds(self, tokens, debug = False):
        # Copy tokens
        result_tokens = [t for t in tokens]
        
        skip_bigram = False
        i = 0
        for t1, t2 in zip(tokens[:-1], tokens[1:]):
            if skip_bigram or not t1.isalnum() or not t2.isalnum():
                # Skip this token
                skip_bigram = False
                
            else:
                possible_bigram = f'{t1}{t2}'
                old_bigram = f'{t1} {t2}'
        
                suggestion = self.lookup(possible_bigram, return_distance = True)
                
                # Replace if previous version did exist in dictionary
                if suggestion is not None and suggestion[1] == 0:
                    result_tokens[i] = possible_bigram
                    del result_tokens[i+1]
                    i -= 1
                    
                    if debug:
                        logger.info(f'{old_bigram} => {possible_bigram}')
                    skip_bigram = True
            i+= 1
        return result_tokens
            
    @utils.catch('FIXDOC_REPLACENUMBERSERROR')
    def replace_numbers(self, text, debug = False):
        # Convert 0 to o
        text = re.sub("([a-z])[0]([a-z])", r"\1o\2", text)
        
        # Convert 9 to ı
        text = re.sub("([a-z])[9]([a-z])", r"\1ı\2", text)
        
        # Convert 1 to i
        text = re.sub("([a-z])[1]([a-z])", r"\1i\2", text)
        
        # Convert ! to i
        text = re.sub("([a-z])[!]([a-z])", r"\1i\2", text)
        return text
    
    @utils.catch('FIXDOC_FIXSPECIALWORDSERROR')
    def special_words_add_space(self, text, debug = False):
        
        # Add space between and after these special names
        for bn in self.special_names:
            text = text.replace(bn, f' {bn} ')
        text = text.replace('  ', ' ').replace('  ', ' ')
        return text
    
        
    @utils.catch('FIXDOC_LOOKUPERROR')
    def lookup(self, token, return_distance = False):
        suggestion_verbosity = Verbosity.CLOSEST
        suggestions = self.sym_spell.lookup(token,
                                       suggestion_verbosity,
                                       self.max_edit_distance_lookup)
        if len(suggestions) > 0:
            if return_distance:
                return suggestions[0].term, suggestions[0].distance
            else:
                return suggestions[0].term
        else:
            return None
    
    @utils.catch('FIXDOC_LOOKUPCOMPUNDERROR')
    def lookup_compound(self, token):
        # Mostly separates compounds, sometimes does fix words.
        
        suggestions = self.sym_spell.lookup_compound(
            token,
            self.max_edit_distance_lookup,
            ignore_non_words = True)
        
        if len(suggestions) > 0:
            return suggestions[0].term
        else:
            return None
    
    @utils.catch('FIXDOC_LOADSYMSPELLPYERROR')
    def load_symspellpy(self, **symspell_params):
        max_edit_distance_dictionary = 2
        self.max_edit_distance_lookup = 2
        prefix_length = 8
        # create object
        sym_spell = SymSpell(max_edit_distance_dictionary,
                             prefix_length,
                             count_threshold=10)

        # term_index is the column of the term and count_index is the
        # column of the term frequency
        if not sym_spell.load_dictionary(**symspell_params):

            print("Dictionary file not found")
        self.sym_spell = sym_spell