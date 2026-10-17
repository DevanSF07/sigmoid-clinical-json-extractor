import json
import unittest
from clinical.prompts import completion_example,fit_prompt

class CharacterTokenizer:
    """A deterministic test double; does not test Llama's actual tokenizer."""
    def __call__(self,text,**kwargs):return {'input_ids':[ord(c) for c in text]}
    def apply_chat_template(self,msgs,tokenize=False,add_generation_prompt=False):
        text=''.join(f"<{m['role']}>{m['content']}|END|" for m in msgs)
        if add_generation_prompt:text+='<assistant>'
        return self(text)['input_ids'] if tokenize else text

class PromptTests(unittest.TestCase):
    def test_only_completion_has_loss(self):
        t=CharacterTokenizer();target={'patient_info':{},'symptoms':[],'vital_signs':{}}
        row,_=completion_example(t,'Patient.',[],[],target,max_length=30000)
        first=next(i for i,x in enumerate(row['labels']) if x!=-100)
        self.assertGreater(first,0);self.assertEqual(row['labels'][first:],row['input_ids'][first:])
        completion=''.join(chr(x) for x in row['input_ids'][first:])
        self.assertTrue(completion.startswith('{'));self.assertTrue(completion.endswith('|END|'))
    def test_examples_drop_before_note(self):
        t=CharacterTokenizer();note='KEEP THE ENTIRE NOTE'
        prompt,_,_=fit_prompt(t,note,[],[],30000)
        limit=len(prompt)+5
        prompt,_,kept=fit_prompt(t,note,[],[{'note':'x'*2000,'target':{}}],limit)
        self.assertEqual(kept,0);self.assertIn(note,prompt)
    def test_oversize_note_fails(self):
        with self.assertRaises(ValueError):fit_prompt(CharacterTokenizer(),'x'*100,[],[],10)

if __name__=='__main__':unittest.main()
