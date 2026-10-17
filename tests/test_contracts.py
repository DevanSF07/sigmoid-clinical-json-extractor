import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from pydantic import ValidationError
from schemas.patient_schema import PatientRecord
from pipeline import ClinicalExtractionPipeline
from clinical.assertions import assertion_status
from clinical.annotations import weak_annotate,align_token_labels,TAGS
from clinical.data import split_records,near_signature
from clinical.evaluation import extraction_metrics
from clinical.ner_metrics import bio_spans,span_metrics


def target():
    return {'patient_info':{'age':45,'gender':'Male'},'visit_motivation':None,'symptoms':['cough'],'vital_signs':{}}

class SchemaTests(unittest.TestCase):
    def test_unknowns_remain_unknown(self):
        r=PatientRecord(patient_info={}).to_clean_dict()
        self.assertIsNone(r['patient_info']['age']);self.assertIsNone(r['patient_info']['gender'])
    def test_unknown_fields_rejected(self):
        with self.assertRaises(ValidationError):PatientRecord(patient_info={},invented=1)
    def test_nested_units_rejected(self):
        with self.assertRaises(ValidationError):PatientRecord(patient_info={},vital_signs={'temperature':{'value':38,'unit':'kelvin'}})
    def test_category_rejected(self):
        with self.assertRaises(ValidationError):PatientRecord(patient_info={},symptoms=['invented'])
    def test_age_is_not_coerced(self):
        with self.assertRaises(ValidationError):PatientRecord(patient_info={'age':'45'})
    def test_nonfinite_measurements_rejected(self):
        with self.assertRaises(ValidationError):PatientRecord(patient_info={},vital_signs={'heart_rate':{'value':float('nan'),'unit':'bpm'}})

class AssertionTests(unittest.TestCase):
    def status(self,note,phrase):
        s=note.index(phrase);return assertion_status(note,s,s+len(phrase))
    def test_negation(self):self.assertEqual(self.status('Patient denies chest pain.','chest pain'),'negated')
    def test_uncertain(self):self.assertEqual(self.status('Possible pneumonia.','pneumonia'),'uncertain')
    def test_historical(self):self.assertEqual(self.status('History of asthma.','asthma'),'historical')
    def test_contrast_resets(self):self.assertEqual(self.status('No fever but cough.','cough'),'present')

class AnnotationTests(unittest.TestCase):
    def test_negated_target_not_aligned(self):
        row={'id':'a','note':'Denies cough. Has fever.','target':{'symptoms':['cough','fever']}}
        a=weak_annotate(row)
        self.assertEqual([e['canonical'] for e in a['entities']],['fever'])
        self.assertFalse(a['complete_annotation'])
    def test_mask_unknown_and_special_tokens(self):
        ents=[{'start':4,'end':14,'label':'SYMPTOM'}]
        labels=align_token_labels([(0,0),(0,3),(4,9),(9,14),(0,0)],ents)
        self.assertEqual(labels,[-100,-100,TAGS.index('B-SYMPTOM'),TAGS.index('I-SYMPTOM'),-100])
    def test_complete_annotations_supervise_outside(self):
        self.assertEqual(align_token_labels([(0,2),(2,3)],[],True),[0,0])
    def test_window_restart(self):
        e=[{'start':0,'end':20,'label':'SYMPTOM'}]
        self.assertEqual(align_token_labels([(10,15),(15,20)],e),[1,2])
    def test_groups_do_not_leak(self):
        rows=[{'id':str(i),'note':f'Patient age {i} has cough'} for i in range(40)]
        splits=split_records(rows)
        nonempty=[s for s in splits.values() if s]
        self.assertEqual(len(nonempty),1)
    def test_split_is_order_independent(self):
        rows=[{'id':str(i),'note':f'note {chr(65+i)}'} for i in range(20)]
        a=split_records(rows);b=split_records(list(reversed(rows)))
        self.assertEqual({k:{r['id'] for r in v} for k,v in a.items()},{k:{r['id'] for r in v} for k,v in b.items()})

class PipelineTests(unittest.TestCase):
    def test_rules_no_guesses(self):
        r=ClinicalExtractionPipeline(mode='rules').process_note('Patient reports cough.')
        self.assertEqual(r['patient_info'],{'age':None,'gender':None});self.assertIsNone(r['visit_motivation'])
    def test_rules_negation(self):
        r=ClinicalExtractionPipeline(mode='rules').process_note('Denies chest pain. Has cough.')
        self.assertEqual(r['symptoms'],['cough'])
    def test_model_components_are_connected(self):
        calls=[]
        class NER:
            def extract_entities(self,n):calls.append('ner');return [{'text':'cough'}]
        class RAG:
            def retrieve(self,n,top_k,exclude_id):
                calls.append(('rag',exclude_id));return [{'id':'past','target':target(),'note':'cough'}]
        class Generator:
            def generate(self,n,entities,cases,constrained):
                calls.append(('generate',entities,cases,constrained));return json.dumps(target()),{'constrained':True}
        pipeline=ClinicalExtractionPipeline(ner=NER(),retriever=RAG(),generator=Generator())
        result=pipeline.process('cough','new')
        self.assertIsNone(result['error']);self.assertEqual(calls[0],'ner');self.assertEqual(calls[1],('rag','new'))
        self.assertEqual(calls[2][1],[{'text':'cough'}]);self.assertEqual(calls[2][2][0]['id'],'past')
        self.assertTrue(result['metadata']['schema_valid'])
    def test_model_failure_is_visible(self):
        class Broken:
            def extract_entities(self,n):raise RuntimeError('checkpoint failure')
        result=ClinicalExtractionPipeline(ner=Broken(),retriever=object(),generator=object()).process('x')
        self.assertIsNone(result['record']);self.assertIn('checkpoint failure',result['error'])
    def test_invalid_generated_json_is_not_repaired(self):
        class N:
            def extract_entities(self,n):return []
        class R:
            def retrieve(self,*a,**kw):return []
        class G:
            def generate(self,*a,**kw):return '{bad',{}
        r=ClinicalExtractionPipeline(ner=N(),retriever=R(),generator=G()).process('note')
        self.assertFalse(r['metadata']['json_valid']);self.assertIsNone(r['record']);self.assertEqual(r['raw'],'{bad')

class MetricTests(unittest.TestCase):
    def test_missing_prediction_counts_as_failure(self):
        m=extraction_metrics([{'id':'a','target':target()}],[])
        self.assertEqual(m['failed_records'],1);self.assertEqual(m['field_leaf_accuracy'],0)
        self.assertEqual(m['symptom_set']['recall'],0)
    def test_exact_record(self):
        t=target();m=extraction_metrics([{'id':'a','target':t}],[{'id':'a','raw':json.dumps(t)}])
        self.assertEqual(m['field_leaf_accuracy'],1);self.assertEqual(m['exact_record_match'],1)
    def test_invalid_schema_does_not_count_valid(self):
        m=extraction_metrics([{'id':'a','target':target()}],[{'id':'a','raw':'{}'}])
        self.assertEqual(m['json_validity'],1);self.assertEqual(m['schema_validity'],0)
    def test_duplicate_prediction_ids_rejected(self):
        with self.assertRaises(ValueError):extraction_metrics([{'id':'a','target':target()}],[{'id':'a'},{'id':'a'}])
    def test_span_metric_requires_exact_boundaries(self):
        m=span_metrics([[{'start':0,'end':4,'label':'SYMPTOM'}]],[[{'start':0,'end':5,'label':'SYMPTOM'}]])
        self.assertEqual(m['f1'],0)
    def test_bio_reconstruction(self):
        self.assertEqual(bio_spans(['B-X','I-X','O','B-Y']),{(0,2,'X'),(3,4,'Y')})

if __name__=='__main__':unittest.main()
