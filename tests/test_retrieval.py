import unittest
from models.rag_retriever import ClinicalRAGRetriever

class RetrievalTests(unittest.TestCase):
    def test_self_and_template_duplicates_excluded(self):
        obj=object.__new__(ClinicalRAGRetriever)
        obj.rows=[{'id':'self','note':'Patient age 45 has cough'},
                  {'id':'template','note':'Patient age 46 has cough'},
                  {'id':'other','note':'Fever and nausea'},
                  {'id':'copy','note':'Fever and nausea'}]
        class Index:
            def search(self,vector,n):return [[1,.9,.8,.7]],[[0,1,2,3]]
        obj.index=Index();obj._embed=lambda *a,**kw:None
        self.assertEqual([r['id'] for r in obj.retrieve('Patient age 45 has cough',3,'self')],['other'])
    def test_empty_corpus(self):
        obj=object.__new__(ClinicalRAGRetriever);obj.rows=[]
        self.assertEqual(obj.retrieve('note'),[])
