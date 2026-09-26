"""Regression checks against the actual production feature path and full blocker."""
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from blocking import CountryBlockingEngine
from decision import load_decision_params, resolve_pairs, passes_pair_gates
from diagnose_embedding_fix import keys_for, init_worker, index_chunk, IdentityIDs, metrics, attributes
from embeddings import EmbeddingCache, load_embedding_model
from features import FEATURE_NAMES
from run_stage6_inference import run_country_inference


class CaptureModel:
    def predict_proba(self, matrix):
        self.matrix = matrix.copy()
        return np.tile([.01,.99],(len(matrix),1))


class IdentityCalibrator:
    def predict(self, probabilities):
        return probabilities


class EmbeddingInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = load_embedding_model('cpu')

    def test_production_columns_equal_direct_training_encoding(self):
        columns=['entity_id','business_name','business_address','country']
        s1=pd.DataFrame([['S1-1','Patanjali','12 Main Road','New Country']],columns=columns)
        s2=pd.DataFrame([['S2-1','पतंजलि','12 Main Rd','New Country'],
                         ['S2-2','Patanjali','', 'New Country']],columns=columns)
        s3=pd.DataFrame([],columns=columns)
        model=CaptureModel()
        with tempfile.TemporaryDirectory() as directory:
            cache=EmbeddingCache(Path(directory)/'vectors.sqlite',model=self.encoder)
            candidates=io.StringIO(); matches=io.StringIO()
            run_country_inference('New Country',s1,s2,s3,model,
                {'type':'isotonic','calibrator':IdentityCalibrator()},candidates,matches,cache,load_decision_params())
            targets=dict(zip(s2.entity_id,[attributes(row.business_name,row.business_address,row.country) for row in s2.itertuples()]))
            source=attributes('Patanjali','12 Main Road','New Country')
            tids=candidates.getvalue().strip().split('\t')[1].split(',')
            for row,tid in enumerate(tids):
                for field,feature in [('norm_name','name_embedding_cosine'),('norm_addr','addr_embedding_cosine')]:
                    left,right=source[field],targets[tid][field]
                    if left and right:
                        vectors=self.encoder.encode([left,right],batch_size=512,normalize_embeddings=True)
                        expected=float(np.dot(vectors[0],vectors[1]))
                    else:
                        expected=0.
                    self.assertAlmostEqual(float(model.matrix[row,FEATURE_NAMES.index(feature)]),expected,places=5)
            row=tids.index('S2-1')
            self.assertNotAlmostEqual(float(model.matrix[row,FEATURE_NAMES.index('name_embedding_cosine')]),
                                      float(model.matrix[row,FEATURE_NAMES.index('name_jaro_winkler')]),places=3)
            cached_path = Path(directory) / 'candidate_pairs.tsv'
            cached_path.write_text('source1_entity_id\tcandidate_entity_ids\n' + candidates.getvalue())
            cached_model = CaptureModel()
            cached_candidates, cached_matches = io.StringIO(), io.StringIO()
            run_country_inference('New Country',s1,s2,s3,cached_model,
                {'type':'isotonic','calibrator':IdentityCalibrator()},
                cached_candidates,cached_matches,cache,load_decision_params(),
                candidate_input=cached_path)
            np.testing.assert_array_equal(model.matrix, cached_model.matrix)
            self.assertEqual(candidates.getvalue(), cached_candidates.getvalue())
            self.assertEqual(matches.getvalue(), cached_matches.getvalue())
            cache.close()
            # Reopened cache must not encode already-seen strings again.
            cache=EmbeddingCache(Path(directory)/'vectors.sqlite',model=self.encoder)
            original=self.encoder.encode
            self.encoder.encode=lambda *args,**kwargs: self.fail('Cached text was re-encoded')
            try:
                self.assertIn(source['norm_name'],cache.get_many([source['norm_name'],'']))
            finally:
                self.encoder.encode=original
                cache.close()

    def test_query_only_streaming_index_matches_full_index(self):
        import math
        rows=[('S2-1','Alpha Ltd','12 Main Road','US'),('S3-2','Beta Inc','12 Main St','US'),
              ('S2-3','Alpha Beta','99 Oak Road','US'),('S2-4','Unrelated','77 Elm St','US')]
        source=attributes('Alpha','12 Main Road','US')
        keys=keys_for(source['norm_name'],source['norm_addr'],source['street_num'])
        init_worker({'us':{kind:set(values) for kind,values in keys.items()}})
        counts,partial=index_chunk(rows)
        sparse=CountryBlockingEngine('us',60)
        sparse.n_target_docs=counts['us']; sparse.target_ids=IdentityIDs()
        for kind,index in partial['us'].items():
            getattr(sparse,kind).update(index)
        for kind,dest in [('name_token_idx','name_idf'),('addr_token_idx','addr_idf')]:
            for key,postings in getattr(sparse,kind).items():
                getattr(sparse,dest)[key]=math.log((counts['us']+1)/(len(postings)+1))+1
        full=CountryBlockingEngine('us',60).fit(pd.DataFrame(rows,columns=['entity_id','business_name','business_address','country']),verbose=False)
        query=(source['norm_name'],source['norm_addr'],source['street_num'])
        self.assertEqual(sparse.query_entity(*query),full.query_entity(*query))

    def test_precision_heavy_competition_metric_and_singletons(self):
        result=metrics(['a','b'],{'a':{'x'},'b':set()},{'a':['x','y']})
        self.assertAlmostEqual(result['macro_f05'],(1.25*.5/(.25*.5+1)+1)/2)
        self.assertEqual(result['singleton_accuracy'],1)

    def test_locked_street_gate_and_budgets(self):
        params=load_decision_params()
        self.assertEqual(params['optimal_threshold'],.6)
        self.assertEqual(params['budget_caps'],{'S2':5,'S3':6,'total':11})
        source=attributes('Alpha','0012 Main Road','US')
        target=attributes('Alpha','14 Main Road','US')
        features={'name_jaro_winkler':1.,'name_token_overlap_ratio':1.,'name_levenshtein_ratio':1.}
        self.assertFalse(passes_pair_gates(features,source,target,params,.99))
        target['street_num']='13'
        self.assertTrue(passes_pair_gates(features,source,target,params,.99))
        target['street_num']=''
        self.assertTrue(passes_pair_gates(features,source,target,params,.99))
        accepted=[('a',f'S2-{i}',1-i*.01) for i in range(7)]+[('b','S2-5',.5)]
        predictions=resolve_pairs(accepted,params)
        self.assertEqual(len(predictions['a']),5)
        self.assertEqual(predictions['b'],['S2-5'])

    def test_stale_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'decision.json'
            path.write_text(json.dumps({'optimal_threshold':.4,'confidence_gap':0.}))
            with self.assertRaisesRegex(ValueError,'Stale/incomplete'):
                load_decision_params(path)


if __name__=='__main__':
    unittest.main()
