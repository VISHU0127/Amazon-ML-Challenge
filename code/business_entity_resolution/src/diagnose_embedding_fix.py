"""Paired full-pool diagnostic; no retraining, threshold search or output overwrite.

Reconstructs the saved 5,000-entity validation sample. Streaming blocking indexes
retain only queried keys, but count every training target and preserve posting caps
and IDF. This is equivalent to full indexes for these queries, not target sampling.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jellyfish
import joblib
import numpy as np

from blocking import CountryBlockingEngine, COMMON_ADDR_STOPWORDS, _extract_all_name_tokens
from config import ARTIFACTS_DIR, TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH, VAL_SPLIT_PATH
from normalize import normalize_name, normalize_address, normalize_country, extract_street_number, extract_postal_code, extract_legal_suffixes
from features import FEATURE_NAMES, extract_pair_features
from decision import load_decision_params, calibrated_probabilities, qualify_pairs, resolve_pairs

OUT = Path(ARTIFACTS_DIR) / 'embedding_fix'
LIMITS = {'name_token_idx':15000, 'phonetic_idx':3000, 'prefix_idx':3000,
          'addr_token_idx':3000, 'street_num_idx':1500}


def attributes(name, address, country):
    return {'raw_name':name, 'raw_addr':address, 'raw_country':country,
            'norm_name':normalize_name(name), 'norm_addr':normalize_address(address),
            'norm_country':normalize_country(country), 'street_num':extract_street_number(address),
            'postal_code':extract_postal_code(address), 'suffixes':extract_legal_suffixes(name)}


def keys_for(name, address, street):
    first = name.split()[0] if name.split() else ''
    try:
        phonetic = [jellyfish.soundex(first)] if first else []
    except Exception:
        phonetic = []
    return {'name_token_idx':_extract_all_name_tokens(name), 'phonetic_idx':phonetic,
            'prefix_idx':[name[:3]] if len(name) >= 3 else [],
            'addr_token_idx':{a for a in address.split() if len(a) >= 4 and a not in COMMON_ADDR_STOPWORDS},
            'street_num_idx':[street] if street.isdigit() and len(street) >= 2 else []}


def init_worker(query_keys):
    global QUERY_KEYS
    QUERY_KEYS = query_keys


def index_chunk(rows):
    counts = Counter()
    indexes = {c:{kind:{} for kind in LIMITS} for c in QUERY_KEYS}
    for tid, name, address, country in rows:
        country = normalize_country(country)
        counts[country] += 1
        if country not in QUERY_KEYS:
            continue
        keys = keys_for(normalize_name(name), normalize_address(address), extract_street_number(address))
        for kind, values in keys.items():
            wanted = QUERY_KEYS[country][kind]
            index = indexes[country][kind]
            for key in values:
                if key in wanted:
                    posting = index.setdefault(key, [])
                    if len(posting) <= LIMITS[kind]:
                        posting.append(tid)
    return counts, indexes


def chunks(path, size=25000):
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        batch = []
        for row in reader:
            batch.append(tuple(row[k] for k in ('entity_id','business_name','business_address','country')))
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch


class IdentityIDs:
    def __getitem__(self, key):
        return key


def prepare(workers=4):
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(VAL_SPLIT_PATH).read_text())
    rng = np.random.RandomState(42)
    rng.choice(split['train_s1_ids'], size=min(15000, len(split['train_s1_ids'])), replace=False)
    sample = set(rng.choice(split['val_s1_ids'], size=5000, replace=False))
    saved = set(np.load(Path(ARTIFACTS_DIR)/'val_features.npz')['s1_ids'])
    if not saved <= sample:
        raise ValueError('Reconstructed sample does not contain the saved validation IDs')
    (OUT/'sample_ids.json').write_text(json.dumps(sorted(sample), indent=2))
    source = {}
    for batch in chunks(TRAIN_SOURCE1):
        for sid,n,a,c in batch:
            if sid in sample:
                source[sid] = attributes(n,a,c)
    truth = {}
    with open(TRAIN_GROUND_TRUTH, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['source1_entity_id'] in sample:
                truth[row['source1_entity_id']] = set(filter(None,row['matched_entity_ids'].split(',')))
    assert len(source) == len(truth) == 5000
    query_keys = {}
    for info in source.values():
        country = info['norm_country']
        query_keys.setdefault(country, {kind:set() for kind in LIMITS})
        for kind, keys in keys_for(info['norm_name'],info['norm_addr'],info['street_num']).items():
            query_keys[country][kind].update(keys)
    engines = {c:CountryBlockingEngine(c,60) for c in query_keys}
    counts = Counter()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(query_keys,)) as pool:
        for path in (TRAIN_SOURCE2,TRAIN_SOURCE3):
            # Keep at most one chunk per worker queued (bounded input/IPC memory).
            iterator = iter(chunks(path))
            while True:
                batches = []
                for _ in range(workers):
                    batch = next(iterator, None)
                    if batch is not None:
                        batches.append(batch)
                if not batches:
                    break
                for partial_counts, partial in pool.map(index_chunk, batches):
                    counts.update(partial_counts)
                    for country, indexes in partial.items():
                        for kind, index in indexes.items():
                            full = getattr(engines[country],kind)
                            for key, postings in index.items():
                                available = LIMITS[kind] + 1 - len(full[key])
                                if available > 0:
                                    full[key].extend(postings[:available])
                print(f'Full-pool scan: {sum(counts.values()):,} targets in {time.time()-t0:.0f}s', flush=True)
    pairs = []
    for c, engine in engines.items():
        engine.n_target_docs = counts[c]
        engine.target_ids = IdentityIDs()
        for kind, dest in (('name_token_idx','name_idf'),('addr_token_idx','addr_idf')):
            for key, postings in getattr(engine,kind).items():
                getattr(engine,dest)[key] = math.log((counts[c]+1)/(len(postings)+1))+1
    for sid in sorted(source):
        info = source[sid]
        cands = engines[info['norm_country']].query_entity(info['norm_name'],info['norm_addr'],info['street_num'])
        pairs.extend((sid,tid,float(max(1,60-rank)),rank) for rank,tid in enumerate(cands,1))
    needed = {p[1] for p in pairs}
    target = {}
    del engines
    print(f'Reading attributes for {len(needed):,} candidate targets', flush=True)
    for path in (TRAIN_SOURCE2,TRAIN_SOURCE3):
        for batch in chunks(path):
            for tid,n,a,c in batch:
                if tid in needed:
                    target[tid] = attributes(n,a,c)
    assert len(target) == len(needed)
    payload = {'source':source,'target':target,'truth':truth,'pairs':pairs,'target_counts':dict(counts)}
    with open(OUT/'full_pool_pairs.pkl','wb') as f:
        pickle.dump(payload,f)
    print(f'Prepared {len(pairs):,} identical pairs for {len(source)} entities', flush=True)


def metrics(source, truth, predictions):
    tp=fp=fn=singleton_ok=singletons=0
    ps=[];rs=[];fs=[]
    for sid in source:
        actual, predicted = truth[sid], set(predictions.get(sid,[]))
        hits = len(actual & predicted)
        tp += hits; fp += len(predicted)-hits; fn += len(actual)-hits
        if not actual:
            singletons += 1
            singleton_ok += not predicted
            p=r=f=float(not predicted)
        else:
            p=hits/len(predicted) if predicted else 0.
            r=hits/len(actual)
            # Competition F0.5, precision-heavy; prior evaluator used F2 instead.
            f=1.25*p*r/(.25*p+r) if p or r else 0.
        ps.append(p);rs.append(r);fs.append(f)
    p=tp/(tp+fp) if tp+fp else 0.
    r=tp/(tp+fn) if tp+fn else 0.
    return {'pooled_precision':p,'pooled_recall':r,'pooled_f05':1.25*p*r/(.25*p+r) if p or r else 0.,
            'macro_precision':float(np.mean(ps)),'macro_recall':float(np.mean(rs)),
            'macro_f05':float(np.mean(fs)),'singleton_accuracy':singleton_ok/singletons if singletons else None,
            'singleton_count':singletons,'singleton_correct':singleton_ok,'tp':tp,'fp':fp,'fn':fn}


def score():
    from embeddings import EmbeddingCache, embedding_cosine, EMBEDDING_MODEL_NAME, EMBEDDING_REVISION
    started = time.time()
    with open(OUT/'full_pool_pairs.pkl','rb') as f:
        data=pickle.load(f)
    source,target,truth,pairs = (data[k] for k in ('source','target','truth','pairs'))
    pair_sha256 = hashlib.sha256('\n'.join(f'{s}\t{t}\t{score}\t{rank}' for s,t,score,rank in pairs).encode()).hexdigest()
    model=joblib.load(Path(ARTIFACTS_DIR)/'model.joblib')
    calibration=joblib.load(Path(ARTIFACTS_DIR)/'calibrator.joblib')
    locked=load_decision_params()
    legacy=json.loads(json.dumps(locked))
    legacy.update(optimal_threshold=.4,confidence_gap=.05)
    legacy['street_number_gate']['enabled']=False
    legacy['name_floor']['enabled']=False
    legacy['budget_caps']={'S2':None,'S3':None,'total':None}
    matrix_path=OUT/'real_features.npz'
    if matrix_path.exists():
        saved=np.load(matrix_path)
        if str(saved['pair_sha256']) != pair_sha256 or list(saved['feature_names']) != FEATURE_NAMES:
            raise ValueError('Cached diagnostic matrix does not match the current candidate pairs/features')
        if str(saved['embedding_revision']) != EMBEDDING_REVISION:
            raise ValueError('Cached diagnostic matrix uses a different embedding model revision')
        matrix=saved['X']
        assert matrix.shape == (len(pairs),len(FEATURE_NAMES))
    else:
        matrix=np.zeros((len(pairs),len(FEATURE_NAMES)),dtype=np.float32)
        cache=EmbeddingCache(Path(ARTIFACTS_DIR)/'minilm_embeddings.sqlite')
        for start in range(0,len(pairs),12000):
            batch=pairs[start:start+12000]
            strings=set()
            for sid,tid,*_ in batch:
                for info in (source[sid],target[tid]):
                    strings.update((info['norm_name'],info['norm_addr']))
            vectors=cache.get_many(strings)
            for i,(sid,tid,blocking_score,rank) in enumerate(batch,start):
                s,t=source[sid],target[tid]
                kw={}
                for prefix,info in (('s1',s),('t',t)):
                    for key in ('norm_name','norm_addr','norm_country','street_num','postal_code','suffixes'):
                        kw[f'{prefix}_{key}']=info[key]
                features=extract_pair_features(s['raw_name'],s['raw_addr'],s['raw_country'],t['raw_name'],t['raw_addr'],t['raw_country'],tid,
                    blocking_score=blocking_score,blocking_rank=rank,
                    name_emb_cosine=embedding_cosine(vectors.get(s['norm_name']),vectors.get(t['norm_name'])),
                    addr_emb_cosine=embedding_cosine(vectors.get(s['norm_addr']),vectors.get(t['norm_addr'])),**kw)
                matrix[i]=[features[k] for k in FEATURE_NAMES]
            print(f'Real embedding features: {min(start+len(batch),len(pairs)):,}/{len(pairs):,}',flush=True)
        cache.close()
        np.savez_compressed(matrix_path,X=matrix,pair_sha256=pair_sha256,
                            feature_names=FEATURE_NAMES,embedding_revision=EMBEDDING_REVISION)
    results={}
    for mode in ('proxy','real'):
        X=matrix.copy()
        if mode=='proxy':
            X[:,FEATURE_NAMES.index('name_embedding_cosine')]=X[:,FEATURE_NAMES.index('name_jaro_winkler')]
            X[:,FEATURE_NAMES.index('addr_embedding_cosine')]=X[:,FEATURE_NAMES.index('addr_token_overlap_ratio')]
        probabilities=calibrated_probabilities(model,calibration,X)
        for label,params in (('legacy',legacy),('locked',locked)):
            accepted=qualify_pairs(pairs,X,probabilities,source,target,params,FEATURE_NAMES)
            predictions=resolve_pairs(accepted,params)
            results[f'{label}_{mode}']=metrics(source,truth,predictions)
            results[f'{label}_{mode}']['by_country']={
                country:metrics([sid for sid in source if source[sid]['norm_country']==country],truth,predictions)
                for country in sorted({info['norm_country'] for info in source.values()})}
            print(f'{label}_{mode}: {json.dumps(results[f"{label}_{mode}"])}',flush=True)
    candidate_map={sid:set() for sid in source}
    for sid,tid,*_ in pairs:
        candidate_map[sid].add(tid)
    found=sum(len(truth[sid]&candidate_map[sid]) for sid in source)
    total=sum(map(len,truth.values()))
    report={'sample_size':len(source),'sample_sha256':hashlib.sha256('\n'.join(sorted(source)).encode()).hexdigest(),
            'target_counts':data['target_counts'],'pair_count':len(pairs),'blocking_recall':found/total,
            'pair_sha256':pair_sha256,'embedding_model':EMBEDDING_MODEL_NAME,'embedding_revision':EMBEDDING_REVISION,
            'model_sha256':hashlib.sha256((Path(ARTIFACTS_DIR)/'model.joblib').read_bytes()).hexdigest(),
            'calibrator_sha256':hashlib.sha256((Path(ARTIFACTS_DIR)/'calibrator.joblib').read_bytes()).hexdigest(),
            'scoring_seconds':time.time()-started,
            'locked_config':locked,'results':results,
            'limitations':['Same saved 5000 validation entities; labels previously used by calibrator, not an untouched holdout.',
                           'Full labeled training target pool used for test-scale density; test labels are unavailable.',
                           'Target uniqueness competition is among these 5000 source entities.',
                           'Earlier sprint full-pool sample/result file was not found; historical 0.835 is not a paired baseline.']}
    (OUT/'diagnostic_results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--score-only',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not args.score_only:
        prepare(args.workers)
    if not args.prepare_only:
        score()
