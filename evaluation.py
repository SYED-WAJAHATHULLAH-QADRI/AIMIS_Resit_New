"""Explicit reviewer judgments; no lexical matching or invented annotations."""
import hashlib,json
from pydantic import BaseModel,ConfigDict,Field
from pipeline import DecisionItem,ActionItem,text_hash,utc_now

class ReferenceDecision(DecisionItem):
    id:str=Field(min_length=1)
class ReferenceAction(ActionItem):
    id:str=Field(min_length=1)
class Reference(BaseModel):
    model_config=ConfigDict(extra='allow')
    annotator:str=Field(min_length=1)
    transcript_sha256:str
    decisions:list[ReferenceDecision]
    action_items:list[ReferenceAction]

def validate_reference(data,transcript):
    ref=Reference.model_validate(data).model_dump(mode='json')
    if ref['transcript_sha256']!=text_hash(transcript): raise ValueError('Reference belongs to a different transcript.')
    source=' '.join(transcript.split())
    for kind in ('decisions','action_items'):
        ids=[item['id'] for item in ref[kind]]
        if len(ids)!=len(set(ids)): raise ValueError('Duplicate reference IDs.')
        for item in ref[kind]:
            for quote in item['evidence']:
                if not quote.strip() or ' '.join(quote.split()) not in source:
                    raise ValueError('Reference evidence quote not found in reviewed transcript.')
    if not ref['annotator'].strip(): raise ValueError('Annotator is required.')
    if not ref['decisions'] and not ref['action_items']: raise ValueError('Reference contains no items.')
    return ref

def reference_template(transcript):
    return {'annotator':'','transcript_sha256':text_hash(transcript),
        'annotation_status':'draft_reference','ai_assisted':False,
        'reviewer_exposed_to_model_outputs':False,'independent_second_review_completed':False,
        'annotation_policy':'Describe your task splitting and owner attribution rules.',
        'decisions':[],'action_items':[]}

def parse_numbers(value,maximum):
    value=str(value).strip()
    if value=='0': return []
    try: numbers=sorted(set(int(s.strip()) for s in value.split(',')))
    except ValueError: raise ValueError('Enter matching numbers, or 0 for missing.')
    if not numbers or any(n<1 or n>maximum for n in numbers): raise ValueError('Invalid matching item number.')
    return numbers

def calculate(reference,output,coverage,fields,support):
    """Coverage and fields use reference-task units. Support uses model-item units."""
    rows=[]
    def metric(name,n,d): rows.append({'metric':name,'numerator':n,'denominator':d,'percentage':100*n/d if d else None})
    for kind in ('decisions','action_items'):
        gold_ids={i['id'] for i in reference[kind]}
        actual=[r['reference_id'] for r in coverage if r['collection']==kind]
        if set(actual)!=gold_ids or len(actual)!=len(gold_ids): raise ValueError('Coverage review is incomplete or duplicated.')
        mapped=[r for r in coverage if r['collection']==kind]
        for r in mapped:
            if any(type(n)!=int or n<1 or n>len(output[kind]) for n in r['model_item_numbers']): raise ValueError('Invalid mapping.')
        metric(kind+'_reference_coverage',sum(bool(r['model_item_numbers']) for r in mapped),len(gold_ids))
    actions=[r for r in coverage if r['collection']=='action_items']; covered=[r for r in actions if r['model_item_numbers']]
    expected={(r['reference_id'],n,f) for r in covered for n in r['model_item_numbers'] for f in ('owner','deadline','condition')}
    judgments={}
    for r in fields:
        key=(r['reference_id'],r['model_item_number'],r['field'])
        if key in judgments or type(r['correct'])!=bool: raise ValueError('Invalid field review.')
        judgments[key]=r['correct']
    if set(judgments)!=expected: raise ValueError('Complete every field judgment.')
    correct_sets=[]
    for f in ('owner','deadline','condition'):
        ids={r['reference_id'] for r in covered if all(judgments[(r['reference_id'],n,f)] for n in r['model_item_numbers'])}
        correct_sets.append(ids)
        metric(f+'_correct_on_covered_actions',len(ids),len(covered))
        metric(f+'_correct_across_all_reference_actions',len(ids),len(actions))
        known={i['id'] for i in reference['action_items'] if i[f] is not None}
        metric(f+'_correct_on_specified_reference_fields',len(ids & known),len(known))
    metric('covered_task_with_all_three_fields_correct',len(set.intersection(*correct_sets)),len(actions))
    for kind in ('decisions','action_items'):
        items=[r for r in support if r['collection']==kind]
        if sorted(r['model_item_number'] for r in items)!=list(range(1,len(output[kind])+1)): raise ValueError('Complete all support judgments.')
        for r in items:
            if r['judgment'] not in {'SUPPORTED','PARTIAL','UNSUPPORTED','UNCERTAIN'} or not r['reason'].strip(): raise ValueError('Support judgment and reason required.')
        for status in ('SUPPORTED','PARTIAL','UNSUPPORTED','UNCERTAIN'):
            metric(kind+'_task_content_'+status.lower(),sum(r['judgment']==status for r in items),len(items))
    return {'created_at_utc':utc_now(),'status':'reviewer_judgments_not_independently_verified',
        'reference_sha256':hashlib.sha256(json.dumps(reference,sort_keys=True).encode()).hexdigest(),
        'calculation':'100 * numerator / denominator; zero denominator = null (N/A)',
        'limitations':['Coverage allows combined model items to cover several reference tasks.',
            'Fields are assessed in their dedicated JSON fields; unknown-value agreement is included.',
            'Specified-reference-field metrics exclude unknown reference values and count missed tasks as incorrect.',
            'Support covers decision/task text only, not all output claims.',
            'No conventional precision, F1, annotation agreement or human time savings are claimed.'],
        'metrics':rows,'coverage':coverage,'field_judgments':fields,'support_judgments':support,'reference':reference}
