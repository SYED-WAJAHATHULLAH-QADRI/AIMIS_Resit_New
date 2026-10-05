"""Streamlit interface; processing runs only on explicit button clicks."""
import hashlib,json,platform,re,sys,uuid
import pandas as pd
import streamlit as st
from pipeline import (DEFAULT_MODEL,clean_transcript,read_text_upload,load_whisper_model,
    transcribe_audio,record_transcript_review,run_meeting_analysis,text_hash,utc_now,redact_error_message)
from evaluation import validate_reference,reference_template,parse_numbers,calculate
from reports import json_bytes,meeting_text,email_draft,docx_bytes,metrics_csv,evidence_zip

st.set_page_config(page_title='AI Meeting Intelligence',page_icon='📝',layout='wide')
st.title('AI Meeting Intelligence')
st.caption('Upload → transcribe → review → extract → evaluate → download')

@st.cache_resource
def cached_whisper(name):
    # CPU avoids assuming the hosted server has the Colab GPU.
    return load_whisper_model(name,'cpu')

if 'project' not in st.session_state: st.session_state.project=None
with st.sidebar:
    st.header('Gemini configuration')
    try: secret=st.secrets.get('GEMINI_API_KEY','')
    except Exception: secret=''
    entered_key=st.text_input('API key (if not configured in secrets)',type='password')
    api_key=entered_key or secret
    model_name=st.text_input('Model',DEFAULT_MODEL)
    temperature=st.number_input('Temperature',0.0,2.0,0.2,0.1)
    st.caption('Model access depends on your account. No key is included in downloads.')
    st.caption('Whisper runs locally on the server CPU. Gemini receives the reviewed transcript only when generation is requested.')
    if st.button('Start another meeting'):
        st.session_state.project=None
        st.rerun()

p=st.session_state.project
if p is None:
    with st.form('upload'):
        mid=st.text_input('Meeting ID','M01')
        source=st.selectbox('Source type',['unknown','real','public','simulated'])
        upload=st.file_uploader('One audio file or UTF-8 text transcript',type=['txt','mp3','wav','m4a','ogg','flac'])
        submit=st.form_submit_button('Load meeting')
    if submit:
        try:
            if not re.fullmatch(r'M\d{2,3}',mid.upper().strip()): raise ValueError('Use an ID such as M01.')
            if upload is None: raise ValueError('Choose a file.')
            data=upload.getvalue()
            if not data or len(data)>50*1024*1024: raise ValueError('File must be nonempty and at most 50 MB.')
            p={'meeting_id':mid.upper().strip(),'source_type':source,'source_name':upload.name,
                'source_sha256':hashlib.sha256(data).hexdigest(),'created_at_utc':utc_now(),
                'session_id':uuid.uuid4().hex,'input_type':'text' if upload.name.lower().endswith('.txt') else 'audio',
                'environment':{'python':sys.version,'platform':platform.platform()},'attempts':[],'evaluations':{}}
            if p['input_type']=='text': p['raw']=read_text_upload(data);p['clean']=clean_transcript(p['raw'])
            else: st.session_state.audio_bytes=data
            st.session_state.project=p;st.rerun()
        except Exception as e: st.error(str(e))
    st.stop()

st.write(f"Meeting **{p['meeting_id']}** · Source **{p['source_type']}**")
if 'raw' not in p:
    st.audio(st.session_state.audio_bytes)
    whisper_name=st.selectbox('English Whisper model',['tiny.en','base.en','small.en'],index=1)
    st.caption('First use downloads model weights. CPU transcription may take several minutes.')
    if st.button('Transcribe audio'):
        try:
            with st.spinner('Loading model and transcribing…'):
                result=transcribe_audio(st.session_state.audio_bytes,p['source_name'],cached_whisper(whisper_name))
            p['raw']=result['raw_transcript'];p['clean']=clean_transcript(p['raw']);p['transcription']=result
            p['transcription']['record']['whisper_model_name']=whisper_name
            st.rerun()
        except Exception as e: st.error(redact_error_message(e))
    st.stop()

with st.expander('Raw transcript and transcription record'):
    st.text(p['raw'])
    if 'transcription' in p: st.json(p['transcription']['record'])

st.subheader('Human transcript review')
with st.form('review_'+p['session_id']):
    reviewed=st.text_area('Check the full text against the audio or original text',p.get('reviewed',p['clean']),height=260)
    reviewer=st.text_input('Reviewer name or ID',p.get('review_record',{}).get('reviewer',''))
    notes=st.text_input('Corrections verified, or No corrections needed',p.get('review_record',{}).get('notes',''))
    confirmed=st.checkbox('I checked the full transcript against its source')
    save_review=st.form_submit_button('Record review')
if save_review:
    try:
        record=record_transcript_review(p['clean'],reviewed,reviewer,notes,confirmed)
        p.setdefault('review_history',[]).append(record);p['review_record']=record;p['reviewed']=reviewed
        p.pop('reference',None)
        st.success('Review recorded. Earlier generation evidence remains preserved.')
    except Exception as e: st.error(str(e))
if 'review_record' not in p: st.stop()
st.download_button('Download reviewed transcript',p['reviewed'],p['meeting_id']+'_reviewed.txt')

st.subheader('Generate meeting intelligence')
choice=st.selectbox('Prompt',['structured','generic','both'])
consent=st.checkbox('Send this reviewed transcript to Gemini for the requested generation')
if st.button('Generate',disabled=not consent):
    labels=['generic','structured'] if choice=='both' else [choice]
    for label in labels:
        try:
            run=1+sum(b['record']['prompt_type']==label for b in p['attempts'])
            with st.spinner('Generating '+label+' output…'):
                bundle=run_meeting_analysis(p['reviewed'],p['review_record'],api_key,label,model_name,temperature,run)
            bundle['record']['meeting_id']=p['meeting_id'];bundle['record']['source_type']=p['source_type']
            p['attempts'].append(bundle)
            if bundle['record']['status']=='FAILED': st.error(bundle['record']['error_message'])
        except Exception as e: st.error(redact_error_message(e,api_key))

for b in p['attempts']:
    r=b['record']
    with st.expander(f"{r['prompt_type']} · run {r['run_number']} · {r['status']}"):
        st.json(r)
        if b['output']:
            st.text(meeting_text(b))
            st.caption('Email draft mirrors this output, including possible errors or omissions. Review before use.')
            st.text(email_draft(b))
            st.download_button('JSON evidence',json_bytes(b),r['attempt_id']+'.json',key='json_'+r['attempt_id'])
            st.download_button('Word report',docx_bytes(b),r['attempt_id']+'.docx',key='doc_'+r['attempt_id'])
            st.download_button('Email draft',email_draft(b),r['attempt_id']+'_email.txt',key='email_'+r['attempt_id'])

st.subheader('Reference annotation and evaluation')
st.caption('Use transcript-based reviewer annotations. AI-assisted or output-exposed annotations are provisional. No consensus is inferred from an upload.')
with st.expander('Prepare or upload reference JSON'):
    st.download_button('Blank reference template',json_bytes(reference_template(p['reviewed'])),p['meeting_id']+'_reference_template.json')
    st.code('''Decision: {"id":"D1","decision":"...","evidence":["exact quote"]}
Action: {"id":"A1","action":"...","owner":null,"deadline":null,"condition":null,"evidence":["exact quote"]}''',language='text')
    ref_upload=st.file_uploader('Reference JSON (your Colab frozen reference is accepted)',type=['json'],key='ref_'+p['session_id'])
    if ref_upload and st.button('Load reference for editing'):
        try: st.session_state['refdraft_'+p['session_id']]=ref_upload.getvalue().decode('utf-8')
        except Exception as e: st.error(str(e))
    with st.form('reference_'+p['session_id']):
        ref_text=st.text_area('Reference JSON',key='refdraft_'+p['session_id'],height=240)
        ref_confirm=st.checkbox('I checked annotation content, quotes and attribution rules against the transcript')
        freeze=st.form_submit_button('Use reviewed reference')
    if freeze:
        try:
            if not ref_confirm: raise ValueError('Check the reference before accepting it.')
            p['reference']=validate_reference(json.loads(ref_text),p['reviewed'])
            st.success('Reference accepted for this session; reviewer metadata retained.')
        except Exception as e: st.error(str(e))

current=[b for b in p['attempts'] if b['output'] and b['record']['transcript_sha256']==text_hash(p['reviewed'])]
if current and p.get('reference'):
    ref=p['reference']
    b=st.selectbox('Generation attempt to evaluate',current,format_func=lambda b:f"{b['record']['prompt_type']} run {b['record']['run_number']} ({b['record']['attempt_id'][:8]})")
    evaluation_reviewer=st.text_input('Evaluation reviewer name or ID',key='evaluator_'+p['session_id'])
    rid=b['record']['attempt_id'];rh=text_hash(json.dumps(ref,sort_keys=True));ek=rid+'_'+rh[:12]
    out=b['output']
    for kind,field in [('decisions','decision'),('action_items','action')]:
        st.write(kind)
        for n,item in enumerate(out[kind],1): st.write(f"{n}. {item[field]}")
    st.caption('Coverage: enter model item numbers, or 0 for missing. Reuse a combined action number for each task it covers. Match within the same category. Quotations alone are not extracted tasks.')
    cover_rows=[{'collection':kind,'reference_id':i['id'],'reference_text':i[field],'model_numbers':''}
        for kind,field in [('decisions','decision'),('action_items','action')] for i in ref[kind]]
    with st.form('coverage_'+ek):
        edited=st.data_editor(pd.DataFrame(cover_rows),disabled=['collection','reference_id','reference_text'],hide_index=True,key='coverage_table_'+ek)
        submit_cover=st.form_submit_button('Record coverage')
    if submit_cover:
        try:
            mappings=[{'collection':r['collection'],'reference_id':r['reference_id'],
                'model_item_numbers':parse_numbers(r['model_numbers'],len(out[r['collection']]))} for r in edited.to_dict('records')]
            st.session_state['maps_'+ek]=mappings
        except Exception as e: st.error(str(e))
    maps=st.session_state.get('maps_'+ek)
    if maps is not None:
        ga={i['id']:i for i in ref['action_items']};frows=[]
        for m in maps:
            if m['collection']!='action_items': continue
            for n in m['model_item_numbers']:
                for f in ['owner','deadline','condition']:
                    frows.append({'reference_id':m['reference_id'],'model_item_number':n,'field':f,
                        'reference_value':json.dumps(ga[m['reference_id']][f]),'model_value':json.dumps(out['action_items'][n-1][f]),'judgment':'PENDING'})
        srows=[{'collection':kind,'model_item_number':n,'text':i[field],'judgment':'PENDING','reason':''}
            for kind,field in [('decisions','decision'),('action_items','action')] for n,i in enumerate(out[kind],1)]
        st.caption('Fields: YES only for equivalent values, including null versus null. Missing action fields receive no credit across all reference actions. Support: task text only; all fields are checked separately.')
        with st.form('judgments_'+ek+'_'+text_hash(json.dumps(maps))[:8]):
            fedit=st.data_editor(pd.DataFrame(frows,columns=['reference_id','model_item_number','field','reference_value','model_value','judgment']),
                disabled=['reference_id','model_item_number','field','reference_value','model_value'],hide_index=True,
                column_config={'judgment':st.column_config.SelectboxColumn(options=['PENDING','YES','NO'],required=True)})
            sedit=st.data_editor(pd.DataFrame(srows,columns=['collection','model_item_number','text','judgment','reason']),
                disabled=['collection','model_item_number','text'],hide_index=True,
                column_config={'judgment':st.column_config.SelectboxColumn(options=['PENDING','SUPPORTED','PARTIAL','UNSUPPORTED','UNCERTAIN'],required=True)})
            evaluate=st.form_submit_button('Calculate from completed judgments')
        if evaluate:
            try:
                if not evaluation_reviewer.strip(): raise ValueError('Enter the evaluation reviewer name or ID.')
                fields=fedit.to_dict('records')
                if any(r['judgment'] not in ['YES','NO'] for r in fields): raise ValueError('Answer every field; do not accept everything automatically.')
                for r in fields: r['correct']=r['judgment']=='YES'
                report=calculate(ref,out,maps,fields,sedit.to_dict('records'))
                report.update(meeting_id=p['meeting_id'],attempt_id=rid,prompt_type=b['record']['prompt_type'],
                    evaluation_reviewer=evaluation_reviewer.strip(),reference_metadata={k:ref.get(k) for k in ['ai_assisted','reviewer_exposed_to_model_outputs','independent_second_review_completed','annotation_status']})
                p['evaluations'][ek]=report
                st.success('Evaluation saved for this attempt and reference.')
            except Exception as e: st.error(str(e))
    if ek in p['evaluations']:
        report=p['evaluations'][ek];st.dataframe(pd.DataFrame(report['metrics']),hide_index=True)
        st.download_button('Evaluation JSON',json_bytes(report),ek+'_evaluation.json')
        st.download_button('Evaluation CSV',metrics_csv(report),ek+'_evaluation.csv')

st.subheader('Download session evidence')
st.caption('Includes transcripts, review records, all successful/failed attempts and saved evaluations. Session memory is temporary; download before closing. Audio is excluded; retain your original audio separately.')
st.download_button('Download evidence ZIP',evidence_zip(p),p['meeting_id']+'_evidence.zip',mime='application/zip')
