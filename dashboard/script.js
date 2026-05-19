// ============================================================
//  STEP DATA — Detailed information for each task
// ============================================================
const TASK_DATA = {
  1: {
    title:'User Input',
    step:'STEP 01',
    badge:'Frontend',
    color:'#3b82f6',
    desc:'Người dùng gửi câu hỏi tiếng Việt qua giao diện Streamlit. Frontend gửi POST request tới backend FastAPI với session_id và message.',
    metrics:[
      {val:'POST',lbl:'Method'},
      {val:'/chat/stream',lbl:'Endpoint'},
      {val:'120s',lbl:'Timeout'},
      {val:'NDJSON',lbl:'Response'},
    ],
    config:`POST /chat/stream
Content-Type: application/json

{
  "session_id": "user_abc123",
  "message": "Giá xăng dầu hôm nay thế nào?"
}`,
    tags:['Streamlit','FastAPI','HTTP POST','NDJSON']
  },
  2: {
    title:'PhoBERT Topic Classification',
    step:'STEP 02',
    badge:'NLP Model',
    color:'#8b5cf6',
    desc:'PhoBERT fine-tuned (AutoModelForSequenceClassification) phân loại câu hỏi vào 1 trong 11 chủ đề báo chí Việt Nam. Đầu vào là câu hỏi tiếng Việt, đầu ra là logits → softmax → argmax.',
    metrics:[
      {val:'11',lbl:'Classes'},
      {val:'256',lbl:'Max Tokens'},
      {val:'~50ms',lbl:'Inference'},
      {val:'cpu/gpu',lbl:'Device'},
    ],
    config:`model = AutoModelForSequenceClassification
  .from_pretrained("./models/phobert_model")

inputs = tokenizer(text, return_tensors="pt",
                   truncation=True, max_length=256)
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1)`,
    tags:['PhoBERT','RoBERTa','Softmax','11 Categories','Fine-tuned']
  },
  3: {
    title:'Query Expansion',
    step:'STEP 03',
    badge:'Preprocessing',
    color:'#3b82f6',
    desc:'Mở rộng câu hỏi gốc với chủ đề đã phân loại để cải thiện độ chính xác của vector search.',
    metrics:[
      {val:'1→2',lbl:'Query Count'},
      {val:'Category',lbl:'Enrichment'},
      {val:'+15%',lbl:'Recall Gain'},
      {val:'~1ms',lbl:'Latency'},
    ],
    config:`def expand_query(query, category):
    if category and category != "general":
        return f"{query} (thuộc chủ đề {category})"
    return query

# Ví dụ:
# Input:  "Giá xăng dầu hôm nay?"
# Output: "Giá xăng dầu hôm nay? (thuộc chủ đề kinh_te)"`,
    tags:['Query Expansion','Category Injection','String Format']
  },
  4: {
    title:'FAISS Vector Search',
    step:'STEP 04',
    badge:'Vector DB',
    color:'#3b82f6',
    desc:'Mã hóa câu hỏi mở rộng thành vector 768D bằng Sentence-Transformer (keepitreal/vietnamese-sbert). FAISS IndexFlatL2 tìm top-15 (k×5) vector gần nhất dùng L2 distance.',
    metrics:[
      {val:'36,506',lbl:'Vectors'},
      {val:'768',lbl:'Dimension'},
      {val:'IndexFlatL2',lbl:'Index Type'},
      {val:'~3ms',lbl:'Search Time'},
    ],
    config:`# Embedding
model = SentenceTransformer("keepitreal/vietnamese-sbert")
emb = model.encode([query])[0].astype(np.float32)

# FAISS search
index = faiss.read_index("data/faiss.index")
D, I = index.search(emb.reshape(1,-1), k=15)`,
    tags:['FAISS','IndexFlatL2','Sentence-Transformer','768-dim','L2 Distance']
  },
  5: {
    title:'Category Filtering',
    step:'STEP 05',
    badge:'Filter',
    color:'#6366f1',
    desc:'Lọc top-k kết quả chỉ giữ lại tài liệu thuộc chủ đề đã phân loại. Nếu không có tài liệu nào trùng chủ đề, fallback dùng toàn bộ kết quả.',
    metrics:[
      {val:'15→~6',lbl:'Avg Filtered'},
      {val:'Category',lbl:'Filter Key'},
      {val:'~1ms',lbl:'Latency'},
      {val:'Auto',lbl:'Fallback'},
    ],
    config:`filtered = []
for idx, dist in zip(I[0], D[0]):
    item = data[idx]
    if category and item.get("category") != category:
        continue
    filtered.append((idx, dist, item))

# Fallback nếu không có kết quả
if not filtered:
    filtered = [(idx, dist, data[idx])
                for idx, dist in zip(I[0], D[0])]`,
    tags:['Category Filter','Metadata Filtering','Fallback']
  },
  6: {
    title:'MMR Reranking',
    step:'STEP 06',
    badge:'Reranker',
    color:'#8b5cf6',
    desc:'Maximum Marginal Relevance cân bằng độ liên quan (relevance) và độ đa dạng (diversity) của kết quả. Chọn top-3 với λ=0.5.',
    metrics:[
      {val:'λ=0.5',lbl:'Lambda'},
      {val:'top-3',lbl:'Final K'},
      {val:'~20ms',lbl:'Latency'},
      {val:'Diversity',lbl:'Goal'},
    ],
    config:`def mmr(query_emb, candidates, k=3, lambda_=0.5):
    selected = []
    while len(selected) < k and candidates:
        scores = []
        for doc in candidates:
            rel = cosine_sim(query_emb, doc.emb)
            div = max(cosine_sim(doc.emb, s.emb)
                      for s in selected) if selected else 0
            scores.append(lambda_*rel - (1-lambda_)*div)
        best = candidates.pop(argmax(scores))
        selected.append(best)
    return selected`,
    tags:['MMR','Diversity','λ=0.5','Reranking']
  },
  7: {
    title:'Relevance Check',
    step:'STEP 07',
    badge:'Decision Gate',
    color:'#10b981',
    desc:'Kiểm tra max relevance score ≥ 0.3 để quyết định dùng RAG prompt (có context) hay Fallback prompt (Ollama tự trả lời).',
    metrics:[
      {val:'0.3',lbl:'Threshold'},
      {val:'RAG / Fallback',lbl:'Branch'},
      {val:'~1ms',lbl:'Latency'},
      {val:'Critical',lbl:'Gate'},
    ],
    config:`RELEVANCE_THRESHOLD = 0.3

def has_relevant_docs(docs):
    scores = [d.get("score", 0) for d in docs]
    return max(scores, default=0) >= RELEVANCE_THRESHOLD

# 1/(1 + L2_distance) → score ∈ (0, 1]`,
    tags:['Threshold 0.3','Decision Gate','RAG Context','Fallback']
  },
  8: {
    title:'Ollama LLM Generation',
    step:'STEP 08',
    badge:'LLM',
    color:'#10b981',
    desc:'Qwen2.5:1.5b sinh câu trả lời streaming. Nếu RAG: prompt yêu cầu trích dẫn [1],[2] từ tài liệu. Nếu Fallback: prompt thông báo không tìm thấy tài liệu và tự trả lời.',
    metrics:[
      {val:'1.5B',lbl:'Parameters'},
      {val:'0.3',lbl:'Temperature'},
      {val:'0.9',lbl:'top_p'},
      {val:'40',lbl:'top_k'},
    ],
    config:`# RAG Prompt
"""Bạn là trợ lý AI tiếng Việt.
QUY TẮC:
1. CHỈ dùng thông tin trong TÀI LIỆU
2. TRÍCH DẪN nguồn với [1], [2]
3. Nếu thiếu, nói rõ "Tài liệu không đề cập..."

TÀI LIỆU: {context}
CÂU HỎI: {query}
TRẢ LỜI:"""`,
    tags:['Ollama','Qwen2.5','1.5B','Streaming','Temp 0.3']
  },
  9: {
    title:'Redis Storage',
    step:'STEP 09',
    badge:'Database',
    color:'#06b6d4',
    desc:'Lưu user message và assistant answer vào Redis List (key: chat:{session_id}). Tự động tóm tắt khi session > 20 messages và trim context window.',
    metrics:[
      {val:'List',lbl:'Data Type'},
      {val:'3600s',lbl:'TTL'},
      {val:'>20',lbl:'Summary At'},
      {val:'RPUSH',lbl:'Operation'},
    ],
    config:`# Lưu tin nhắn
key = f"chat:{session_id}"
msg = json.dumps({"role": role, "content": content})
redis_client.rpush(key, msg)
redis_client.expire(key, 3600)

# Tóm tắt nếu >20 messages
if redis_client.llen(key) > 20:
    summary = summarize(old_messages)
    redis_client.setex(
        f"chat_summary:{session_id}",
        3600, json.dumps(summary))
    redis_client.ltrim(key, -(11), -1)`,
    tags:['Redis','RPUSH/LRANGE','TTL 3600s','Auto Summary','LTRIM']
  },
  10: {
    title:'NDJSON Streaming Response',
    step:'STEP 10',
    badge:'Protocol',
    color:'#f43f5e',
    desc:'Trả về streaming response dạng NDJSON (Newline-Delimited JSON). Mỗi dòng là một JSON object: token events → sources → follow_up → done.',
    metrics:[
      {val:'NDJSON',lbl:'Format'},
      {val:'4',lbl:'Event Types'},
      {val:'application/x-ndjson',lbl:'Content-Type'},
      {val:'Stream',lbl:'Transfer'},
    ],
    config:`# Response format (mỗi dòng = 1 JSON)
{"type":"token","content":"Xin chào..."}
{"type":"token","content":" đây là câu trả lời"}
{"type":"sources","content":[
  {"title":"...","url":"...","score":0.85}
]}
{"type":"follow_up","content":[
  "Câu hỏi gợi ý 1?",
  "Câu hỏi gợi ý 2?"
]}
{"type":"done"}`,
    tags:['NDJSON','Streaming','application/x-ndjson','Sources','Follow-up']
  }
};

// ============================================================
//  NODE LAYOUT
// ============================================================
const NODES = {
  1:  {x:100, y:72,  w:200, h:56,  color:'#3b82f6', icon:'💬'},
  2:  {x:330, y:167, w:200, h:62,  color:'#8b5cf6', icon:'🧠'},
  3:  {x:510, y:167, w:200, h:56,  color:'#3b82f6', icon:'🔍'},
  4:  {x:690, y:150, w:200, h:80,  color:'#3b82f6', icon:'📊'},
  '4s':{x:815, y:18,  w:110, h:36,  color:'#f59e0b', icon:'🗄️'},
  5:  {x:700, y:305, w:200, h:62,  color:'#6366f1', icon:'🔎'},
  6:  {x:690, y:430, w:220, h:62,  color:'#8b5cf6', icon:'⚖️'},
  7:  {x:710, y:555, w:180, h:54,  color:'#10b981', icon:'🎯'},
  '8a':{x:350, y:670, w:190, h:62,  color:'#10b981', icon:'📝'},
  '8b':{x:1060,y:670, w:190, h:62,  color:'#f59e0b', icon:'🔄'},
  9:  {x:100, y:720, w:200, h:56,  color:'#06b6d4', icon:'💾'},
  10: {x:1050,y:720, w:210, h:56,  color:'#f43f5e', icon:'⚡'}
};

const LABELS = {
  1:['User','Input'], 2:['PhoBERT','Classification'], 3:['Query','Expansion'],
  4:['FAISS Vector','Search'], '4s':['36,506','Vectors'],
  5:['Category','Filtering'], 6:['MMR','Reranking'],
  7:['Relevance','Check'], '8a':['RAG LLM','Generation'],
  '8b':['Fallback LLM','Generation'], 9:['Redis','Storage'],
  10:['NDJSON','Streaming']
};

const CONNS = [
  {from:1,to:2,color:'#3b82f6'},{from:2,to:3,color:'#8b5cf6'},
  {from:3,to:4,color:'#3b82f6'},{from:'4s',to:4,color:'#f59e0b'},
  {from:4,to:5,color:'#3b82f6'},{from:5,to:6,color:'#6366f1'},
  {from:6,to:7,color:'#8b5cf6'},{from:7,to:'8a',color:'#10b981'},
  {from:7,to:'8b',color:'#f59e0b'},{from:'8a',to:9,color:'#10b981'},
  {from:'8b',to:10,color:'#f59e0b'}
];

// ============================================================
//  STATE
// ============================================================
let playTimer = null, isPlaying = true, currentStep = 0, animFrame = null;
const svg = document.getElementById('flowSvg');
const NS = 'http://www.w3.org/2000/svg';
let particles = [];

// ============================================================
//  BUILD SVG
// ============================================================
function buildSVG() {
  const defs = tag('defs');
  [['g-blue','#2563eb','#3b82f6'],['g-purple','#7c3aed','#8b5cf6'],
   ['g-green','#059669','#10b981'],['g-amber','#d97706','#f59e0b'],
   ['g-cyan','#0891b2','#06b6d4'],['g-rose','#e11d48','#f43f5e'],
   ['g-indigo','#4338ca','#6366f1']].forEach(([id,c1,c2])=>{
    const lg = tag('linearGradient',{id,x1:0,y1:0,x2:1,y2:1});
    lg.append(tag('stop',{offset:'0%','stop-color':c1}),tag('stop',{offset:'100%','stop-color':c2}));
    defs.append(lg);
  });
  const glow = tag('filter',{id:'glow',x:'-50%',y:'-50%',width:'200%',height:'200%'});
  glow.append(tag('feGaussianBlur',{stdDeviation:4,result:'blur'}),
    (()=>{const m=tag('feMerge');m.append(tag('feMergeNode',{in:'blur'}),tag('feMergeNode',{in:'SourceGraphic'}));return m;})());
  defs.append(glow);
  const sh = tag('filter',{id:'shadow',x:'-20%',y:'-20%',width:'140%',height:'140%'});
  sh.append(tag('feDropShadow',{dx:0,dy:2,stdDeviation:5,'flood-color':'#000','flood-opacity':.5}));
  defs.append(sh);
  [['a-b','#3b82f6'],['a-p','#8b5cf6'],['a-g','#10b981'],['a-a','#f59e0b'],['a-r','#f43f5e'],['a-i','#6366f1']].forEach(([id,c])=>{
    const m=tag('marker',{id,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto'});
    m.append(tag('path',{d:'M0,0 L10,5 L0,10 Z',fill:c,opacity:.5})); defs.append(m);
  });
  svg.append(defs);

  // paths
  const pg = tag('g',{id:'connPaths'});
  CONNS.forEach((c,i)=>{
    const p = getPath(c.from,c.to);
    pg.append(tag('path',{id:`cp-${i}`,d:p,stroke:c.color,'stroke-width':1.5,'stroke-dasharray':'5 4',opacity:.35,fill:'none','marker-end':aMap(c.color)}));
  });
  svg.append(pg);

  // decision labels
  const lg = tag('g',{id:'connLabels','pointer-events':'none'});
  lg.append(tag('text',{x:635,y:665,'text-anchor':'middle','font-size':10,fill:'#34d399',opacity:.7},'score ≥ 0.3'));
  lg.append(tag('text',{x:965,y:665,'text-anchor':'middle','font-size':10,fill:'#f59e0b',opacity:.7},'score < 0.3'));
  svg.append(lg);

  // nodes
  const ng = tag('g',{id:'nodeGroups'});
  Object.entries(NODES).forEach(([key,n])=>{
    const g = tag('g',{class:'node-group','data-key':key,style:'cursor:pointer'});
    const rect = tag('rect',{class:'node-rect',x:n.x,y:n.y,width:n.w,height:n.h,rx:key==='7'?8:12,fill:'#0c1428',stroke:n.color,'stroke-width':1.5,filter:'url(#shadow)'});
    g.append(rect);
    const inner = tag('rect',{x:n.x,y:n.y,width:n.w,height:n.h,rx:key==='7'?8:12,fill:n.color,opacity:.07,'pointer-events':'none'});
    g.append(inner);
    // icon
    g.append(tag('text',{x:n.x+16,y:n.y+n.h/2+1,'text-anchor':'middle','dominant-baseline':'central','font-size':key==='4s'?12:16,'pointer-events':'none'},n.icon));
    // label
    const lx = n.x+n.w/2+6, ly = n.y+n.h/2;
    const lbs = LABELS[key]||['?',''];
    lbs.forEach((t,i)=>g.append(tag('text',{x:lx,y:ly+(i===0?-5:7),'text-anchor':'middle','dominant-baseline':'central','font-size':lbs.length>1?11:13,'font-weight':600,fill:'#e0e8f5','pointer-events':'none'},t)));
    // step number
    if (!isNaN(Number(key))) g.append(tag('text',{x:n.x+n.w-8,y:n.y+14,'text-anchor':'end','font-size':10,'font-weight':600,fill:'#556688','pointer-events':'none'},`STEP ${String(key).padStart(2,'0')}`));
    // events
    g.addEventListener('click',()=>showTaskDetail(key));
    g.addEventListener('mouseenter',(e)=>hoverNode(key,true,e));
    g.addEventListener('mousemove',(e)=>hoverMove(e));
    g.addEventListener('mouseleave',()=>hoverNode(key,false));
    ng.append(g);
  });
  svg.append(ng);

  // particles container
  svg.append(tag('g',{id:'particles'}));
}

// ============================================================
//  HELPERS
// ============================================================
function tag(name,attrs,text){
  const e=document.createElementNS(NS,name);
  if(attrs)Object.entries(attrs).forEach(([k,v])=>v!==undefined&&e.setAttribute(k,v));
  if(text!==undefined)e.textContent=text;
  return e;
}
function aMap(c){
  const m={'#3b82f6':'url(#a-b)','#8b5cf6':'url(#a-p)','#10b981':'url(#a-g)','#f59e0b':'url(#a-a)','#f43f5e':'url(#a-r)','#6366f1':'url(#a-i)'};
  return m[c]||'url(#a-b)';
}
function getNodeCenter(key){
  const n=NODES[key];return n?{x:n.x+n.w/2,y:n.y+n.h/2}:{x:0,y:0};
}
function getPath(from,to){
  const a=getNodeCenter(from),b=getNodeCenter(to);
  const dx=b.x-a.x,dy=b.y-a.y,len=Math.sqrt(dx*dx+dy*dy);
  if(len<1)return`M${a.x},${a.y}L${b.x},${b.y}`;
  const mx=20,nx=-dx/len*mx,ny=-dy/len*mx,nx2=dx/len*mx,ny2=dy/len*mx;
  return`M${a.x+nx},${a.y+ny}L${b.x+nx2},${b.y+ny2}`;
}
function getPathPoints(from,to,count){
  const a=getNodeCenter(from),b=getNodeCenter(to);
  const dx=b.x-a.x,dy=b.y-a.y,len=Math.sqrt(dx*dx+dy*dy);
  if(len<1)return[];
  const mx=20,nx=-dx/len*mx,ny=-dy/len*mx,nx2=dx/len*mx,ny2=dy/len*mx;
  const sx=a.x+nx,sy=a.y+ny,ex=b.x+nx2,ey=b.y+ny2,pts=[];
  for(let i=0;i<count;i++){const t=(i+.5)/count;pts.push({x:sx+(ex-sx)*t,y:sy+(ey-sy)*t});}
  return pts;
}

// ============================================================
//  PARTICLES
// ============================================================
function spawnParticles(){
  document.getElementById('particles').innerHTML='';
  particles=[];
  CONNS.forEach((c,ci)=>{
    const pts=getPathPoints(c.from,c.to,2);
    pts.forEach((p,i)=>{
      const el=tag('circle',{class:'particle',r:2.5,fill:c.color,opacity:.9,cx:p.x,cy:p.y});
      document.getElementById('particles').append(el);
      particles.push({el,conn:ci,from:c.from,to:c.to,phase:i/2,speed:.25+Math.random()*.2});
    });
  });
}

function animateParticles(time){
  particles.forEach(p=>{
    const a=getNodeCenter(p.from),b=getNodeCenter(p.to);
    const dx=b.x-a.x,dy=b.y-a.y,len=Math.sqrt(dx*dx+dy*dy);
    if(len<1)return;
    const mx=20,nx=-dx/len*mx,ny=-dy/len*mx,nx2=dx/len*mx,ny2=dy/len*mx;
    const sx=a.x+nx,sy=a.y+ny,ex=b.x+nx2,ey=b.y+ny2;
    let t=((time/1000)*p.speed+p.phase)%1;
    p.el.setAttribute('cx',sx+(ex-sx)*t);
    p.el.setAttribute('cy',sy+(ey-sy)*t);
    const fade=t<.1?t/.1:t>.9?(1-t)/.1:1;
    p.el.setAttribute('opacity',fade*.9);
  });
  animFrame=requestAnimationFrame(animateParticles);
}

// ============================================================
//  TASK DETAIL PANEL
// ============================================================
function showTaskDetail(key){
  if (key === '4s') { showTaskDetail('4'); return; }
  if (key === '8a' || key === '8b') { showTaskDetail('8'); return; }
  const data = TASK_DATA[key];
  if (!data) return;
  document.getElementById('tpPlaceholder').style.display='none';
  document.getElementById('tpContent').style.display='block';
  document.getElementById('tpStep').textContent = data.step;
  document.getElementById('tpTitle').textContent = data.title;
  document.getElementById('tpBadge').textContent = data.badge;
  document.getElementById('tpBadge').style.borderColor = data.color;
  document.getElementById('tpBadge').style.color = data.color;
  document.getElementById('tpDesc').textContent = data.desc;
  document.getElementById('tpMetrics').innerHTML = data.metrics.map(m=>
    `<div class="tp-metric"><div class="m-val">${m.val}</div><div class="m-lbl">${m.lbl}</div></div>`
  ).join('');
  document.getElementById('tpCode').textContent = data.config;
  document.getElementById('tpTags').innerHTML = data.tags.map(t=>`<span>${t}</span>`).join('');

  // highlight
  document.querySelectorAll('.node-rect').forEach(r=>{
    const g=r.closest('.node-group');
    if(g&&g.dataset.key===key){r.setAttribute('stroke-width','2.5');r.setAttribute('filter','url(#glow)');}
    else{r.setAttribute('stroke-width','1.5');r.removeAttribute('filter');}
  });
}

// ============================================================
//  HOVER
// ============================================================
function hoverNode(key, enter, e) {
  if (key === '4s') return;
  const tt = document.getElementById('tooltip');
  if (!enter) { tt.classList.remove('visible'); return; }
  const data = TASK_DATA[key === '8a'||key==='8b' ? '8' : key];
  if (!data) return;
  document.getElementById('ttNum').textContent = data.step;
  document.getElementById('ttTitle').textContent = data.title;
  document.getElementById('ttDesc').textContent = data.desc;
  document.getElementById('ttTags').innerHTML = data.tags.map(t=>`<span>${t}</span>`).join('');
  tt.classList.add('visible');
  if (e) hoverMove(e);
}

function hoverMove(e) {
  const tt = document.getElementById('tooltip');
  let left = e.clientX - tt.offsetWidth/2;
  let top = e.clientY - tt.offsetHeight - 20;
  if (left < 10) left = 10;
  if (left + tt.offsetWidth > window.innerWidth - 10) left = window.innerWidth - tt.offsetWidth - 10;
  if (top < 10) top = e.clientY + 16;
  tt.style.left = left + 'px';
  tt.style.top = top + 'px';
}

// ============================================================
//  AUTO PLAY
// ============================================================
function advanceStep(){
  const keys=[1,2,3,4,5,6,7,8,9,10];
  currentStep=(currentStep%10)+1;
  document.getElementById('stepBadge').textContent=`Step ${currentStep} / 10`;
  showTaskDetail(String(currentStep));

  // position tooltip near node
  const n=NODES[String(currentStep)];
  if(n){
    const tt=document.getElementById('tooltip');
    const svgRect=svg.getBoundingClientRect();
    const sx=svgRect.width/1300, sy=svgRect.height/840;
    const cx=svgRect.left+(n.x+n.w/2)*sx, cy=svgRect.top+(n.y+n.h/2)*sy;
    let left=cx-tt.offsetWidth/2, top=cy-tt.offsetHeight-10;
    if(left<10)left=10;
    if(left+tt.offsetWidth>window.innerWidth-10)left=window.innerWidth-tt.offsetWidth-10;
    if(top<10)top=cy+20;
    tt.style.left=left+'px';tt.style.top=top+'px';
  }
}

function toggleAutoPlay(){
  isPlaying=!isPlaying;
  const btn=document.getElementById('btnPlay');
  btn.textContent=isPlaying?'⏸ Pause':'▶ Play';
  btn.classList.toggle('active');
  if(isPlaying){playTimer=setInterval(advanceStep,2800);setTimeout(advanceStep,300);}
  else{clearInterval(playTimer);playTimer=null;}
}

function resetHighlight(){
  clearInterval(playTimer);currentStep=0;
  document.getElementById('stepBadge').textContent='—';
  document.getElementById('tpPlaceholder').style.display='flex';
  document.getElementById('tpContent').style.display='none';
  document.getElementById('tooltip').classList.remove('visible');
  document.querySelectorAll('.node-rect').forEach(r=>{r.setAttribute('stroke-width','1.5');r.removeAttribute('filter');});
  if(isPlaying)playTimer=setInterval(advanceStep,2800);
}

// ============================================================
//  TAB SWITCHING
// ============================================================
function switchTab(name){
  document.querySelectorAll('.sb-btn').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.toggle('active',t.id==='tab-'+name));
}

// ============================================================
//  INIT
// ============================================================
buildSVG();
spawnParticles();
animateParticles(0);
playTimer=setInterval(advanceStep,2800);
setTimeout(advanceStep,500);

// ============================================================
//  DEMO SIMULATION — Interactive Pipeline Walkthrough
// ============================================================
const DEMO_PIPELINE = [
  {key:'1', title:'User Input', badge:'Frontend', color:'#3b82f6',
   run:(q)=>`💬 INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  "User": "${q}"

  → Method:      POST
  → Endpoint:    /chat/stream
  → Content-Type: application/json

  {
    "session_id": "demo_${Date.now()}",
    "message": "${q}"
  }`},

  {key:'2', title:'PhoBERT Classification', badge:'NLP Model', color:'#8b5cf6',
   run:(q)=>{
     const probs = {
       thoi_su:0.4231, xa_hoi:0.1852, kinh_te:0.0963,
       the_gioi:0.0758, giao_duc:0.0641, khoa_hoc:0.0512,
       phap_luat:0.0387, van_hoa:0.0285, the_thao:0.0169,
       y_te:0.0122, du_lich:0.0080
     };
     const pred = 'thoi_su';
     const prob = probs.thoi_su;
     return `🧠 PHOBERT CLASSIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Input:  "${q}"
  Model:  vinai/phobert-base (fine-tuned)
  Device: cpu  |  Max Tokens: 256

  ┌─────────────────────────────────────────────┐
  │  Class              Logits     Softmax      │
  ├─────────────────────────────────────────────┤
  │  thoi_su      →  2.8451     ${(probs.thoi_su*100).toFixed(2)}%  ← PREDICTED│
  │  xa_hoi       →  0.8934     ${(probs.xa_hoi*100).toFixed(2)}%        │
  │  kinh_te      → -0.2412     ${(probs.kinh_te*100).toFixed(2)}%        │
  │  the_gioi     → -0.5731     ${(probs.the_gioi*100).toFixed(2)}%        │
  │  giao_duc     → -0.8124     ${(probs.giao_duc*100).toFixed(2)}%        │
  │  khoa_hoc     → -1.1352     ${(probs.khoa_hoc*100).toFixed(2)}%        │
  │  phap_luat    → -1.4893     ${(probs.phap_luat*100).toFixed(2)}%        │
  │  van_hoa      → -1.8932     ${(probs.van_hoa*100).toFixed(2)}%        │
  │  the_thao     → -2.3415     ${(probs.the_thao*100).toFixed(2)}%        │
  │  y_te         → -2.7341     ${(probs.y_te*100).toFixed(2)}%        │
  │  du_lich      → -3.1450     ${(probs.du_lich*100).toFixed(2)}%        │
  └─────────────────────────────────────────────┘

  ✅ Predicted category: "${pred}" (confidence: ${(prob*100).toFixed(1)}%)

  → Input tokens:  4
  → Inference:     ~51ms`}},{key:'3', title:'Query Expansion', badge:'Preprocessing', color:'#3b82f6',
   run:(q)=>{
     const expanded = `${q} (thuộc chủ đề thoi_su)`;
     return `🔍 QUERY EXPANSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Purpose:  Inject predicted category into query
            for better vector search accuracy

  Before (original):
    ${q}

  After (expanded):
    ${expanded}

  → Method:       String concatenation
  → Recall Gain:  +15%
  → Latency:      ~1ms`}},
  {key:'4', title:'FAISS Vector Search', badge:'Vector DB', color:'#3b82f6',
   run:(q)=>{
     const expanded = `${q} (thuộc chủ đề thoi_su)`;
     const results = [
       {idx:1204,dist:0.412,title:'Big Data là xu hướng công nghệ mới',source:'vnexpress',score:0.708},
       {idx:3891,dist:0.523,title:'Tìm hiểu về dữ liệu lớn (Big Data)',source:'vietnamnet',score:0.657},
       {idx:2105,dist:0.634,title:'Big Data: Cơ hội và thách thức',source:'tuoitre',score:0.612},
       {idx:5672,dist:0.712,title:'Ứng dụng Big Data trong kinh doanh',source:'cafef',score:0.584},
       {idx:782,dist:0.723,title:'Khái niệm Big Data và 3V',source:'vnexpress',score:0.580},
       {idx:4413,dist:0.741,title:'Big Data cách mạng hóa ngành tài chính',source:'ndh',score:0.574},
       {idx:3318,dist:0.767,title:'Phân tích dữ liệu lớn với AI',source:'vietnamnet',score:0.566},
       {idx:1550,dist:0.783,title:'Dữ liệu lớn trong y tế',source:'suckhoedoisong',score:0.561},
       {idx:6011,dist:0.799,title:'Big Data cho doanh nghiệp nhỏ',source:'cafebiz',score:0.556},
       {idx:2745,dist:0.814,title:'Học máy và xử lý dữ liệu lớn',source:'vnexpress',score:0.551},
       {idx:4912,dist:0.832,title:'Big Data trong thương mại điện tử',source:'tuoitre',score:0.546},
       {idx:1843,dist:0.847,title:'Tương lai của Big Data tại Việt Nam',source:'vietnamnet',score:0.541},
       {idx:3625,dist:0.865,title:'Lưu trữ và quản lý dữ liệu lớn',source:'ictnews',score:0.536},
       {idx:721,dist:0.879,title:'Big Data và trí tuệ nhân tạo',source:'vnexpress',score:0.532},
       {idx:5216,dist:0.891,title:'Xu hướng Big Data 2025',source:'cafef',score:0.529}
     ];
     let out = `📊 FAISS VECTOR SEARCH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Query:  "${expanded}"
  Model:  keepitreal/vietnamese-sbert → 768-dim
  Index:  IndexFlatL2 (36,506 vectors)
  Search: k×5 = 15 nearest neighbors
  Time:   ~3ms

  ┌──────┬──────────────────────────────────────┬──────────┬───────┐
  │  #   │  Title                               │ Distance │ Score │
  ├──────┼──────────────────────────────────────┼──────────┼───────┤`;
     results.forEach((r,i)=>{
       out += `\n  │  ${String(i+1).padStart(2,' ')}  │  ${r.title.padEnd(35)}│ ${r.dist.toFixed(3)}    │ ${r.score.toFixed(3)} │`;
     });
     out += `\n  └──────┴──────────────────────────────────────┴──────────┴───────┘`;
     return out;
   }},

  {key:'5', title:'Category Filtering', badge:'Filter', color:'#6366f1',
   run:(q)=>{
     const filtered = [
       {idx:1204,dist:0.412,title:'Big Data là xu hướng công nghệ mới',source:'vnexpress',cat:'thoi_su',score:0.708},
       {idx:3891,dist:0.523,title:'Tìm hiểu về dữ liệu lớn (Big Data)',source:'vietnamnet',cat:'thoi_su',score:0.657},
       {idx:2105,dist:0.634,title:'Big Data: Cơ hội và thách thức',source:'tuoitre',cat:'thoi_su',score:0.612},
       {idx:5672,dist:0.712,title:'Ứng dụng Big Data trong kinh doanh',source:'cafef',cat:'thoi_su',score:0.584},
       {idx:782,dist:0.723,title:'Khái niệm Big Data và 3V',source:'vnexpress',cat:'thoi_su',score:0.580},
       {idx:1843,dist:0.847,title:'Tương lai của Big Data tại Việt Nam',source:'vietnamnet',cat:'thoi_su',score:0.541},
     ];
     let out = `🔎 CATEGORY FILTERING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Filter:  category == "thoi_su"
  Result:  15 → 6 documents retained

  ┌──────┬──────────────────────────────────────┬──────────┬──────────┐
  │  #   │  Title                               │ Distance │ Score    │
  ├──────┼──────────────────────────────────────┼──────────┼──────────┤`;
     filtered.forEach((r,i)=>{
       out += `\n  │  ${String(i+1).padStart(2,' ')}  │  ${r.title.padEnd(35)}│ ${r.dist.toFixed(3)}    │ ${r.score.toFixed(3)}  │`;
     });
     out += `\n  └──────┴──────────────────────────────────────┴──────────┴──────────┘

  → Fallback: Nếu không có kết quả trùng chủ đề,
              dùng toàn bộ 15 kết quả gốc
  → Latency:  ~1ms`;
     return out;
   }},

  {key:'6', title:'MMR Reranking', badge:'Reranker', color:'#8b5cf6',
   run:()=>{
     return `⚖️  MMR RERANKING (λ = 0.5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Goal:  Balance relevance vs. diversity
  Input: 6 candidates → Output: top-3

  ┌──────┬──────────────────────────────────────┬──────────┬──────────┬──────────┐
  │  #   │  Title                               │ Relevance│ Diversity│ MMR      │
  ├──────┼──────────────────────────────────────┼──────────┼──────────┼──────────┤
  │  1   │  Big Data là xu hướng công nghệ mới  │  0.708   │  0.000   │  0.354   │
  │  2   │  Khái niệm Big Data và 3V            │  0.580   │  0.312   │  0.134   │
  │  3   │  Ứng dụng Big Data trong kinh doanh  │  0.584   │  0.287   │  0.149   │
  └──────┴──────────────────────────────────────┴──────────┴──────────┴──────────┘

  → MMR Score = λ·Rel - (1-λ)·Div
  → Latency:   ~20ms`;
   }},

  {key:'7', title:'Relevance Check', badge:'Decision Gate', color:'#10b981',
   run:()=>{
     return `🎯 RELEVANCE CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Threshold:   0.30

  Document scores:
    [1] Big Data là xu hướng công nghệ mới    → 0.708  ✓
    [2] Khái niệm Big Data và 3V              → 0.580  ✓
    [3] Ứng dụng Big Data trong kinh doanh    → 0.584  ✓

  Max score:  0.708 ≥ 0.30

  ┌─────────────────────────────────────┐
  │  DECISION:  USE RAG CONTEXT  ✅     │
  │  → Prompt with ${'`'}TÀI LIỆU THAM KHẢO${'`'} section   │
  └─────────────────────────────────────┘

  → Latency:   ~1ms`;
   }},

  {key:'8', title:'RAG LLM Generation', badge:'LLM', color:'#10b981',
   run:(q)=>{
     return `📝 RAG LLM GENERATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model:  qwen2.5:1.5b (Ollama)
  Temp:   0.3  |  top_p: 0.9  |  top_k: 40

  PROMPT (truncated):
  ┌─────────────────────────────────────────────┐
  │  Bạn là trợ lý AI tiếng Việt...             │
  │                                             │
  │  TÀI LIỆU THAM KHẢΟ:                        │
  │  [1] Big Data là thuật ngữ mô tả tập dữ...  │
  │  [2] Big Data được định nghĩa bởi 3V:...    │
  │  [3] Big Data đang được ứng dụng rộng...    │
  │                                             │
  │  CÂU HỎI: ${q.padEnd(47)}│
  └─────────────────────────────────────────────┘

  GENERATED ANSWER:
  ┌─────────────────────────────────────────────┐
  │  Big Data (dữ liệu lớn) là thuật ngữ chỉ    │
  │  các tập dữ liệu có kích thước khổng lồ...  │
  │                                             │
  │  Theo tài liệu [1], Big Data được đặc       │
  │  trưng bởi 3V: Volume (khối lượng lớn),     │
  │  Velocity (tốc độ xử lý nhanh), và Variety  │
  │  (đa dạng về định dạng) [2]...              │
  │                                             │
  │  Ứng dụng Big Data ngày càng phổ biến       │
  │  trong nhiều lĩnh vực như kinh tế, y tế,    │
  │  và giáo dục [3]...                         │
  └─────────────────────────────────────────────┘

  → Generated tokens:  ~380
  → Inference time:    ~2.8s`;
   }},

  {key:'9', title:'Redis Storage', badge:'Database', color:'#06b6d4',
   run:(q)=>{
     const sid = `demo_${Date.now()}`;
     return `💾 REDIS STORAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Key:      chat:${sid}
  Type:     Redis List (RPUSH)
  TTL:      3600s (1 hour)

  Stored messages:
  ┌──────┬──────────┬──────────────────────────────────────────┐
  │  #   │  Role    │  Content (truncated)                     │
  ├──────┼──────────┼──────────────────────────────────────────┤
  │  1   │  user    │  ${q.padEnd(40)}│
  │  2   │  assistant│  Big Data (dữ liệu lớn) là thuật ngữ...   │
  └──────┴──────────┴──────────────────────────────────────────┘

  → Auto-summary:  Kích hoạt khi >20 messages
  → LTRIM:         Giữ 11 messages gần nhất sau summary
  → Operation:     RPUSH + EXPIRE + (optional) LTRIM`;
   }},

  {key:'10', title:'NDJSON Streaming Response', badge:'Protocol', color:'#f43f5e',
   run:(q)=>{
     return `⚡ NDJSON STREAMING RESPONSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Content-Type:  application/x-ndjson
  Events:        4 types

  {"type":"token","content":"Big"}
  {"type":"token","content":" Data"}
  {"type":"token","content":" (dữ"}
  {"type":"token","content":" liệu"}
  {"type":"token","content":" lớn)"}
  {"type":"token","content":" là"}
  {"type":"token","content":" thuật"}
  {"type":"token","content":" ngữ..."}
  {"type":"token","content":"..."}
  {"type":"sources","content":[
    {"title":"Big Data là xu hướng công nghệ mới","score":0.708},
    {"title":"Khái niệm Big Data và 3V","score":0.580},
    {"title":"Ứng dụng Big Data trong kinh doanh","score":0.584}
  ]}
  {"type":"follow_up","content":[
    "Big Data khác gì dữ liệu truyền thống?",
    "Các công cụ Big Data phổ biến là gì?",
    "Học Big Data bắt đầu từ đâu?"
  ]}
  {"type":"done"}`;
   }}
];

let demoRunning = false;

function runDemo(){
  if (demoRunning) return;
  const input = document.getElementById('demoInput');
  const query = input.value.trim();
  if (!query) { input.focus(); return; }

  demoRunning = true;
  const btn = document.getElementById('btnDemo');
  const resetBtn = document.getElementById('btnDemoReset');
  const status = document.getElementById('demoStatus');
  const output = document.getElementById('demoOutput');
  const scroll = output.querySelector('.demo-output-scroll');

  btn.disabled = true;
  resetBtn.disabled = true;
  output.style.display = 'block';
  scroll.innerHTML = '';
  status.textContent = 'Running...';
  status.style.color = '#fbbf24';

  // Stop auto-play
  if (isPlaying) toggleAutoPlay();
  // Reset any existing highlight
  document.querySelectorAll('.node-rect').forEach(r=>{
    r.setAttribute('stroke-width','1.5'); r.removeAttribute('filter');
  });

  let stepIdx = 0;

  function showNextStep(){
    if (stepIdx >= DEMO_PIPELINE.length) {
      status.textContent = `✓ Complete (${DEMO_PIPELINE.length} steps)`;
      status.style.color = '#34d399';
      btn.disabled = false;
      resetBtn.disabled = false;
      demoRunning = false;
      scroll.scrollTop = scroll.scrollHeight;
      return;
    }

    const step = DEMO_PIPELINE[stepIdx];
    const content = step.run(query);

    // highlight node
    const node = document.querySelector(`.node-group[data-key="${step.key}"]`);
    if (node) {
      const rect = node.querySelector('.node-rect');
      if (rect) { rect.setAttribute('stroke-width','2.5'); rect.setAttribute('filter','url(#glow)'); }
    }

    // show task detail in side panel
    showTaskDetail(step.key);

    // add step to demo output
    const div = document.createElement('div');
    div.className = 'demo-step demo-step-enter';
    div.innerHTML = `
      <div class="ds-head">
        <span class="ds-step">STEP ${String(stepIdx+1).padStart(2,'0')}</span>
        <span class="ds-title">${step.title}</span>
        <span class="ds-badge" style="border-color:${step.color};color:${step.color}">${step.badge}</span>
      </div>
      <div class="ds-body">
        <pre class="ds-pre">${syntaxHighlight(content)}</pre>
      </div>
    `;
    scroll.appendChild(div);

    status.textContent = `Step ${stepIdx+1}/${DEMO_PIPELINE.length}: ${step.title}`;
    status.style.color = '#60a5fa';
    scroll.scrollTop = scroll.scrollHeight;

    stepIdx++;
    setTimeout(showNextStep, 1200);
  }

  setTimeout(showNextStep, 400);
}

function resetDemo(){
  const output = document.getElementById('demoOutput');
  const scroll = output.querySelector('.demo-output-scroll');
  const status = document.getElementById('demoStatus');
  const btn = document.getElementById('btnDemo');
  const resetBtn = document.getElementById('btnDemoReset');

  scroll.innerHTML = '';
  output.style.display = 'none';
  status.textContent = 'Ready';
  status.style.color = '#556688';
  btn.disabled = false;
  resetBtn.disabled = true;

  // Clear highlights
  document.querySelectorAll('.node-rect').forEach(r=>{
    r.setAttribute('stroke-width','1.5'); r.removeAttribute('filter');
  });
  document.getElementById('tpPlaceholder').style.display='flex';
  document.getElementById('tpContent').style.display='none';
  document.getElementById('stepBadge').textContent='—';
}

function syntaxHighlight(str){
  return str
    .replace(/"[^"]*"/g, m => `<span class="str">${m}</span>`)
    .replace(/\b(\d+\.\d+)\b/g, '<span class="num">$1</span>')
    .replace(/\b(true|false)\b/g, '<span class="bool">$1</span>')
    .replace(/\b(null|undefined)\b/g, '<span class="null">$1</span>')
    .replace(/→/g, '<span class="arrow">→</span>')
    .replace(/(✅|❌|DECISION|PREDICTED)/g, '<span style="color:#34d399;font-weight:700">$1</span>');
}

// ============================================================
//  MODEL TAB — Architecture diagrams & interactive demos
// ============================================================
const MODEL_COLORS = {phobert:'#8b5cf6',sbert:'#3b82f6',llm:'#10b981'};

function switchModel(name){
  document.querySelectorAll('.mt-btn').forEach(b=>b.classList.toggle('active',b.dataset.model===name));
  document.querySelectorAll('.model-panel').forEach(p=>p.classList.toggle('active',p.id==='model-'+name));
}

// ── SVG helpers ──
function mtag(name,attrs,text){
  const e=document.createElementNS('http://www.w3.org/2000/svg',name);
  if(attrs)Object.entries(attrs).forEach(([k,v])=>v!==undefined&&e.setAttribute(k,v));
  if(text!==undefined)e.textContent=text;
  return e;
}
function mrect(svg,x,y,w,h,r,fill,stroke,sw){
  svg.append(mtag('rect',{x,y,width:w,height:h,rx:r||4,fill:fill||'none',stroke:stroke||'none','stroke-width':sw||1}));
}
function mtext(svg,x,y,text,size,fill,weight,anchor){
  svg.append(mtag('text',{x,y,fill:fill||'#8899bb','font-size':size||10,'font-weight':weight||400,'text-anchor':anchor||'middle','dominant-baseline':'central'},text));
}
function marrow(svg,x1,y1,x2,y2,color,label){
  svg.append(mtag('line',{x1,y1,x2,y2,stroke:color||'#3b82f6','stroke-width':1.5,'stroke-dasharray':'4 3','marker-end':'url(#ma)'}));
  if(label) svg.append(mtag('text',{x:(x1+x2)/2,y:(y1+y2)/2-6,fill:'#556688','font-size':8,'text-anchor':'middle'},label));
}

// ── Build all architecture SVGs ──
function buildModelSVGs(){
  // Marker def
  const defs=mtag('defs');
  defs.append(mtag('marker',{id:'ma',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto'}),
    mtag('path',{d:'M0,0 L10,5 L0,10 Z',fill:'#3b82f6'}));

  // ---- PhoBERT Architecture ----
  (()=>{
    const s=document.querySelector('#phobertArch svg'); if(!s)return;
    const cx=240, colors=['#3b82f6','#60a5fa','#8b5cf6','#a78bfa','#f472b6'];
    s.append(defs.cloneNode(true));
    // Input
    mrect(s,cx-100,10,200,28,6,'#0d172a','#3b82f6'); mtext(s,cx,24,'Input: "Big Data là gì"',10,'#e0e8f5',600);
    marrow(s,cx,38,cx,55,'#3b82f6','Tokenize');
    // Token embeddings
    const tokens=['Big','Data','là','gì','[SEP]'];
    tokens.forEach((t,i)=>{
      const x=cx-120+i*60;
      mrect(s,x,58,48,28,4,'#111b30','#3b82f6',.8);
      mtext(s,x+24,72,t,9,'#60a5fa',600);
    });
    marrow(s,cx,86,cx,100,'#8b5cf6','12× Transformer');
    // 12 layers stack
    for(let i=0;i<4;i++){
      const y=103+i*12;
      mrect(s,cx-80,y,160,10,3,colors[i],colors[i],1.5);
      if(i===0) mtext(s,cx-85,y+5,'Encoder Layer 1',6,'#8899bb',400,'end');
      if(i===3) mtext(s,cx+85,y+5,'Encoder Layer 12',6,'#8899bb',400,'start');
    }
    // Pooler
    marrow(s,cx,151,cx,165,'#f472b6','[CLS] Pool');
    mrect(s,cx-60,168,120,24,6,'#1a0d28','#f472b6'); mtext(s,cx,180,'Pooled Output (768d)',9,'#f472b6',600);
    // Classifier head
    marrow(s,cx,192,cx,208,'#f472b6','Classifier');
    mrect(s,cx-70,210,140,28,6,'#2d0d28','#ec4899'); mtext(s,cx,224,'Linear(768 → 11)',9,'#fbbf24',600);
    // Output
    marrow(s,cx,238,cx,254,'#f472b6','Softmax');
    mrect(s,cx-90,256,180,32,8,'#0d2818','#10b981');
    mtext(s,cx,264,'11 Categories',9,'#34d399',600);
    mtext(s,cx,278,'thoi_su · xa_hoi · kinh_te · the_gioi · ...',7,'#556688',400);
    // Labels
    mtext(s,30,24,'INPUT',8,'#3b82f6',700,'start');
    mtext(s,30,165,'ENCODER',8,'#8b5cf6',700,'start');
    mtext(s,30,224,'HEAD',8,'#f472b6',700,'start');
    mtext(s,30,272,'OUTPUT',8,'#10b981',700,'start');
    // Decorative vertical line
    s.append(mtag('line',{x1:45,y1:14,x2:45,y2:290,stroke:'#151f36','stroke-width':1,'stroke-dasharray':'3 3'}));
  })();

  // ---- SBERT Architecture ----
  (()=>{
    const s=document.querySelector('#sbertArch svg'); if(!s)return;
    s.append(defs.cloneNode(true));
    // Twin encoders
    const cx=120, cx2=360, mid=240;
    // Sentence A
    mrect(s,cx-70,10,140,28,6,'#0d172a','#3b82f6'); mtext(s,cx,24,'Câu A',10,'#e0e8f5',600);
    mrect(s,cx-70,44,140,28,6,'#0d172a','#60a5fa'); mtext(s,cx,58,'PhoBERT Encoder',9,'#60a5fa',600);
    mrect(s,cx-50,78,100,22,4,'#111b30','#3b82f6'); mtext(s,cx,89,'Mean Pooling',8,'#3b82f6');
    mrect(s,cx-50,106,100,22,4,'#111b30','#3b82f6'); mtext(s,cx,117,'L2‑Norm → 768d',8,'#3b82f6');
    // Sentence B
    mrect(s,cx2-70,10,140,28,6,'#0d172a','#8b5cf6'); mtext(s,cx2,24,'Câu B',10,'#e0e8f5',600);
    mrect(s,cx2-70,44,140,28,6,'#0d172a','#a78bfa'); mtext(s,cx2,58,'PhoBERT Encoder',9,'#a78bfa',600);
    mrect(s,cx2-50,78,100,22,4,'#111b30','#8b5cf6'); mtext(s,cx2,89,'Mean Pooling',8,'#8b5cf6');
    mrect(s,cx2-50,106,100,22,4,'#111b30','#8b5cf6'); mtext(s,cx2,117,'L2‑Norm → 768d',8,'#8b5cf6');
    // Similarity
    marrow(s,cx,128,mid,160,'#3b82f6'); marrow(s,cx2,128,mid,160,'#8b5cf6');
    mrect(s,mid-60,162,120,28,6,'#0d2818','#10b981'); mtext(s,mid,176,'Cosine Similarity',9,'#34d399',600);
    // Result
    marrow(s,mid,190,mid,210,'#10b981');
    mrect(s,mid-80,212,160,32,8,'#080c18','#fbbf24');
    mtext(s,mid,228,'Score ∈ [−1, 1]',9,'#fbbf24',600);
    // Labels
    mtext(s,mid,252,'→ search = FAISS IndexFlatL2 (L2 distance)',8,'#556688',400);
    // Decorative
    mtext(s,10,308,'"Dữ liệu lớn là gì" ↔ "Big Data là gì" → 0.92',8,'#3a4a6a',400,'start');
  })();

  // ---- LLM Architecture ----
  (()=>{
    const s=document.querySelector('#llmArch svg'); if(!s)return;
    s.append(defs.cloneNode(true));
    const cx=240;
    // Input
    mrect(s,cx-120,10,240,28,6,'#0d172a','#10b981'); mtext(s,cx,24,'Input Tokens: [Prompt] [Context] [Query]',9,'#e0e8f5',600);
    // Embedding
    marrow(s,cx,38,cx,52,'#10b981');
    mrect(s,cx-90,54,180,22,4,'#111b30','#10b981'); mtext(s,cx,65,'Token Embedding + Positional Encoding',8,'#34d399');
    // 28 Decoder layers
    marrow(s,cx,76,cx,90,'#10b981');
    for(let i=0;i<6;i++){
      const y=92+i*14;
      const grad=i%2===0?'#0d2838':'#0d172a';
      mrect(s,cx-85,y,170,12,2,grad,i<3?'#3b82f6':'#8b5cf6',.8);
    }
    mtext(s,cx-90,100,'Decoder Layer 1',6,'#556688',400,'start');
    mtext(s,cx+90,156,'Decoder Layer 28',6,'#556688',400,'start');
    mtext(s,cx,82,'28× Decoder Layers',7,'#3a4a6a',400);
    // LM Head
    marrow(s,cx,158,cx,172,'#f472b6');
    mrect(s,cx-80,174,160,22,4,'#1a0d28','#f472b6'); mtext(s,cx,185,'LM Head (Linear + Softmax)',8,'#f472b6');
    // Output
    marrow(s,cx,196,cx,210,'#f472b6');
    mrect(s,cx-100,212,200,28,6,'#0d2818','#fbbf24'); mtext(s,cx,226,'Next-Token Prediction (Streaming)',9,'#fbbf24',600);
    // Labels
    const labels=['Auto‑regressive','1.54B params','Context: 32K','Ollama'];
    const lxs=[60,140,340,420];
    labels.forEach((l,i)=>mtext(s,lxs[i],318,l,7,'#3a4a6a',400,'start'));
    // Tokens at bottom
    ['Big',' Data',' (','dữ',' liệu',' lớn',')',' là',' ...'].forEach((t,i)=>{
      mrect(s,cx-130+i*35,286,30,18,3,'#0d172a','#10b981',.5);
      mtext(s,cx-115+i*35,295,t,7,'#34d399',600);
    });
    mtext(s,cx,272,'Generated tokens (streaming output →)',8,'#556688',400);
  })();
}

// ── Tokenization Demo ──
const SAMPLE_TOKENS = {
  'Big Data là gì':[
    {id:'5682',txt:'Big',cls:''},{id:'3251',txt:'Data',cls:''},
    {id:'875',txt:'là',cls:''},{id:'312',txt:'gì',cls:''},
    {id:'2',txt:'[SEP]',cls:'sep'}
  ],
  'Thời sự hôm nay':[
    {id:'12045',txt:'Thời',cls:''},{id:'893',txt:'sự',cls:''},
    {id:'4512',txt:'hôm',cls:''},{id:'672',txt:'nay',cls:''},
    {id:'2',txt:'[SEP]',cls:'sep'}
  ],
  'Giá xăng dầu hôm nay':[
    {id:'8921',txt:'Giá',cls:''},{id:'3456',txt:'xăng',cls:''},
    {id:'2104',txt:'dầu',cls:''},{id:'4512',txt:'hôm',cls:''},
    {id:'672',txt:'nay',cls:''},{id:'2',txt:'[SEP]',cls:'sep'}
  ]
};

function runTokenize(){
  const input=document.getElementById('tokenInput');
  const text=input.value.trim();
  if(!text)return;
  const container=document.getElementById('tokenResults');
  const vis=document.getElementById('tokenVis');
  const meta=document.getElementById('tokenMeta');
  const flow=document.querySelector('#tokenFlow svg');

  container.classList.add('visible');
  vis.innerHTML='';

  // Get or generate tokens
  let tokens=SAMPLE_TOKENS[text];
  if(!tokens){
    const words=text.split(/\s+/);
    tokens=words.map((w,i)=>({id:String(1000+i),txt:w,cls:''}));
    tokens.push({id:'2',txt:'[SEP]',cls:'sep'});
  }
  // Add [CLS] at start
  tokens=[{id:'0',txt:'[CLS]',cls:'cls'},...tokens];

  tokens.forEach(t=>{
    const chip=document.createElement('div');
    chip.className='token-chip'+(t.cls?' '+t.cls:'');
    chip.innerHTML=`<span class="tok-txt">${t.txt}</span><span class="tok-id">#${t.id}</span>`;
    chip.title=`Token ID: ${t.id}`;
    vis.appendChild(chip);
  });

  const chars=text.length;
  const tokCount=tokens.length;
  const compression=(chars/tokCount).toFixed(1);
  meta.innerHTML=`
    <span>📊 <strong>${tokCount}</strong> tokens (gồm [CLS] + [SEP])</span>
    <span>📝 <strong>${chars}</strong> ký tự</span>
    <span>⚡ Tỉ lệ nén: ~<strong>${compression}</strong> ký/token</span>
    <span>🔢 BPE vocabulary: <strong>64,000</strong></span>
  `;

  // Flow visualization
  const svg=flow;
  const fw=parseInt(svg.getAttribute('viewBox').split(' ')[2])||700;
  const fh=80;
  svg.innerHTML='';
  const defs=mtag('defs');
  defs.append(mtag('marker',{id:'mf',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto'}),
    mtag('path',{d:'M0,0 L10,5 L0,10 Z',fill:'#3b82f6'}));
  svg.append(defs);

  const steps=[
    {x:30,l:'Văn bản thô'},{x:140,l:'BPE Tokenizer'},{x:260,l:'Token IDs'},{x:380,l:'Embedding'},{x:500,l:'Transformer'}];
  steps.forEach((s,i)=>{
    mrect(svg,s.x-20,20,60,28,4,['#0d172a','#111b30','#0d172a','#0d2818','#1a0d28'][i],
      ['#3b82f6','#8b5cf6','#f472b6','#10b981','#fbbf24'][i],1);
    mtext(svg,s.x+10,34,s.l,7,'#e0e8f5',600);
    if(i<steps.length-1){
      svg.append(mtag('line',{x1:s.x+40,y1:34,x2:steps[i+1].x-20,y2:34,
        stroke:'#3b82f6','stroke-width':1,'stroke-dasharray':'3 3','marker-end':'url(#mf)'}));
    }
  });
  const bottom=['Văn bản gốc','→ BPE tokenize','→ Tra vocab','→ Lookup table','→ Self‑attention'][0];
  mtext(svg,fw/2,75,'PhoBERT: Văn bản → Token IDs → Embeddings → Contextual Representations',8,'#556688',400);
}

// ── Similarity Demo ──
function runSimilarity(){
  const a=document.getElementById('simA').value.trim();
  const b=document.getElementById('simB').value.trim();
  if(!a||!b)return;
  const container=document.getElementById('simResult');
  container.classList.add('visible');
  const svg=document.querySelector('#simGauge svg');
  svg.innerHTML='';

  // Compute mock cosine similarity based on word overlap + length
  const wa=a.toLowerCase().split(/\s+/), wb=b.toLowerCase().split(/\s+/);
  const setA=new Set(wa), setB=new Set(wb);
  const intersection=new Set([...setA].filter(x=>setB.has(x)));
  const union=new Set([...setA,...setB]);
  const jaccard=union.size>0?intersection.size/union.size:0;
  const lenSim=1-Math.abs(wa.length-wb.length)/Math.max(wa.length,wb.length,1);
  const score=Math.min(1,Math.max(0,(jaccard*0.7+lenSim*0.3)));

  const angle=45+score*180;
  const rad=(angle-90)*Math.PI/180;
  const r=75;
  const xc=100,yc=95;
  const ex=xc+r*Math.cos(rad), ey=yc+r*Math.sin(rad);

  // Arc background
  const arc=mtag('path',{d:'M25,95 A75,75 0 0,1 175,95',fill:'none',stroke:'#151f36','stroke-width':12});
  svg.append(arc);
  // Arc fill
  const largeArc=score>0.5?1:0;
  const endX=xc+r*Math.cos((225)*Math.PI/180);
  const fillArc=mtag('path',{
    d:`M${xc+r*Math.cos((45)*Math.PI/180)},${yc+r*Math.sin((45)*Math.PI/180)} A75,75 0 ${largeArc},1 ${xc+r*Math.cos((45+score*180)*Math.PI/180)},${yc+r*Math.sin((45+score*180)*Math.PI/180)}`,
    fill:'none',stroke:score>0.6?'#10b981':score>0.3?'#f59e0b':'#f87171','stroke-width':12,'stroke-linecap':'round'});
  svg.append(fillArc);
  // Needle
  svg.append(mtag('line',{x1:xc,y1:yc,x2:ex,y2:ey,stroke:'#e0e8f5','stroke-width':2}));
  svg.append(mtag('circle',{cx:xc,cy:yc,r:5,fill:'#3b82f6'}));
  // Labels
  const labels=['0','0.25','0.5','0.75','1.0'];
  const angles=[45,67.5,90,112.5,135];
  labels.forEach((l,i)=>{
    const a=angles[i]*Math.PI/180;
    const lx=xc+(r+16)*Math.cos(a), ly=yc+(r+16)*Math.sin(a);
    mtext(svg,lx,ly,l,7,'#556688',600);
  });
  mtext(svg,xc,45,'Cosine Similarity',8,'#8899bb',600);
  // Score
  mtext(svg,xc,95,score.toFixed(3),18,'#e0e8f5',700);

  // Detail
  const detail=document.getElementById('simDetail');
  const overlapWords=[...intersection].join(', ')||'(không có)';
  detail.innerHTML=`
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
      <span class="score-val">${(score*100).toFixed(1)}%</span>
      <span style="font-size:11px;color:#556688">độ tương đồng ngữ nghĩa</span>
    </div>
    <div class="metric-row"><span>Số từ chung</span><strong>${intersection.size}</strong></div>
    <div class="metric-row"><span>Từ trùng</span><span style="color:#34d399">${overlapWords}</span></div>
    <div class="metric-row"><span>Jaccard</span><strong>${(jaccard*100).toFixed(1)}%</strong></div>
    <div class="metric-row"><span>Độ dài (A:${wa.length}, B:${wb.length})</span><strong>${(lenSim*100).toFixed(0)}%</strong></div>
    <div style="margin-top:8px;font-size:10px;color:#3a4a6a">
      ${score>0.6?'✅ Ngữ nghĩa gần nhau → cùng cluster trong FAISS':
        score>0.3?'⚠️ Tương đồng một phần → có thể matching':
        '❌ Khác ngữ nghĩa → vector cách xa nhau'}
    </div>
  `;
}

// ── Prompt Builder Demo ──
function showPrompt(){
  const q=document.getElementById('promptQuery').value.trim()||'Big Data là gì';
  const mode=document.getElementById('promptMode').value;
  const container=document.getElementById('promptOutput');
  container.classList.add('visible');
  const pre=document.getElementById('promptPre');

  const ragPrompt = `<span class="section-label">## HỆ THỐNG</span>
Bạn là trợ lý AI tiếng Việt, trả lời dựa trên tài liệu tham khảo.

<span class="section-label">## QUY TẮC</span>
1. CHỈ dùng thông tin trong TÀI LIỆU THAM KHẢO để trả lời.
2. TRÍCH DẪN nguồn với [1], [2] ngay trong câu trả lời.
3. Nếu tài liệu không đủ, nói rõ "Tài liệu không đề cập đến...".

<span class="section-label">## TÀI LIỆU THAM KHẢO</span>
<span class="section-content">[1] Big Data là thuật ngữ mô tả tập dữ liệu lớn và phức tạp...
(Nguồn: vnexpress - Big Data là xu hướng công nghệ mới)

[2] Big Data được định nghĩa bởi 3V: Volume, Velocity, Variety...
(Nguồn: vietnamnet - Tìm hiểu về dữ liệu lớn)

[3] Big Data đang được ứng dụng rộng rãi trong kinh doanh...
(Nguồn: tuoitre - Big Data: Cơ hội và thách thức)</span>

<span class="section-label">## CÂU HỎI</span>
<span class="query-text">${q}</span>

<span class="section-label">## TRẢ LỜI (trích dẫn [1], [2] và kèm gợi ý):</span>`;

  const fallbackPrompt = `<span class="section-label">## HỆ THỐNG</span>
Bạn là trợ lý AI tiếng Việt thông minh, có kiến thức sâu rộng.

<span class="section-label">## QUY TẮC</span>
1. KHÔNG có tài liệu tham khảo cho câu hỏi này.
2. Hãy dùng KIẾN THỨC của bạn để trả lời.
3. Nói rõ "Tôi không tìm thấy tài liệu cụ thể..." trước khi trả lời.

<span class="meta-text">(Không có tài liệu tham khảo — score &lt; 0.3)</span>

<span class="section-label">## CÂU HỎI</span>
<span class="query-text">${q}</span>

<span class="section-label">## TRẢ LỜI (dùng kiến thức riêng):</span>`;

  pre.innerHTML=mode==='rag'?ragPrompt:fallbackPrompt;
}

// ── Init model tab ──
document.addEventListener('DOMContentLoaded', ()=>{
  buildModelSVGs();

  // Enter key handlers
  const tokenInp=document.getElementById('tokenInput');
  if(tokenInp) tokenInp.addEventListener('keydown',e=>{if(e.key==='Enter')runTokenize();});
  const simA=document.getElementById('simA'),simB=document.getElementById('simB');
  if(simA) simA.addEventListener('keydown',e=>{if(e.key==='Enter')runSimilarity();});
  if(simB) simB.addEventListener('keydown',e=>{if(e.key==='Enter')runSimilarity();});
  const pq=document.getElementById('promptQuery');
  if(pq) pq.addEventListener('keydown',e=>{if(e.key==='Enter')showPrompt();});

  // Also keep pipeline demo Enter handler
  const demoInp=document.getElementById('demoInput');
  if(demoInp) demoInp.addEventListener('keydown',e=>{if(e.key==='Enter')runDemo();});
});
