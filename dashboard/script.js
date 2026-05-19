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
