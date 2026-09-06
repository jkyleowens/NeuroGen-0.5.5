const defaults = {
  projects: [
    { name: 'Website Redesign', owner: 'Maya Chen', due: 'Apr 12', progress: 72, color: 'plum', initial: 'W' },
    { name: 'Q2 Marketing Campaign', owner: 'Tom Wilson', due: 'Apr 28', progress: 48, color: 'blue', initial: 'Q' },
    { name: 'Office Expansion', owner: 'James Kim', due: 'May 15', progress: 31, color: 'orange', initial: 'O' },
    { name: 'Mobile App v2.0', owner: 'Sarah Patel', due: 'Jun 02', progress: 18, color: 'green', initial: 'M' }
  ]
};
let state = structuredClone(defaults);
let timerSeconds = 1476;
let timerRunning = true;
let timerId;

const projectList = document.querySelector('#projectList');
function renderProjects() {
  projectList.innerHTML = state.projects.slice(0, 4).map(project => `<div class="project-row">
    <div class="project-glyph ${project.color}">${project.initial}</div><div class="project-info"><strong>${project.name}</strong><span>${project.owner} · Due <b>${project.due}</b></span></div>
    <div class="project-meta"><div><span>PROGRESS</span><strong>${project.progress}%</strong></div><div class="bar"><b class="${project.color}" style="width:${project.progress}%"></b></div></div></div>`).join('');
}

function save() { window.workbench?.save(state); }
function toast(message) { const el = document.querySelector('#toast'); el.textContent = message; el.classList.add('show'); setTimeout(() => el.classList.remove('show'), 2200); }

document.querySelectorAll('.nav-item[data-view]').forEach(button => button.addEventListener('click', () => showView(button.dataset.view)));
document.querySelectorAll('[data-view-link]').forEach(button => button.addEventListener('click', () => showView(button.dataset.viewLink)));
function showView(view) {
  document.querySelectorAll('.nav-item[data-view]').forEach(button => button.classList.toggle('active', button.dataset.view === view));
  document.querySelector('#overviewView').classList.toggle('hidden', view !== 'overview');
  const sub = document.querySelector('#subView'); sub.classList.toggle('hidden', view === 'overview');
  if (view === 'overview') return;
  const content = {
    projects: ['Projects', 'Plan, track, and deliver your team’s best work.', ['PROJECT','OWNER','DUE','PROGRESS'], state.projects.map(p => [p.name,p.owner,p.due,`<span class="badge">${p.progress}% complete</span>`])],
    inventory: ['Inventory', 'Monitor supplies, equipment, and critical stock.', ['ITEM','CATEGORY','AVAILABLE','STATUS'], [['MacBook Pro 14”','Equipment','8','In stock'],['USB-C Dock','Equipment','3','Low stock'],['A4 Copy Paper','Office supplies','24 cases','In stock'],['Ergonomic Chair','Furniture','2','Low stock']]],
    team: ['Team', 'Your people, roles, and recent contributions.', ['EMPLOYEE','POSITION','HIRED','CONTRIBUTIONS'], [['Maya Chen','Lead Product Designer','Jan 12, 2022','184'],['James Kim','Operations Manager','Jun 03, 2021','147'],['Sarah Patel','Senior Engineer','Mar 18, 2024','92'],['Tom Wilson','Marketing Lead','Sep 22, 2022','136']]],
    time: ['Time', 'Review focused hours and team capacity.', ['DAY','FOCUS TIME','MEETINGS','TOTAL'], [['Monday','6h 12m','1h 30m','7h 42m'],['Tuesday','5h 48m','2h 00m','7h 48m'],['Wednesday','4h 35m','2h 30m','7h 05m'],['Thursday','5h 51m','1h 45m','7h 36m']]]
  }[view];
  sub.innerHTML = `<div class="subview-head"><div><h1>${content[0]}</h1><p>${content[1]}</p></div><button class="primary-btn" onclick="document.querySelector('#projectDialog').showModal()">＋ Add new</button></div><table class="data-table"><thead><tr>${content[2].map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${content[3].map(row=>`<tr>${row.map(cell=>`<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

const dialog = document.querySelector('#projectDialog');
document.querySelector('#newProject').onclick = () => dialog.showModal();
document.querySelector('#quickAdd').onclick = () => dialog.showModal();
document.querySelector('#projectForm').addEventListener('submit', event => {
  const submitter = event.submitter;
  if (submitter?.value === 'cancel') return;
  event.preventDefault();
  const data = new FormData(event.target);
  const due = new Date(`${data.get('due')}T12:00:00`).toLocaleDateString('en-US',{month:'short',day:'2-digit'});
  state.projects.unshift({name:data.get('name'),owner:data.get('owner'),due,progress:0,color:data.get('color'),initial:data.get('name')[0].toUpperCase()});
  renderProjects(); save(); dialog.close(); event.target.reset(); toast('Project created successfully');
});

function drawTimer() { const m = String(Math.floor(timerSeconds/60)).padStart(2,'0'); const s = String(timerSeconds%60).padStart(2,'0'); document.querySelector('#timerText').textContent=`${m}:${s}`; }
function runTimer() { clearInterval(timerId); if(timerRunning) timerId=setInterval(()=>{timerSeconds=Math.max(0,timerSeconds-1);drawTimer();},1000); document.querySelector('#toggleTimer').textContent=timerRunning?'Ⅱ':'▶'; }
document.querySelector('#toggleTimer').onclick=()=>{timerRunning=!timerRunning;runTimer()};
document.querySelector('#resetTimer').onclick=()=>{timerSeconds=1500;drawTimer();toast('Focus timer reset')};
document.querySelector('#globalSearch').addEventListener('keydown',e=>{if(e.key==='Enter')toast(`Searching for “${e.target.value}”`)});
document.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='k'){e.preventDefault();document.querySelector('#globalSearch').focus()}});

(async function init(){ const stored=await window.workbench?.load(); if(stored?.projects) state=stored; renderProjects(); runTimer(); })();
