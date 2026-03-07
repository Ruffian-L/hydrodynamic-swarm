//! SplatLens -- 3D Viewer Generator
//!
//! "Architectural Fluidity" palette -- deep space / ocean depths aesthetic.
//! Canvas 2D with software 3D projection, no WebGL needed.
//!
//! Trail: ice-blue cyan, thickness/opacity = confidence (inverse delta).
//! Jumps: white/gold marker dots at high-delta steps.
//! Attractors: soft gold/amber, subtle.  Splats: teal (pleasure) / rust (pain).
//! Token orbs: small, at real 3D positions with force-directed label avoidance.

use crate::viz::VizRenderData;
use std::path::Path;
use std::process::Command;

/// Generate an HTML 3D viewer and open it in the default browser.
pub fn launch(data: VizRenderData) {
    let html_path = Path::new("logs").join("splatlens_viewer.html");

    let trajectory_js = points_to_js(&data.trajectory_3d);
    let deltas_js = floats_to_js(&data.trajectory_deltas);
    let field_js = points_to_js(&data.field_points_3d);
    let goal_js = format!(
        "[{},{},{}]",
        data.goal_position_3d[0], data.goal_position_3d[1], data.goal_position_3d[2]
    );
    let splats_js = points_to_js(&data.splat_positions_3d);
    let splat_alphas_js = floats_to_js(&data.splat_alphas);
    let prompt_escaped = data
        .prompt
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', " ");

    let tokens_js = {
        let strs: Vec<String> = data
            .trajectory_tokens
            .iter()
            .map(|t| {
                let escaped = t
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', " ")
                    .replace('\r', "");
                format!("\"{}\"", escaped)
            })
            .collect();
        format!("[{}]", strs.join(","))
    };

    let neighbors_js = {
        let step_strs: Vec<String> = data
            .step_neighbors
            .iter()
            .map(|step_n| {
                let n_strs: Vec<String> = step_n
                    .iter()
                    .map(|n| {
                        let text_escaped = n
                            .token_text
                            .replace('\\', "\\\\")
                            .replace('"', "\\\"")
                            .replace('\n', " ")
                            .replace('\r', "");
                        format!(
                            "{{t:\"{}\",s:{:.3},p:[{:.4},{:.4},{:.4}]}}",
                            text_escaped,
                            n.probability,
                            n.position_3d[0],
                            n.position_3d[1],
                            n.position_3d[2]
                        )
                    })
                    .collect();
                format!("[{}]", n_strs.join(","))
            })
            .collect();
        format!("[{}]", step_strs.join(","))
    };

    let ridge_js = points_to_js(&data.ridge_ghost);

    let html = format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SplatLens -- Token Field Visualizer</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0F111A;overflow:hidden;font-family:'Inter','SF Pro',system-ui,sans-serif}}
canvas{{display:block;width:100vw;height:100vh}}
#hud{{
  position:fixed;top:16px;left:16px;color:#7B8CA8;font-size:12px;
  background:rgba(15,17,26,0.88);padding:14px 18px;border-radius:10px;
  border:1px solid rgba(80,100,140,0.15);max-width:420px;
  backdrop-filter:blur(14px);pointer-events:auto;z-index:10;
}}
#hud h2{{color:#A0B4D0;font-size:14px;margin-bottom:4px;letter-spacing:1.5px;font-weight:600}}
#hud p{{color:#5A6A82;font-size:11px;line-height:1.5}}
.legend{{display:flex;gap:10px;margin-top:8px;flex-wrap:wrap}}
.legend span{{display:flex;align-items:center;gap:4px;font-size:10px;color:#5A6A82}}
.dot{{width:7px;height:7px;border-radius:50%;display:inline-block}}
#gen-text{{
  position:fixed;top:16px;right:16px;color:#A0B4D0;font-size:13px;
  background:rgba(15,17,26,0.88);padding:14px 18px;border-radius:10px;
  border:1px solid rgba(80,100,140,0.15);max-width:380px;max-height:55vh;
  overflow-y:auto;backdrop-filter:blur(14px);z-index:10;
  line-height:1.7;word-wrap:break-word;font-family:'SF Mono',Monaco,Consolas,monospace;
}}
#gen-text .token-current{{color:#D4A44C;font-weight:bold}}
#controls{{
  position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
  display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:center;
  background:rgba(15,17,26,0.9);padding:10px 18px;border-radius:12px;
  border:1px solid rgba(80,100,140,0.12);backdrop-filter:blur(14px);z-index:10;
}}
#controls button{{
  background:rgba(60,80,120,0.15);border:1px solid rgba(80,100,140,0.2);
  color:#6A7A96;font-size:11px;padding:5px 10px;border-radius:6px;cursor:pointer;
  font-family:inherit;transition:all 0.15s;
}}
#controls button:hover{{background:rgba(60,80,120,0.3)}}
#controls button.active{{background:rgba(64,224,208,0.15);color:#40E0D0;border-color:rgba(64,224,208,0.3)}}
#controls button.off{{opacity:0.3;border-color:rgba(80,100,140,0.08)}}
.slider-group{{display:flex;align-items:center;gap:6px}}
.slider-group label{{color:#4A5A72;font-size:10px;white-space:nowrap}}
.ctrl-slider{{
  width:80px;height:3px;-webkit-appearance:none;appearance:none;
  background:rgba(60,80,120,0.25);border-radius:2px;outline:none;cursor:pointer;
}}
.ctrl-slider::-webkit-slider-thumb{{
  -webkit-appearance:none;width:12px;height:12px;border-radius:50%;
  background:#40E0D0;border:2px solid #0F111A;cursor:pointer;
}}
#step-label{{color:#4A5A72;font-size:11px;min-width:55px;text-align:center}}
#speed-label{{color:#4A5A72;font-size:10px;min-width:30px;text-align:center}}
#settings-btn{{
  position:fixed;bottom:70px;right:20px;width:36px;height:36px;border-radius:50%;
  background:rgba(15,17,26,0.85);border:1px solid rgba(80,100,140,0.2);
  color:#6A7A96;font-size:18px;cursor:pointer;display:flex;align-items:center;
  justify-content:center;z-index:11;backdrop-filter:blur(14px);transition:all 0.15s;
}}
#settings-btn:hover{{background:rgba(60,80,120,0.3);color:#40E0D0}}
#settings-panel{{
  position:fixed;bottom:70px;right:64px;background:rgba(15,17,26,0.94);
  border:1px solid rgba(80,100,140,0.15);border-radius:12px;padding:16px 20px;
  backdrop-filter:blur(14px);z-index:11;display:none;min-width:220px;
}}
#settings-panel.open{{display:block}}
#settings-panel h3{{color:#A0B4D0;font-size:12px;margin:0 0 10px;letter-spacing:1px;font-weight:600}}
.setting-row{{display:flex;align-items:center;gap:8px;margin-bottom:8px}}
.setting-row label{{color:#5A6A82;font-size:10px;min-width:60px}}
.setting-row .ctrl-slider{{width:90px}}
.setting-row span{{color:#4A5A72;font-size:10px;min-width:28px;text-align:right}}
.mode-btn{{
  background:rgba(60,80,120,0.15);border:1px solid rgba(80,100,140,0.2);
  color:#6A7A96;font-size:10px;padding:4px 10px;border-radius:5px;cursor:pointer;
  font-family:inherit;transition:all 0.15s;
}}
.mode-btn:hover{{background:rgba(60,80,120,0.3)}}
.mode-btn.active{{background:rgba(64,224,208,0.15);color:#40E0D0;border-color:rgba(64,224,208,0.3)}}
#prompt-bar{{
  position:fixed;top:16px;left:50%;transform:translateX(-50%);
  display:flex;align-items:center;gap:8px;z-index:12;
  background:rgba(15,17,26,0.92);padding:10px 16px;border-radius:14px;
  border:1px solid rgba(80,100,140,0.18);backdrop-filter:blur(16px);
  box-shadow:0 4px 24px rgba(0,0,0,0.3);
}}
#prompt-bar label{{color:#5A6A82;font-size:11px;white-space:nowrap;letter-spacing:0.5px}}
#prompt-input{{
  background:rgba(30,35,50,0.6);border:1px solid rgba(80,100,140,0.15);
  color:#C8D8F0;font-size:14px;padding:8px 14px;border-radius:8px;
  outline:none;width:420px;font-family:'SF Mono',Monaco,Consolas,monospace;
  transition:border-color 0.2s;
}}
#prompt-input:focus{{border-color:rgba(64,224,208,0.4)}}
#generate-btn{{
  background:linear-gradient(135deg,rgba(64,224,208,0.2),rgba(64,224,208,0.08));
  border:1px solid rgba(64,224,208,0.35);color:#40E0D0;font-size:12px;
  padding:8px 18px;border-radius:8px;cursor:pointer;font-family:inherit;
  font-weight:600;letter-spacing:0.5px;transition:all 0.2s;
}}
#generate-btn:hover{{background:linear-gradient(135deg,rgba(64,224,208,0.35),rgba(64,224,208,0.15));box-shadow:0 0 16px rgba(64,224,208,0.2)}}
</style>
</head>
<body>
<div id="prompt-bar">
  <label>PROMPT</label>
  <input type="text" id="prompt-input" value="{prompt}" spellcheck="false">
  <button id="generate-btn">Generate</button>
</div>
<div id="hud" style="top:72px">
  <h2>SPLATLENS</h2>
  <p>"{prompt}"</p>
  <p>{traj_len} steps | {field_len} field pts</p>
  <div class="legend">
    <span><span class="dot" style="background:#40E0D0"></span>Trail</span>
    <span><span class="dot" style="background:#F0E6C8"></span>Jump</span>
    <span><span class="dot" style="background:#D4A44C"></span>Goal</span>
    <span><span class="dot" style="background:#1E2840"></span>Field</span>
    <span><span class="dot" style="background:#2DD4A8"></span>Pleasure</span>
    <span><span class="dot" style="background:#8B3A3A"></span>Pain</span>
    <span><span class="dot" style="background:#C49650"></span>Attractors</span>
  </div>
  <p style="margin-top:6px;color:#2A3448">Drag=orbit | Scroll=zoom | Shift+drag=pan</p>
</div>
<div id="gen-text"></div>
<div id="controls">
  <button id="btn-play" class="active">Play</button>
  <button id="btn-pause">Pause</button>
  <div class="slider-group">
    <label>Step</label>
    <input type="range" id="scrubber" class="ctrl-slider" style="width:140px" min="0" max="{traj_max}" value="{traj_max}">
    <span id="step-label">{traj_len}/{traj_len}</span>
  </div>
  <div class="slider-group">
    <label>Speed</label>
    <input type="range" id="speed" class="ctrl-slider" min="30" max="800" value="200">
    <span id="speed-label">200ms</span>
  </div>
  <span style="color:#1A1E2A;margin:0 2px">|</span>
  <button id="btn-field" class="active">Field</button>
  <button id="btn-trail" class="active">Trail</button>
  <button id="btn-splats" class="active">Splats</button>
  <button id="btn-tokens" class="active">Tokens</button>
</div>
<button id="settings-btn">&#9881;</button>
<div id="settings-panel">
  <h3>SETTINGS</h3>
  <div class="setting-row">
    <label>Theme</label>
    <button id="btn-dark" class="mode-btn active">Dark</button>
    <button id="btn-light" class="mode-btn">Light</button>
  </div>
  <div class="setting-row">
    <label>Rotate</label>
    <button id="btn-rotate-on" class="mode-btn active">On</button>
    <button id="btn-rotate-off" class="mode-btn">Off</button>
  </div>
  <div class="setting-row">
    <label>UI Scale</label>
    <input type="range" id="ui-scale" class="ctrl-slider" min="50" max="150" value="100">
    <span id="ui-scale-val">100%</span>
  </div>
  <div class="setting-row">
    <label>Text Size</label>
    <input type="range" id="text-scale" class="ctrl-slider" min="50" max="200" value="100">
    <span id="text-scale-val">100%</span>
  </div>
  <div class="setting-row">
    <label>Particles</label>
    <input type="range" id="particle-density" class="ctrl-slider" min="0" max="300" value="80">
    <span id="particle-val">80</span>
  </div>
</div>
<canvas id="c"></canvas>
<script>
const T={trajectory_js};
const D={deltas_js};
const F={field_js};
const G={goal_js};
const SP={splats_js};
const SA={splat_alphas_js};
const NB={neighbors_js};
const RG={ridge_js};
const TK={tokens_js};

const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
const dpr=devicePixelRatio;
const genText=document.getElementById('gen-text');

function resize(){{canvas.width=innerWidth*dpr;canvas.height=innerHeight*dpr;canvas.style.width=innerWidth+'px';canvas.style.height=innerHeight+'px'}}
resize();window.addEventListener('resize',resize);

// Scene centering
let cx=0,cy=0,cz=0;
for(const p of T){{cx+=p[0];cy+=p[1];cz+=p[2]}}
const n=T.length||1;cx/=n;cy/=n;cz/=n;
let maxDist=1;
for(const p of T){{const d=Math.hypot(p[0]-cx,p[1]-cy,p[2]-cz);if(d>maxDist)maxDist=d}}
const tC=T.map(p=>[p[0]-cx,p[1]-cy,p[2]-cz]);
const fC=F.map(p=>[p[0]-cx,p[1]-cy,p[2]-cz]);
const gC=[G[0]-cx,G[1]-cy,G[2]-cz];
const spC=SP.map(p=>[p[0]-cx,p[1]-cy,p[2]-cz]);
const rgC=RG.map(p=>[p[0]-cx,p[1]-cy,p[2]-cz]);
const nbC=NB.map(stepN=>stepN.map(nb=>({{t:nb.t,s:nb.s,p:[nb.p[0]-cx,nb.p[1]-cy,nb.p[2]-cz]}})));
const maxD=Math.max(...D,1);

// Detect jump threshold (top 15% of deltas = jumps)
const sortedD=[...D].sort((a,b)=>b-a);
const jumpThreshold=sortedD[Math.floor(sortedD.length*0.15)]||maxD*0.7;

// Camera
let camTheta=0.5,camPhi=0.4,camDist=maxDist*2.5;
let panX=0,panY=0;
let dragging=false,panning=false,lastX=0,lastY=0;
canvas.addEventListener('mousedown',e=>{{dragging=true;panning=e.shiftKey;lastX=e.clientX;lastY=e.clientY}});
window.addEventListener('mouseup',()=>{{dragging=false;panning=false}});
window.addEventListener('mousemove',e=>{{
  if(!dragging)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;
  if(panning){{panX-=dx*camDist*0.001;panY+=dy*camDist*0.001}}
  else{{camTheta+=dx*0.005;camPhi=Math.max(-1.4,Math.min(1.4,camPhi+dy*0.005))}}
}});
canvas.addEventListener('wheel',e=>{{e.preventDefault();camDist=Math.max(0.5,camDist*(1+e.deltaY*0.001))}},{{passive:false}});

// State
let visibleSteps=T.length,playing=false,replaySpeed=200;
let showField=true,showTrail=true,showSplats=true,showTokens=true;
let isDark=true,uiScale=1,textScale=1,particleCount=80,autoRotate=true;
const darkBg='#0F111A',lightBg='#E0E4EA';
const darkHalo='#0F111A',lightHalo='#E0E4EA';

// Ambient particles
let particles=[];
function initParticles(count){{
  particles=[];
  for(let i=0;i<count;i++){{
    particles.push({{x:(Math.random()-0.5)*maxDist*4,y:(Math.random()-0.5)*maxDist*4,z:(Math.random()-0.5)*maxDist*4,
      vx:(Math.random()-0.5)*0.003,vy:(Math.random()-0.5)*0.003,vz:(Math.random()-0.5)*0.003,
      r:1+Math.random()*3,hue:170+Math.random()*50}});
  }}
}}
initParticles(particleCount);

const scrubber=document.getElementById('scrubber');
const stepLabel=document.getElementById('step-label');
const speedSlider=document.getElementById('speed');
const speedLabel=document.getElementById('speed-label');
const btnPlay=document.getElementById('btn-play');
const btnPause=document.getElementById('btn-pause');
scrubber.addEventListener('input',()=>{{visibleSteps=parseInt(scrubber.value)+1;playing=false;updateBtns();updateGenText()}});
speedSlider.addEventListener('input',()=>{{replaySpeed=parseInt(speedSlider.value);speedLabel.textContent=replaySpeed+'ms'}});
btnPlay.addEventListener('click',()=>{{playing=true;visibleSteps=1;updateBtns();updateGenText()}});
btnPause.addEventListener('click',()=>{{playing=false;visibleSteps=T.length;scrubber.value=T.length-1;updateBtns();updateGenText()}});
// Prompt bar -- Generate triggers Play replay
const promptInput=document.getElementById('prompt-input');
const generateBtn=document.getElementById('generate-btn');
function triggerGenerate(){{
  // Reset and replay the pre-baked generation
  visibleSteps=1;playing=true;genText.innerHTML='';updateBtns();updateGenText();
  // Flash the button
  generateBtn.style.boxShadow='0 0 20px rgba(64,224,208,0.5)';
  setTimeout(()=>generateBtn.style.boxShadow='',400);
}}
generateBtn.addEventListener('click',triggerGenerate);
promptInput.addEventListener('keydown',e=>{{if(e.key==='Enter')triggerGenerate()}});
function toggleLayer(id,get,set){{const b=document.getElementById(id);b.addEventListener('click',()=>{{set(!get());b.classList.toggle('active');b.classList.toggle('off')}})}}
toggleLayer('btn-field',()=>showField,v=>showField=v);
toggleLayer('btn-trail',()=>showTrail,v=>showTrail=v);
toggleLayer('btn-splats',()=>showSplats,v=>showSplats=v);
toggleLayer('btn-tokens',()=>showTokens,v=>showTokens=v);
function updateBtns(){{
  btnPlay.classList.toggle('active',playing);btnPause.classList.toggle('active',!playing);
  stepLabel.textContent=visibleSteps+'/'+T.length;scrubber.value=Math.max(0,visibleSteps-1);
}}
function updateGenText(){{
  let html='';
  for(let i=0;i<Math.min(visibleSteps,TK.length);i++){{
    const tok=TK[i].replace(/</g,'&lt;').replace(/>/g,'&gt;');
    if(i===visibleSteps-1)html+=`<span class="token-current">${{tok}}</span>`;
    else html+=tok;
  }}
  genText.innerHTML=html||'...';
  genText.scrollTop=genText.scrollHeight;
}}
updateGenText();

// Settings panel
const settingsBtn=document.getElementById('settings-btn');
const settingsPanel=document.getElementById('settings-panel');
settingsBtn.addEventListener('click',()=>settingsPanel.classList.toggle('open'));
document.getElementById('btn-dark').addEventListener('click',()=>{{isDark=true;document.getElementById('btn-dark').classList.add('active');document.getElementById('btn-light').classList.remove('active');
  document.body.style.background=darkBg;document.getElementById('hud').style.background='rgba(15,17,26,0.88)';document.getElementById('gen-text').style.background='rgba(15,17,26,0.88)';
  document.getElementById('hud').style.color='#7B8CA8';document.getElementById('gen-text').style.color='#A0B4D0';
  settingsPanel.style.background='rgba(15,17,26,0.94)';
}});
document.getElementById('btn-light').addEventListener('click',()=>{{isDark=false;document.getElementById('btn-light').classList.add('active');document.getElementById('btn-dark').classList.remove('active');
  document.body.style.background=lightBg;document.getElementById('hud').style.background='rgba(240,242,248,0.92)';document.getElementById('gen-text').style.background='rgba(240,242,248,0.92)';
  document.getElementById('hud').style.color='#3A4A5A';document.getElementById('gen-text').style.color='#2A3A4A';
  settingsPanel.style.background='rgba(240,242,248,0.94)';
}});
document.getElementById('ui-scale').addEventListener('input',e=>{{uiScale=parseInt(e.target.value)/100;document.getElementById('ui-scale-val').textContent=e.target.value+'%';
  document.getElementById('hud').style.transform=`scale(${{uiScale}})`;document.getElementById('hud').style.transformOrigin='top left';
  document.getElementById('controls').style.transform=`translateX(-50%) scale(${{uiScale}})`;
  document.getElementById('gen-text').style.transform=`scale(${{uiScale}})`;document.getElementById('gen-text').style.transformOrigin='top right';
}});
document.getElementById('text-scale').addEventListener('input',e=>{{textScale=parseInt(e.target.value)/100;document.getElementById('text-scale-val').textContent=e.target.value+'%'}});
document.getElementById('btn-rotate-on').addEventListener('click',()=>{{autoRotate=true;document.getElementById('btn-rotate-on').classList.add('active');document.getElementById('btn-rotate-off').classList.remove('active')}});
document.getElementById('btn-rotate-off').addEventListener('click',()=>{{autoRotate=false;document.getElementById('btn-rotate-off').classList.add('active');document.getElementById('btn-rotate-on').classList.remove('active')}});
document.getElementById('particle-density').addEventListener('input',e=>{{particleCount=parseInt(e.target.value);document.getElementById('particle-val').textContent=e.target.value;initParticles(particleCount)}});

// Projection
function project(x,y,z){{
  const ex=camDist*Math.cos(camPhi)*Math.cos(camTheta)+panX;
  const ey=camDist*Math.sin(camPhi)+panY;
  const ez=camDist*Math.cos(camPhi)*Math.sin(camTheta);
  const fx=panX-ex,fy=panY-ey,fz=-ez;
  const fl=Math.hypot(fx,fy,fz);const fdx=fx/fl,fdy=fy/fl,fdz=fz/fl;
  const rx=fdy*0-fdz*1,ry=fdz*0-fdx*0,rz=fdx*1-fdy*0;
  const rl=Math.hypot(rx,ry,rz)||1;const rdx=rx/rl,rdy=ry/rl,rdz=rz/rl;
  const udx=rdy*fdz-rdz*fdy,udy=rdz*fdx-rdx*fdz,udz=rdx*fdy-rdy*fdx;
  const dx=x-ex,dy=y-ey,dz=z-ez;
  const vx=rdx*dx+rdy*dy+rdz*dz;const vy=udx*dx+udy*dy+udz*dz;const vz=fdx*dx+fdy*dy+fdz*dz;
  if(vz<0.01)return null;
  const fov=1.2,w=canvas.width,h=canvas.height,asp=w/h;
  return {{x:w/2+(vx/vz)*w/(fov*asp),y:h/2-(vy/vz)*w/fov,z:vz}};
}}

function drawCircle(px,py,r,color,alpha){{
  ctx.beginPath();ctx.arc(px,py,r,0,Math.PI*2);ctx.globalAlpha=alpha;ctx.fillStyle=color;ctx.fill();
}}
function drawPill(x,y,w,h,r){{
  ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.arcTo(x+w,y,x+w,y+r,r);
  ctx.lineTo(x+w,y+h-r);ctx.arcTo(x+w,y+h,x+w-r,y+h,r);
  ctx.lineTo(x+r,y+h);ctx.arcTo(x,y+h,x,y+h-r,r);ctx.lineTo(x,y+r);ctx.arcTo(x,y,x+r,y,r);ctx.closePath();
}}
// Text with dark halo outline for readability
function haloText(text,x,y,fillColor,alpha){{
  ctx.globalAlpha=1;ctx.lineWidth=3*dpr;ctx.strokeStyle=isDark?'#0F111A':'#E0E4EA';ctx.lineJoin='round';
  ctx.strokeText(text,x,y);ctx.fillStyle=fillColor;ctx.fillText(text,x,y);
}}

// Z-depth fog: dims alpha for far objects, enhances near objects
function fog(z,alpha){{const near=camDist*0.3,far=camDist*4;const t=Math.max(0,Math.min(1,(z-near)/(far-near)));return alpha*(1-t*0.85);}}

// ---- RENDER ----
let lastTick=0;
function frame(ts){{
  requestAnimationFrame(frame);
  // Subtle ambient camera breathing
  if(!dragging&&autoRotate)camTheta+=0.0003;
  if(playing&&ts-lastTick>replaySpeed){{
    lastTick=ts;visibleSteps=Math.min(visibleSteps+1,T.length);
    if(visibleSteps>=T.length)playing=false;
    updateBtns();updateGenText();
  }}
  const w=canvas.width,h=canvas.height;
  const bgColor=isDark?darkBg:lightBg;
  const haloColor=isDark?darkHalo:lightHalo;
  ctx.globalAlpha=1;ctx.fillStyle=bgColor;ctx.fillRect(0,0,w,h);
  const steps=visibleSteps;
  const ts_f=textScale; // text scale factor

  // ==== Ambient floating particles ====
  for(const pt of particles){{
    pt.x+=pt.vx;pt.y+=pt.vy;pt.z+=pt.vz;
    // Wrap around
    const bound=maxDist*3;
    if(pt.x>bound)pt.x=-bound;if(pt.x<-bound)pt.x=bound;
    if(pt.y>bound)pt.y=-bound;if(pt.y<-bound)pt.y=bound;
    if(pt.z>bound)pt.z=-bound;if(pt.z<-bound)pt.z=bound;
    const pp=project(pt.x,pt.y,pt.z);
    if(pp&&pp.z>0){{
      const pa=fog(pp.z,isDark?0.95:0.85);
      const col=isDark?`hsl(${{pt.hue}},55%,55%)`:`hsl(${{pt.hue}},30%,60%)`;
      ctx.globalAlpha=pa;ctx.fillStyle=col;
      const pr=Math.max(5,pt.r*10*dpr/pp.z);
      ctx.beginPath();ctx.arc(pp.x,pp.y,pr,0,Math.PI*2);ctx.fill();
    }}
  }}

  // ==== Field cloud (drifting dust motes) ====
  if(showField){{
    const drift=ts*0.0001;
    for(let i=0;i<fC.length;i+=3){{
      // Slow sinusoidal drift per particle
      const sx=fC[i][0]+Math.sin(drift+i*0.7)*0.02;
      const sy=fC[i][1]+Math.cos(drift+i*1.1)*0.02;
      const sz=fC[i][2]+Math.sin(drift+i*0.3)*0.015;
      const p=project(sx,sy,sz);
      if(p&&p.z>0){{
        const fa=fog(p.z,0.1);
        ctx.globalAlpha=fa;ctx.fillStyle='#1E2840';
        ctx.beginPath();ctx.arc(p.x,p.y,Math.max(0.5,1.2*dpr/p.z),0,Math.PI*2);ctx.fill();
      }}
    }}
  }}

  // ==== Ridge ghost trail ====
  if(rgC.length>1){{
    ctx.setLineDash([5*dpr,4*dpr]);ctx.lineWidth=1*dpr;ctx.strokeStyle='#2A3A52';ctx.globalAlpha=0.25;
    ctx.beginPath();let started=false;
    for(let i=0;i<rgC.length;i++){{const p=project(rgC[i][0],rgC[i][1],rgC[i][2]);
      if(p){{if(!started){{ctx.moveTo(p.x,p.y);started=true}}else ctx.lineTo(p.x,p.y)}}
    }}ctx.stroke();ctx.setLineDash([]);
  }}

  // ==== Trail: ice-blue, energy-based width/opacity ====
  if(showTrail&&steps>1){{
    ctx.lineCap='round';ctx.lineJoin='round';
    for(let i=1;i<steps;i++){{
      const p0=project(tC[i-1][0],tC[i-1][1],tC[i-1][2]);
      const p1=project(tC[i][0],tC[i][1],tC[i][2]);
      if(!p0||!p1)continue;
      const dn=(D[i]||0)/maxD;
      // Confidence = inverse of delta. Low delta = confident = bright/thick
      const confidence=1-dn;
      const lineW=(0.8+confidence*3)*dpr;
      const alpha=fog(p1.z,0.25+confidence*0.5);
      ctx.globalAlpha=alpha;
      ctx.strokeStyle='#40E0D0';
      ctx.lineWidth=lineW;
      ctx.beginPath();ctx.moveTo(p0.x,p0.y);ctx.lineTo(p1.x,p1.y);ctx.stroke();
    }}

    // Trail dots -- small, subtle, fog-faded
    for(let i=0;i<steps;i++){{const p=project(tC[i][0],tC[i][1],tC[i][2]);
      if(p&&p.z>0){{
        const dn=(D[i]||0)/maxD;
        const confidence=1-dn;
        const r=Math.max(1.5,(1.5+confidence*2)*dpr/p.z);
        drawCircle(p.x,p.y,r,'#40E0D0',fog(p.z,0.3+confidence*0.5));
      }}
    }}

    // ==== Jump markers (high-delta steps) ====
    for(let i=0;i<steps;i++){{
      if((D[i]||0)>=jumpThreshold){{
        const p=project(tC[i][0],tC[i][1],tC[i][2]);
        if(p&&p.z>0){{
          const r=Math.max(4,8*dpr/p.z);
          // Soft glow
          drawCircle(p.x,p.y,r*2.5,'#F0E6C8',0.1);
          drawCircle(p.x,p.y,r*1.5,'#F0E6C8',0.2);
          // Solid marker
          drawCircle(p.x,p.y,r,'#F0E6C8',0.85);
        }}
      }}
    }}

    // Decoded token labels on trail (every 5 steps)
    for(let i=0;i<steps;i++){{
      if(i===steps-1 || (i%5===0 && TK[i])){{
        const p=project(tC[i][0],tC[i][1],tC[i][2]);
        if(p&&p.z>0){{
          const tok=(TK[i]||'').trim();
          if(tok.length>0){{
            const fs=Math.round(13*dpr*ts_f);
            ctx.font=`600 ${{fs}}px 'Inter','SF Pro',system-ui,sans-serif`;
            ctx.textBaseline='bottom';
            // Current = bright white, past = muted dark gray (not transparent)
            const col=i===steps-1?'#FFFFFF':'#3A4A5A';
            haloText(tok,p.x+8*dpr,p.y-3*dpr,col,1);
          }}
        }}
      }}
    }}

    // Current step highlight
    if(steps>0&&steps<=tC.length){{const p=project(tC[steps-1][0],tC[steps-1][1],tC[steps-1][2]);
      if(p){{
        drawCircle(p.x,p.y,Math.max(5,14*dpr/p.z),'#40E0D0',0.15);
        drawCircle(p.x,p.y,Math.max(3,8*dpr/p.z),'#40E0D0',0.7);
      }}
    }}
  }}

  // ==== Goal attractor (soft gold, subtle) ====
  const gp=project(gC[0],gC[1],gC[2]);
  if(gp){{
    drawCircle(gp.x,gp.y,Math.max(6,14*dpr/gp.z),'#D4A44C',0.12);
    drawCircle(gp.x,gp.y,Math.max(4,10*dpr/gp.z),'#D4A44C',0.6);
    // Label
    const fs=Math.round(10*dpr);
    ctx.font=`600 ${{fs}}px 'Inter','SF Pro',system-ui,sans-serif`;
    ctx.globalAlpha=0.5;ctx.fillStyle='#D4A44C';ctx.textBaseline='middle';
    ctx.fillText('GOAL',gp.x+12*dpr,gp.y);
  }}

  // ==== Start / End markers ====
  if(showTrail&&steps>0){{
    const sp=project(tC[0][0],tC[0][1],tC[0][2]);
    if(sp)drawCircle(sp.x,sp.y,Math.max(3,8*dpr/sp.z),'#40E0D0',0.8);
  }}
  if(showTrail&&steps>1){{
    const idx=Math.min(steps-1,tC.length-1);
    const ep=project(tC[idx][0],tC[idx][1],tC[idx][2]);
    if(ep)drawCircle(ep.x,ep.y,Math.max(3,8*dpr/ep.z),'#D4A44C',0.8);
  }}

  // ==== Splat scars (muted teal / rust) ====
  if(showSplats){{
    for(let i=0;i<spC.length;i++){{const p=project(spC[i][0],spC[i][1],spC[i][2]);
      if(p&&p.z>0){{
        const isPleasure=(SA[i]||1)>0;
        const col=isPleasure?'#2DD4A8':'#8B3A3A';
        const r=Math.max(2,6*dpr/p.z);
        if(isPleasure)drawCircle(p.x,p.y,r*2,col,0.05);
        drawCircle(p.x,p.y,r,col,0.45);
      }}
    }}
  }}

  // ==== Ephemeral token attraction orbs (screen-space fan, fixed pixel radius) ====
  if(showTokens){{
    const fadeWindow=5;
    for(let si=Math.max(0,steps-fadeWindow);si<steps;si++){{
      if(!nbC[si]||nbC[si].length===0)continue;
      const age=steps-1-si;
      const fade=Math.max(0,1-age/fadeWindow);
      const stepP=project(tC[si][0],tC[si][1],tC[si][2]);
      if(!stepP)continue;
      const neighbors=nbC[si].filter(nb=>nb.s>=0.001);
      const count=neighbors.length;
      if(!count)continue;
      const fanR=110*dpr; // fixed pixel radius -- zoom independent
      const startAngle=-Math.PI*0.8;
      const sweep=Math.PI*1.6;
      for(let j=0;j<count;j++){{
        const nb=neighbors[j];const prob=nb.s;
        // Fan position in screen space around trajectory point
        const angle=startAngle+(j/(Math.max(count-1,1)))*sweep;
        const ox=stepP.x+Math.cos(angle)*fanR*(0.7+prob*0.3);
        const oy=stepP.y+Math.sin(angle)*fanR*(0.7+prob*0.3);
        // Orb
        const orbR=Math.max(4,(5+prob*10)*dpr);
        const color=`hsl(35,65%,${{45+prob*15}}%)`;
        drawCircle(ox,oy,orbR*1.5,color,0.08*fade);
        drawCircle(ox,oy,orbR,color,0.6*fade);
        // Connection line (current step only)
        if(age===0){{
          ctx.globalAlpha=0.2;ctx.strokeStyle=color;ctx.lineWidth=1*dpr;
          ctx.setLineDash([3*dpr,3*dpr]);
          ctx.beginPath();ctx.moveTo(stepP.x,stepP.y);ctx.lineTo(ox,oy);ctx.stroke();
          ctx.setLineDash([]);
        }}
        // Label with pill background + halo text
        if(fade>0.3){{
          const label=nb.t.trim();
          if(label.length>0&&label.length<25){{
            const fs=Math.round(14*dpr*ts_f);
            ctx.font=`700 ${{fs}}px 'Inter','SF Pro',system-ui,sans-serif`;
            ctx.textBaseline='middle';
            const probText=`${{(prob*100).toFixed(1)}}%`;
            const fullText=label+'  '+probText;
            const tw=ctx.measureText(fullText).width;
            const lx=ox+orbR+5*dpr;
            // Background pill
            ctx.globalAlpha=0.85*fade;ctx.fillStyle='rgba(15,17,26,0.92)';
            drawPill(lx-4*dpr,oy-fs/2-3*dpr,tw+10*dpr,fs+6*dpr,4*dpr);ctx.fill();
            // Token name -- bright
            haloText(label,lx,oy,'#E0D0A0',fade);
            // Probability
            const labelW=ctx.measureText(label+'  ').width;
            haloText(probText,lx+labelW,oy,'#8A7A5A',0.7*fade);
          }}
        }}
      }}
    }}
  }}

  ctx.globalAlpha=1;
}}
requestAnimationFrame(frame);
</script>
</body>
</html>"##,
        prompt = prompt_escaped,
        traj_len = data.trajectory_3d.len(),
        traj_max = data.trajectory_3d.len().saturating_sub(1),
        field_len = data.field_points_3d.len(),
        trajectory_js = trajectory_js,
        deltas_js = deltas_js,
        field_js = field_js,
        goal_js = goal_js,
        splats_js = splats_js,
        splat_alphas_js = splat_alphas_js,
        neighbors_js = neighbors_js,
        ridge_js = ridge_js,
        tokens_js = tokens_js,
    );

    if let Err(e) = std::fs::write(&html_path, &html) {
        eprintln!("    [VIZ] Failed to write viewer: {}", e);
        return;
    }

    println!("    [VIZ] 3D viewer: {}", html_path.display());
    try_open_viewer(&html_path);
}

// ---------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------

fn try_open_viewer(path: &Path) {
    let mut last_error = None;
    for (program, args) in viewer_open_commands(path) {
        match Command::new(program).args(&args).spawn() {
            Ok(_) => return,
            Err(err) => last_error = Some(format!("{}: {}", program, err)),
        }
    }

    if let Some(err) = last_error {
        eprintln!("    [VIZ] Viewer auto-open unavailable: {}", err);
        eprintln!("    [VIZ] Open this file manually: {}", path.display());
    }
}

fn viewer_open_commands(path: &Path) -> Vec<(&'static str, Vec<String>)> {
    let resolved = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .into_owned();

    #[cfg(target_os = "macos")]
    {
        vec![("open", vec![resolved])]
    }

    #[cfg(target_os = "linux")]
    {
        vec![
            ("xdg-open", vec![resolved.clone()]),
            ("gio", vec!["open".to_string(), resolved]),
        ]
    }

    #[cfg(target_os = "windows")]
    {
        vec![(
            "cmd",
            vec![
                "/C".to_string(),
                "start".to_string(),
                "".to_string(),
                resolved,
            ],
        )]
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        Vec::new()
    }
}

fn points_to_js(points: &[[f32; 3]]) -> String {
    let entries: Vec<String> = points
        .iter()
        .map(|p| format!("[{:.4},{:.4},{:.4}]", p[0], p[1], p[2]))
        .collect();
    format!("[{}]", entries.join(","))
}

fn floats_to_js(vals: &[f32]) -> String {
    let entries: Vec<String> = vals.iter().map(|v| format!("{:.4}", v)).collect();
    format!("[{}]", entries.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewer_open_commands_have_platform_candidates() {
        let commands = viewer_open_commands(Path::new("logs/splatlens_viewer.html"));

        #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
        assert!(
            !commands.is_empty(),
            "viewer auto-open should have at least one platform candidate"
        );

        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        assert!(
            commands.is_empty(),
            "unsupported platforms should return empty command list"
        );

        #[cfg(target_os = "linux")]
        assert_eq!(commands[0].0, "xdg-open");

        #[cfg(target_os = "macos")]
        assert_eq!(commands[0].0, "open");

        #[cfg(target_os = "windows")]
        assert_eq!(commands[0].0, "cmd");
    }
}
