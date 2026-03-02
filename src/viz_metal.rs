//! SplatLens -- 3D Viewer Generator
//!
//! Generates a standalone HTML file with an embedded Canvas 2D scene
//! that visualizes the token trajectory, embedding field, and splat scars.
//!
//! Features:
//! - Software 3D perspective projection (no WebGL needed)
//! - Variable-width lines colored by steering force magnitude
//! - Step-by-step replay animation with speed control
//! - Toggle layers: field cloud, trajectory, splat scars, token neighbors
//! - Token attraction orbs: large labeled orbs showing what the model is pulled toward
//! - Decoded token labels on trajectory showing the prompt as it generates
//! - Ridge ghost trail: dashed predicted path from ridge runner
//! - Orbit (drag), zoom (scroll), pan (shift+drag)
//!
//! Fully self-contained: no CDN imports, no network requests, no servers.

use crate::viz::VizRenderData;
use std::path::Path;

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

    // Serialize per-step token texts as JS array
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

    // Serialize per-step neighbors as JS array of arrays
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
body{{background:#0a0a14;overflow:hidden;font-family:'SF Mono',Monaco,Consolas,monospace}}
canvas{{display:block;width:100vw;height:100vh}}
#hud{{
  position:fixed;top:16px;left:16px;color:#8af;font-size:12px;
  background:rgba(5,5,20,0.82);padding:14px 18px;border-radius:10px;
  border:1px solid rgba(100,160,255,0.15);max-width:420px;
  backdrop-filter:blur(12px);pointer-events:auto;z-index:10;
}}
#hud h2{{color:#adf;font-size:14px;margin-bottom:4px;letter-spacing:1px}}
#hud p{{color:#5a7a9a;font-size:11px;line-height:1.5}}
.legend{{display:flex;gap:10px;margin-top:8px;flex-wrap:wrap}}
.legend span{{display:flex;align-items:center;gap:4px;font-size:10px;color:#5a7a9a}}
.dot{{width:7px;height:7px;border-radius:50%;display:inline-block}}
#gen-text{{
  position:fixed;top:16px;right:16px;color:#adf;font-size:13px;
  background:rgba(5,5,20,0.82);padding:14px 18px;border-radius:10px;
  border:1px solid rgba(100,160,255,0.15);max-width:400px;max-height:60vh;
  overflow-y:auto;backdrop-filter:blur(12px);z-index:10;
  line-height:1.6;word-wrap:break-word;
}}
#gen-text .token-current{{color:#ffcc77;font-weight:bold;text-decoration:underline}}
#controls{{
  position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
  display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:center;
  background:rgba(5,5,20,0.85);padding:10px 18px;border-radius:12px;
  border:1px solid rgba(100,160,255,0.12);backdrop-filter:blur(12px);z-index:10;
}}
#controls button{{
  background:rgba(80,120,200,0.2);border:1px solid rgba(100,160,255,0.25);
  color:#8af;font-size:11px;padding:5px 10px;border-radius:6px;cursor:pointer;
  font-family:inherit;transition:all 0.15s;
}}
#controls button:hover{{background:rgba(80,120,200,0.35)}}
#controls button.active{{background:rgba(80,120,200,0.5);color:#fff;border-color:#6af}}
#controls button.off{{opacity:0.4;border-color:rgba(100,160,255,0.08)}}
.slider-group{{display:flex;align-items:center;gap:6px}}
.slider-group label{{color:#5a7a9a;font-size:10px;white-space:nowrap}}
.ctrl-slider{{
  width:80px;height:4px;-webkit-appearance:none;appearance:none;
  background:rgba(80,120,200,0.2);border-radius:2px;outline:none;cursor:pointer;
}}
.ctrl-slider::-webkit-slider-thumb{{
  -webkit-appearance:none;width:12px;height:12px;border-radius:50%;
  background:#6af;border:2px solid #0a0a14;cursor:pointer;
}}
#step-label{{color:#5a7a9a;font-size:11px;min-width:55px;text-align:center}}
#speed-label{{color:#5a7a9a;font-size:10px;min-width:30px;text-align:center}}
</style>
</head>
<body>
<div id="hud">
  <h2>SplatLens</h2>
  <p>"{prompt}"</p>
  <p>{traj_len} steps | {field_len} field points</p>
  <div class="legend">
    <span><span class="dot" style="background:#4af"></span>Trail</span>
    <span><span class="dot" style="background:#fe3"></span>Goal</span>
    <span><span class="dot" style="background:#1a2244"></span>Field</span>
    <span><span class="dot" style="background:#2f4"></span>Pleasure</span>
    <span><span class="dot" style="background:#f44"></span>Pain</span>
    <span><span class="dot" style="background:#ffaa44"></span>Attractors</span>
  </div>
  <p style="margin-top:6px;color:#334">Drag=orbit | Scroll=zoom | Shift+drag=pan</p>
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
  <span style="color:#222;margin:0 2px">|</span>
  <button id="btn-field" class="active">Field</button>
  <button id="btn-trail" class="active">Trail</button>
  <button id="btn-splats" class="active">Splats</button>
  <button id="btn-tokens" class="active">Tokens</button>
  <button id="btn-ridge" class="active">Ridge</button>
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

let visibleSteps=T.length,playing=false;
let replaySpeed=200;
let showField=true,showTrail=true,showSplats=true,showTokens=true,showRidge=true;

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
function toggleLayer(id,get,set){{const b=document.getElementById(id);b.addEventListener('click',()=>{{set(!get());b.classList.toggle('active');b.classList.toggle('off')}})}}
toggleLayer('btn-field',()=>showField,v=>showField=v);
toggleLayer('btn-trail',()=>showTrail,v=>showTrail=v);
toggleLayer('btn-splats',()=>showSplats,v=>showSplats=v);
toggleLayer('btn-tokens',()=>showTokens,v=>showTokens=v);
toggleLayer('btn-ridge',()=>showRidge,v=>showRidge=v);
function updateBtns(){{
  btnPlay.classList.toggle('active',playing);btnPause.classList.toggle('active',!playing);
  stepLabel.textContent=visibleSteps+'/'+T.length;scrubber.value=Math.max(0,visibleSteps-1);
}}

// Update the generated text panel
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

function drawCircle(px,py,r,color,alpha){{ctx.beginPath();ctx.arc(px,py,r,0,Math.PI*2);ctx.globalAlpha=alpha;ctx.fillStyle=color;ctx.fill()}}
function forceColor(i,steps){{const t=i/steps;const dn=(D[i]||0)/maxD;return `rgb(${{Math.floor(40+t*180+dn*35)}},${{Math.floor((1-t)*140+30)}},242)`}}
function drawGlow(x1,y1,x2,y2,col,w,a){{
  ctx.strokeStyle=col;ctx.globalAlpha=a*0.3;ctx.lineWidth=w*3;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
  ctx.globalAlpha=a*0.6;ctx.lineWidth=w*1.8;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
  ctx.globalAlpha=a;ctx.lineWidth=w;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
}}

let lastTick=0;
function frame(ts){{
  requestAnimationFrame(frame);
  if(playing&&ts-lastTick>replaySpeed){{
    lastTick=ts;visibleSteps=Math.min(visibleSteps+1,T.length);
    if(visibleSteps>=T.length)playing=false;
    updateBtns();updateGenText();
  }}
  const w=canvas.width,h=canvas.height;
  ctx.globalAlpha=1;ctx.fillStyle='#0a0a14';ctx.fillRect(0,0,w,h);
  const steps=visibleSteps;

  // ==== Field cloud ====
  if(showField){{ctx.globalAlpha=0.12;ctx.fillStyle='#1a2244';
    for(let i=0;i<fC.length;i+=3){{const p=project(fC[i][0],fC[i][1],fC[i][2]);
      if(p&&p.z>0){{ctx.beginPath();ctx.arc(p.x,p.y,Math.max(0.5,1.5*dpr/p.z),0,Math.PI*2);ctx.fill()}}
  }}}}

  // ==== Ridge ghost trail (dashed) ====
  if(showRidge&&rgC.length>1){{
    ctx.setLineDash([6*dpr,4*dpr]);ctx.lineWidth=1.5*dpr;ctx.strokeStyle='#446688';ctx.globalAlpha=0.35;
    ctx.beginPath();let started=false;
    for(let i=0;i<rgC.length;i++){{const p=project(rgC[i][0],rgC[i][1],rgC[i][2]);
      if(p){{if(!started){{ctx.moveTo(p.x,p.y);started=true}}else ctx.lineTo(p.x,p.y)}}
    }}ctx.stroke();ctx.setLineDash([]);
  }}

  // ==== Trajectory ====
  if(showTrail&&steps>1){{
    ctx.lineCap='round';ctx.lineJoin='round';
    for(let i=1;i<steps;i++){{
      const p0=project(tC[i-1][0],tC[i-1][1],tC[i-1][2]);const p1=project(tC[i][0],tC[i][1],tC[i][2]);
      if(!p0||!p1)continue;const dn=(D[i]||0)/maxD;const col=forceColor(i,steps);const bw=(1+dn*5)*dpr;
      if(dn>0.3)drawGlow(p0.x,p0.y,p1.x,p1.y,col,bw,0.5+dn*0.5);
      else{{ctx.globalAlpha=0.7;ctx.strokeStyle=col;ctx.lineWidth=bw;ctx.beginPath();ctx.moveTo(p0.x,p0.y);ctx.lineTo(p1.x,p1.y);ctx.stroke()}}
    }}
    // Trajectory dots with decoded token labels
    ctx.font=`${{Math.round(11*dpr)}}px 'SF Mono',Monaco,Consolas,monospace`;
    ctx.textBaseline='bottom';
    for(let i=0;i<steps;i++){{const p=project(tC[i][0],tC[i][1],tC[i][2]);
      if(p&&p.z>0){{
        const dn=(D[i]||0)/maxD;
        const r=Math.max(2,(3+dn*4)*dpr/p.z);
        drawCircle(p.x,p.y,r,forceColor(i,steps),0.85);
        // Show decoded token text every few steps or at current replay position
        if(i===steps-1 || (i%5===0 && TK[i])){{
          const tok=TK[i]||'';
          if(tok.trim().length>0){{
            ctx.globalAlpha=i===steps-1?0.95:0.5;
            ctx.fillStyle=i===steps-1?'#ffcc77':'#6688aa';
            ctx.fillText(tok.trim(),p.x+r+3*dpr,p.y-2*dpr);
          }}
        }}
      }}
    }}
    // Current step highlight
    if(steps>0&&steps<=tC.length){{const p=project(tC[steps-1][0],tC[steps-1][1],tC[steps-1][2]);
      if(p){{drawCircle(p.x,p.y,Math.max(6,18*dpr/p.z),'#fff',0.3);drawCircle(p.x,p.y,Math.max(4,10*dpr/p.z),'#adf',0.9)}}
    }}
  }}

  // ==== Token attraction orbs + labels (BIG) ====
  if(showTokens){{
    let targetStep=steps-1;
    while(targetStep>=0&&(!nbC[targetStep]||nbC[targetStep].length===0))targetStep--;
    if(targetStep>=0&&nbC[targetStep]){{
      const stepP=project(tC[targetStep][0],tC[targetStep][1],tC[targetStep][2]);
      const neighbors=nbC[targetStep];
      for(let j=0;j<neighbors.length;j++){{
        const nb=neighbors[j];
        const np=project(nb.p[0],nb.p[1],nb.p[2]);
        if(!np||np.z<0.01)continue;
        const sim=nb.s;
        // BIG orbs -- base radius 12-30px depending on similarity
        const orbR=Math.max(8,(12+sim*25)*dpr/np.z);
        const alpha=0.4+sim*0.55;
        const hue=30+sim*15;
        const color=`hsl(${{hue}},90%,${{50+sim*20}}%)`;
        // Large glow halo
        drawCircle(np.x,np.y,orbR*3,color,alpha*0.1);
        drawCircle(np.x,np.y,orbR*1.8,color,alpha*0.2);
        drawCircle(np.x,np.y,orbR,color,alpha);
        // Attraction line from trajectory to neighbor
        if(stepP){{
          ctx.globalAlpha=alpha*0.3;ctx.strokeStyle=color;ctx.lineWidth=1.5*dpr;
          ctx.setLineDash([4*dpr,4*dpr]);
          ctx.beginPath();ctx.moveTo(stepP.x,stepP.y);ctx.lineTo(np.x,np.y);ctx.stroke();
          ctx.setLineDash([]);
        }}
        // BIG label text
        const label=nb.t.trim();
        if(label.length>0&&label.length<25){{
          ctx.font=`bold ${{Math.round(16*dpr)}}px 'SF Mono',Monaco,Consolas,monospace`;
          ctx.globalAlpha=alpha;ctx.fillStyle='#ffcc77';
          ctx.textBaseline='middle';
          ctx.fillText(label,np.x+orbR+8*dpr,np.y);
          // Probability percentage
          ctx.font=`${{Math.round(13*dpr)}}px 'SF Mono',Monaco,Consolas,monospace`;
          ctx.globalAlpha=alpha*0.7;ctx.fillStyle='#998866';
          ctx.fillText(`${{(sim*100).toFixed(1)}}%`,np.x+orbR+8*dpr,np.y+20*dpr);
        }}
      }}
    }}
  }}

  // ==== Goal marker (pulsing) ====
  const gp=project(gC[0],gC[1],gC[2]);
  if(gp){{const pulse=0.7+0.3*Math.sin(ts*0.003);
    drawCircle(gp.x,gp.y,Math.max(8,20*dpr/gp.z),'#ffe933',0.2*pulse);
    drawCircle(gp.x,gp.y,Math.max(5,14*dpr/gp.z),'#ffe933',pulse);
  }}

  // ==== Start / End markers ====
  if(showTrail&&steps>0){{const sp=project(tC[0][0],tC[0][1],tC[0][2]);if(sp)drawCircle(sp.x,sp.y,Math.max(4,12*dpr/sp.z),'#4ffa99',1)}}
  if(showTrail&&steps>1){{const idx=Math.min(steps-1,tC.length-1);const ep=project(tC[idx][0],tC[idx][1],tC[idx][2]);if(ep)drawCircle(ep.x,ep.y,Math.max(4,12*dpr/ep.z),'#f4a',1)}}

  // ==== Splat scars ====
  if(showSplats){{for(let i=0;i<spC.length;i++){{const p=project(spC[i][0],spC[i][1],spC[i][2]);
    if(p&&p.z>0){{const isP=(SA[i]||1)>0;const col=isP?'#33ff66':'#ff3333';const r=Math.max(3,10*dpr/p.z);
      if(isP)drawCircle(p.x,p.y,r*2.5,col,0.08);drawCircle(p.x,p.y,r,col,0.7)}}
  }}}}

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
    let _ = std::process::Command::new("open").arg(&html_path).spawn();
}

// ---------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------

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
