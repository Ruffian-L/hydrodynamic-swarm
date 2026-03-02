//! SplatLens -- 3D Viewer Generator
//!
//! Generates a standalone HTML file with an embedded WebGL 3D scene
//! that visualizes the token trajectory, embedding field, and splat scars.
//!
//! Fully self-contained: no CDN imports, no network requests, no servers.
//! Opens as a local file in the default browser.
//! On macOS, the system compositor uses Metal for hardware-accelerated WebGL.

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
    let prompt_escaped = data.prompt.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " ");

    let html = format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SplatLens -- Token Field Visualizer</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a14;overflow:hidden;font-family:'SF Mono',Monaco,monospace}}
canvas{{display:block;width:100vw;height:100vh}}
#hud{{
  position:fixed;top:16px;left:16px;color:#8af;font-size:12px;
  background:rgba(5,5,20,0.75);padding:14px 18px;border-radius:10px;
  border:1px solid rgba(100,160,255,0.15);max-width:380px;
  backdrop-filter:blur(10px);pointer-events:none;
}}
#hud h2{{color:#adf;font-size:14px;margin-bottom:4px;letter-spacing:1px}}
#hud p{{color:#5a7a9a;font-size:11px;line-height:1.5}}
.legend{{display:flex;gap:10px;margin-top:8px;flex-wrap:wrap}}
.legend span{{display:flex;align-items:center;gap:4px;font-size:10px;color:#5a7a9a}}
.dot{{width:7px;height:7px;border-radius:50%;display:inline-block}}
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
    <span><span class="dot" style="background:#4fa"></span>Start</span>
    <span><span class="dot" style="background:#f4a"></span>End</span>
  </div>
  <p style="margin-top:6px;color:#334">Drag=orbit | Scroll=zoom | Shift+drag=pan</p>
</div>
<canvas id="c"></canvas>
<script>
// ---- DATA (embedded by Rust, no network) ----
const T={trajectory_js};
const D={deltas_js};
const F={field_js};
const G={goal_js};
const SP={splats_js};
const SA={splat_alphas_js};

// ---- WebGL setup ----
const canvas=document.getElementById('c');
const gl=canvas.getContext('webgl',{{antialias:true,alpha:false}});
if(!gl){{document.body.innerHTML='<h1 style="color:red;padding:40px">WebGL not supported</h1>';throw'no gl'}}

function resize(){{
  canvas.width=innerWidth*devicePixelRatio;
  canvas.height=innerHeight*devicePixelRatio;
  canvas.style.width=innerWidth+'px';
  canvas.style.height=innerHeight+'px';
  gl.viewport(0,0,canvas.width,canvas.height);
}}
resize();
window.addEventListener('resize',resize);

// ---- Shaders ----
const VS=`attribute vec3 a_pos;attribute vec4 a_col;uniform mat4 u_mvp;uniform float u_psz;varying vec4 v_col;
void main(){{gl_Position=u_mvp*vec4(a_pos,1.0);gl_PointSize=u_psz;v_col=a_col;}}`;
const FS=`precision mediump float;varying vec4 v_col;
void main(){{float d=length(gl_PointCoord-vec2(0.5));if(d>0.5)discard;gl_FragColor=v_col*smoothstep(0.5,0.3,d);}}`;
const VS_LINE=`attribute vec3 a_pos;attribute vec4 a_col;uniform mat4 u_mvp;varying vec4 v_col;
void main(){{gl_Position=u_mvp*vec4(a_pos,1.0);v_col=a_col;}}`;
const FS_LINE=`precision mediump float;varying vec4 v_col;void main(){{gl_FragColor=v_col;}}`;

function mkShader(src,type){{const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);return s}}
function mkProg(vs,fs){{
  const p=gl.createProgram();
  gl.attachShader(p,mkShader(vs,gl.VERTEX_SHADER));
  gl.attachShader(p,mkShader(fs,gl.FRAGMENT_SHADER));
  gl.linkProgram(p);return p;
}}

const progPt=mkProg(VS,FS);
const progLn=mkProg(VS_LINE,FS_LINE);

// ---- Compute scene center + extent from trajectory ----
let cx=0,cy=0,cz=0;
for(const p of T){{cx+=p[0];cy+=p[1];cz+=p[2]}}
const n=T.length||1;cx/=n;cy/=n;cz/=n;
let maxDist=1;
for(const p of T){{const d=Math.hypot(p[0]-cx,p[1]-cy,p[2]-cz);if(d>maxDist)maxDist=d}}

// ---- Build vertex data ----
function packVerts(positions,colors){{
  const buf=new Float32Array(positions.length*7);
  for(let i=0;i<positions.length;i++){{
    buf[i*7]=positions[i][0];buf[i*7+1]=positions[i][1];buf[i*7+2]=positions[i][2];
    buf[i*7+3]=colors[i][0];buf[i*7+4]=colors[i][1];buf[i*7+5]=colors[i][2];buf[i*7+6]=colors[i][3];
  }}
  return buf;
}}

// Field points
const fPos=F.map(p=>[p[0]-cx,p[1]-cy,p[2]-cz]);
const fCol=F.map(()=>[0.1,0.13,0.27,0.2]);
const fBuf=packVerts(fPos,fCol);const fCount=fPos.length;

// Trajectory points + line
const maxD=Math.max(...D,1);
const tPos=T.map(p=>[p[0]-cx,p[1]-cy,p[2]-cz]);
const tCol=T.map((_,i)=>{{
  const t=i/T.length;const dn=(D[i]||0)/maxD;
  return[0.15+t*0.7+dn*0.15, (1-t)*0.6+0.1, 0.95, 0.9];
}});
const tBuf=packVerts(tPos,tCol);const tCount=tPos.length;

// Special points: goal, start, end, splats
const specPos=[];const specCol=[];
// Goal
specPos.push([G[0]-cx,G[1]-cy,G[2]-cz]);specCol.push([1,0.93,0.2,1]);
// Start
if(T.length>0){{specPos.push(tPos[0]);specCol.push([0.3,1,0.7,1])}}
// End
if(T.length>1){{specPos.push(tPos[tPos.length-1]);specCol.push([1,0.3,0.7,1])}}
// Splats
for(let i=0;i<SP.length;i++){{
  specPos.push([SP[i][0]-cx,SP[i][1]-cy,SP[i][2]-cz]);
  specCol.push((SA[i]||1)>0?[0.2,0.9,0.3,0.7]:[0.9,0.2,0.2,0.7]);
}}
const sBuf=packVerts(specPos,specCol);const sCount=specPos.length;

// Create GL buffers
function mkGLBuf(data){{const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);return b}}
const glField=mkGLBuf(fBuf);
const glTrail=mkGLBuf(tBuf);
const glSpec=mkGLBuf(sBuf);

// ---- Camera (orbit) ----
let camTheta=0.5,camPhi=0.4,camDist=maxDist*2.5;
let panX=0,panY=0;
let dragging=false,panning=false,lastX=0,lastY=0;

canvas.addEventListener('mousedown',e=>{{
  dragging=true;panning=e.shiftKey;lastX=e.clientX;lastY=e.clientY;
}});
window.addEventListener('mouseup',()=>{{dragging=false;panning=false}});
window.addEventListener('mousemove',e=>{{
  if(!dragging)return;
  const dx=e.clientX-lastX,dy=e.clientY-lastY;
  lastX=e.clientX;lastY=e.clientY;
  if(panning){{panX-=dx*camDist*0.001;panY+=dy*camDist*0.001}}
  else{{camTheta+=dx*0.005;camPhi=Math.max(-1.4,Math.min(1.4,camPhi+dy*0.005))}}
}});
canvas.addEventListener('wheel',e=>{{
  e.preventDefault();camDist=Math.max(0.5,camDist*(1+e.deltaY*0.001));
}},{{passive:false}});

// ---- Matrix math ----
function perspective(fov,asp,near,far){{
  const f=1/Math.tan(fov/2),nf=1/(near-far);
  return[f/asp,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0];
}}
function lookAt(ex,ey,ez,tx,ty,tz,ux,uy,uz){{
  let fx=tx-ex,fy=ty-ey,fz=tz-ez;
  let fl=Math.hypot(fx,fy,fz);fx/=fl;fy/=fl;fz/=fl;
  let sx=fy*uz-fz*uy,sy=fz*ux-fx*uz,sz=fx*uy-fy*ux;
  let sl=Math.hypot(sx,sy,sz);sx/=sl;sy/=sl;sz/=sl;
  let uux=sy*fz-sz*fy,uuy=sz*fx-sx*fz,uuz=sx*fy-sy*fx;
  return[sx,uux,-fx,0, sy,uuy,-fy,0, sz,uuz,-fz,0,
    -(sx*ex+sy*ey+sz*ez),-(uux*ex+uuy*ey+uuz*ez),(fx*ex+fy*ey+fz*ez),1];
}}
function mul4(a,b){{
  const o=new Float32Array(16);
  for(let i=0;i<4;i++)for(let j=0;j<4;j++){{
    let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;
  }}return o;
}}

// ---- Draw helpers ----
function bindAndDraw(prog,buf,count,mode,ptSize){{
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER,buf);
  const aPos=gl.getAttribLocation(prog,'a_pos');
  const aCol=gl.getAttribLocation(prog,'a_col');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos,3,gl.FLOAT,false,28,0);
  if(aCol>=0){{gl.enableVertexAttribArray(aCol);gl.vertexAttribPointer(aCol,4,gl.FLOAT,false,28,12)}}
  const uMvp=gl.getUniformLocation(prog,'u_mvp');
  gl.uniformMatrix4fv(uMvp,false,mvp);
  if(ptSize!==undefined){{const uPsz=gl.getUniformLocation(prog,'u_psz');gl.uniform1f(uPsz,ptSize)}}
  gl.drawArrays(mode,0,count);
}}

// ---- Render loop ----
gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
gl.clearColor(0.04,0.04,0.08,1);

let mvp;
function frame(){{
  requestAnimationFrame(frame);
  // Auto-rotate when idle
  if(!dragging)camTheta+=0.002;

  const asp=canvas.width/canvas.height;
  const ex=camDist*Math.cos(camPhi)*Math.cos(camTheta)+panX;
  const ey=camDist*Math.sin(camPhi)+panY;
  const ez=camDist*Math.cos(camPhi)*Math.sin(camTheta);
  const proj=perspective(0.85,asp,0.01,camDist*10);
  const view=lookAt(ex,ey,ez,panX,panY,0,0,1,0);
  mvp=mul4(proj,view);

  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);

  // Field cloud (small, dim)
  if(fCount>0)bindAndDraw(progPt,glField,fCount,gl.POINTS,1.5*devicePixelRatio);
  // Trajectory line
  if(tCount>1)bindAndDraw(progLn,glTrail,tCount,gl.LINE_STRIP);
  // Trajectory dots
  if(tCount>0)bindAndDraw(progPt,glTrail,tCount,gl.POINTS,4*devicePixelRatio);
  // Special markers (goal, start, end, splats)
  if(sCount>0)bindAndDraw(progPt,glSpec,sCount,gl.POINTS,10*devicePixelRatio);
}}
frame();
</script>
</body>
</html>"##,
        prompt = prompt_escaped,
        traj_len = data.trajectory_3d.len(),
        field_len = data.field_points_3d.len(),
        trajectory_js = trajectory_js,
        deltas_js = deltas_js,
        field_js = field_js,
        goal_js = goal_js,
        splats_js = splats_js,
        splat_alphas_js = splat_alphas_js,
    );

    if let Err(e) = std::fs::write(&html_path, &html) {
        eprintln!("    [VIZ] Failed to write viewer: {}", e);
        return;
    }

    println!(
        "    [VIZ] 3D viewer: {}",
        html_path.display()
    );

    // Open in default browser (local file, no server)
    let _ = std::process::Command::new("open")
        .arg(&html_path)
        .spawn();
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
