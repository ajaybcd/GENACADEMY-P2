"""Local, dependency-free SVG viewer for the authored architecture diagram."""


def architecture_viewer(svg):
    start = svg.index("<svg")
    end = svg.rindex("</svg>") + len("</svg>")
    drawing = svg[start:end]
    return r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}html,body{margin:0;height:100%;font-family:Arial,sans-serif;color:#133d39;background:#fff}
#viewer{height:100vh;display:flex;flex-direction:column;border:1px solid #c7ddd5;border-radius:10px;overflow:hidden;background:white}
.tools{display:flex;gap:7px;align-items:center;flex-wrap:wrap;padding:10px;background:#eaf2ef;border-bottom:1px solid #c7ddd5;flex:none}
button{background:white;border:1px solid #9dbdb3;color:#133d39;border-radius:6px;padding:8px 11px;font-size:13px;cursor:pointer}
button:hover{background:#d5e9e1}button:focus-visible{outline:3px solid #126c60;outline-offset:2px}
#percent{min-width:46px;text-align:center;font-weight:bold;font-size:13px}
#canvas{flex:1;min-height:0;overflow:auto;position:relative;cursor:grab;overscroll-behavior:contain;touch-action:pan-x pan-y}
#canvas.dragging{cursor:grabbing}#canvas svg{display:block;max-width:none;flex:none;user-select:none;touch-action:inherit}
.hint{font-size:12px;padding:7px 10px;margin:0;background:#f7faf9;flex:none;color:#42665c}
#viewer:fullscreen{width:100vw;height:100vh;border:0;border-radius:0}
</style></head><body>
<div id="viewer">
<div class="tools" role="toolbar" aria-label="Architecture diagram controls">
 <button id="out" aria-label="Zoom out">−</button>
 <span id="percent" role="status" aria-live="polite"></span>
 <button id="in" aria-label="Zoom in">+</button>
 <button id="fit" aria-label="Fit complete diagram">Fit</button>
 <button id="actual" aria-label="Actual size 100 percent">100%</button>
 <button id="full" aria-label="Full screen diagram">Full screen</button>
</div>
<div id="canvas" tabindex="0" role="region" aria-label="Zoomable architecture diagram. Drag to pan. Use plus and minus keys to zoom.">__SVG__</div>
<p id="hint" class="hint">Drag to pan or use the scrollbars. Ctrl + mouse wheel zooms. Escape exits full screen.</p>
</div>
<script>
const root=document.getElementById('viewer'),canvas=document.getElementById('canvas'),svg=canvas.querySelector('svg');
const box=svg.viewBox.baseVal,w=box.width*4/3,h=box.height*4/3;
let scale=1,drag=null,hasFitted=false;
function zoom(next,x=canvas.clientWidth/2,y=canvas.clientHeight/2){
 const old=scale,left=(canvas.scrollLeft+x)/old,top=(canvas.scrollTop+y)/old;
 scale=Math.max(.05,Math.min(4,next));
 svg.style.width=(w*scale)+'px';svg.style.height=(h*scale)+'px';
 canvas.scrollLeft=left*scale-x;canvas.scrollTop=top*scale-y;
 document.getElementById('percent').textContent=Math.round(scale*100)+'%';
}
function fit(){if(!canvas.clientWidth||!canvas.clientHeight)return;zoom(Math.min(canvas.clientWidth/w,canvas.clientHeight/h)*.97);canvas.scrollLeft=0;canvas.scrollTop=0;hasFitted=true}
document.getElementById('in').onclick=()=>zoom(scale*1.25);
document.getElementById('out').onclick=()=>zoom(scale/1.25);
document.getElementById('actual').onclick=()=>zoom(1);
document.getElementById('fit').onclick=fit;
document.getElementById('full').onclick=async()=>{
 try{if(document.fullscreenElement)await document.exitFullscreen();else await root.requestFullscreen();}
 catch(e){document.getElementById('hint').textContent='Full screen is unavailable in this browser panel. Use Open larger view above, or download the SVG and open it in your browser.';}
};
document.addEventListener('fullscreenchange',()=>{document.getElementById('full').textContent=document.fullscreenElement?'Exit full screen':'Full screen';setTimeout(fit,50)});
canvas.addEventListener('wheel',e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();const r=canvas.getBoundingClientRect();zoom(scale*(e.deltaY<0?1.12:1/1.12),e.clientX-r.left,e.clientY-r.top)}},{passive:false});
canvas.addEventListener('pointerdown',e=>{if(e.button!==0||e.pointerType==='touch')return;drag={x:e.clientX,y:e.clientY,l:canvas.scrollLeft,t:canvas.scrollTop};canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging')});
canvas.addEventListener('pointermove',e=>{if(drag){canvas.scrollLeft=drag.l-(e.clientX-drag.x);canvas.scrollTop=drag.t-(e.clientY-drag.y)}});
for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>{drag=null;canvas.classList.remove('dragging')});
canvas.addEventListener('keydown',e=>{if(e.key==='+'||e.key==='='){e.preventDefault();zoom(scale*1.25)}else if(e.key==='-'){e.preventDefault();zoom(scale/1.25)}else if(e.key==='0'){e.preventDefault();fit()}});
new ResizeObserver(()=>{if(!hasFitted)fit()}).observe(canvas);
requestAnimationFrame(fit);
</script></body></html>'''.replace("__SVG__", drawing)
