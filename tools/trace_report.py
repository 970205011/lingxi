#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lingxi per-token 逐段 GPU 时间 → 自包含 HTML 分布报告。

数据源：`LINGXI_EVPROF_TRACE=<path>` 跑 talk 产生的 JSONL（每 decode token 每 GPU 一行）。
用法：
  python3 tools/trace_report.py out.html \
      --run "短提示(12tok上下文)|./trace_short.jsonl|24.4 ms|最优参数:双卡split 9600/10000槽+钉表200+Q4三档+tmpfs混档arena" \
      --run "64K 上下文|./trace_64k.jsonl|—|同上" \
      --git <hash>

报告内容（全部 canvas 绘制、零外部依赖、可离线打开）：
  1. 每段耗时分布（mean/p10/p50/p90 + 直方图），GPU0/GPU1 并排
  2. 逐 token 堆叠分解（段 + GPU 空闲 gap）
  3. 双卡并行视角（busy 对比 + 每段对比）
  4. 层×段热力图（层均 ms）
  5. wall / busy / gap 分布
  6. 专家段现场相关性（miss / 组数 vs 耗时散点）
"""
import json, sys, html, argparse, statistics as st

SEG_NAMES = ["入口+PLE+attn_GR", "attn(QSA/GDN)", "mlp_GR", "MoE:路由+host",
             "MoE:专家段", "gr_write(收尾)", "head(lm_head+采样)"]
SEG_COLORS = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#b3de69", "#fdb462"]


def pct(v, p):
    if not v:
        return 0.0
    return sorted(v)[min(len(v) - 1, int(p * len(v)))]


def load_run(name, path, true_wall, note):
    rows = [json.loads(l) for l in open(path)]
    gpus = {}
    for g in (0, 1):
        gg = [r for r in rows if r["gpu"] == g]
        if not gg:
            continue
        segs = {i: [r["segs"][i] for r in gg] for i in range(7)}
        layers = {}  # (l, s) -> [mean_ms, n, info_mean...]
        info_acc = {}
        for r in gg:
            for entry in r["layers"]:
                l, s, ms = entry[0], entry[1], entry[2]
                a = layers.setdefault((l, s), [0.0, 0])
                a[0] += ms
                a[1] += 1
                if s == 4:  # 专家段：记现场相关性点
                    info_acc.setdefault(len(info_acc), None)
        # 专家段现场：逐 (层,token) 点，按 (miss,ms) 收集（从原始行取，量小）
        scatter = []
        for r in gg:
            for entry in r["layers"]:
                l, s, ms = entry[0], entry[1], entry[2]
                if s == 4:
                    scatter.append([entry[3], entry[4], entry[5], entry[6], round(ms * 1000, 1)])
        n_layers = max((e[0] for r in gg for e in r["layers"]), default=47) + 1
        layer_mean = [[0.0] * 7 for _ in range(n_layers)]
        layer_cnt = [[0] * 7 for _ in range(n_layers)]
        for (l, s), (sum_, n) in layers.items():
            if l < n_layers:
                layer_mean[l][s] = sum_ / n
                layer_cnt[l][s] = n
        gpus[g] = {
            "tok": [r["tok"] for r in gg],
            "wall": [r["wall_ms"] * 1000 for r in gg],
            "busy": [r["busy_ms"] * 1000 for r in gg],
            "segs_tok": [[round(r["segs"][i] * 1000, 1) for i in range(7)] for r in gg],
            "segs": {i: {"mean": st.mean(segs[i]) * 1000, "p10": pct(segs[i], 0.1) * 1000,
                          "p50": pct(segs[i], 0.5) * 1000, "p90": pct(segs[i], 0.9) * 1000,
                          "min": min(segs[i]) * 1000, "max": max(segs[i]) * 1000,
                          "sd": (st.stdev(segs[i]) if len(segs[i]) > 1 else 0) * 1000,
                          "hist": hist(segs[i], 40)}
                       for i in range(7)},
            "layer_mean": [[round(v * 1000, 1) for v in row] for row in layer_mean],
            "scatter": scatter[:: max(1, len(scatter) // 3000)], # 上限 3000 点
        }
    return {"name": name, "true_wall": true_wall, "note": note,
            "n_tok": max(len([r for r in rows if r["gpu"] == g]) for g in (0, 1)),
            "gpus": gpus}


def hist(v, nb):
    if not v:
        return []
    lo, hi = min(v), max(v)
    if hi <= lo:
        hi = lo + 1e-9
    b = [0] * nb
    for x in v:
        b[min(nb - 1, int((x - lo) / (hi - lo) * nb))] += 1
    return {"lo": lo * 1000, "hi": hi * 1000, "bins": b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--run", action="append", required=True,
                    help="name|jsonl|true_wall|note")
    ap.add_argument("--git", default="")
    a = ap.parse_args()
    runs = []
    for spec in a.run:
        parts = spec.split("|", 3)
        name, path = parts[0], parts[1]
        tw = parts[2] if len(parts) > 2 else "—"
        note = parts[3] if len(parts) > 3 else ""
        runs.append(load_run(name, path, tw, note))
    data = {"seg_names": SEG_NAMES, "seg_colors": SEG_COLORS, "git": a.git, "runs": runs}
    payload = json.dumps(data, ensure_ascii=False)
    page = TEMPLATE.replace("__DATA__", payload)
    open(a.out, "w").write(page)
    print(f"written: {a.out} ({len(page) // 1024} KB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>lingxi decode 逐 token 逐段耗时报告</title>
<style>
  body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;margin:0;background:#0d1117;color:#c9d1d9}
  header{padding:20px 28px;border-bottom:1px solid #21262d}
  h1{font-size:20px;margin:0 0 6px}
  .sub{color:#8b949e;font-size:12px}
  .wrap{padding:20px 28px;max-width:1280px}
  h2{font-size:15px;margin:28px 0 4px;color:#58a6ff}
  .desc{color:#8b949e;font-size:12px;margin:0 0 12px}
  .cards{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
  .card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:12px 16px;min-width:150px}
  .card .k{font-size:11px;color:#8b949e}
  .card .v{font-size:20px;font-weight:600;margin-top:2px}
  .card .u{font-size:11px;color:#8b949e}
  table{border-collapse:collapse;font-size:12px;margin:8px 0}
  th,td{border:1px solid #21262d;padding:4px 10px;text-align:right}
  th{background:#161b22;color:#8b949e;font-weight:500}
  td.l,th.l{text-align:left}
  canvas{background:#161b22;border:1px solid #21262d;border-radius:8px;margin:6px 0}
  .row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
  .tt{position:fixed;background:#1c2128;border:1px solid #444c56;border-radius:6px;padding:6px 10px;font-size:12px;pointer-events:none;display:none;z-index:9;white-space:pre}
  .tabs button{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:6px 6px 0 0;padding:6px 14px;cursor:pointer;font-size:12px}
  .tabs button.on{background:#1f6feb;border-color:#1f6feb;color:#fff}
  .legend span{display:inline-block;margin-right:14px;font-size:11px}
  .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
</style>
</head>
<body>
<header>
  <h1>lingxi decode：逐 token × 逐环节 GPU 耗时分布</h1>
  <div class="sub">数据源 LINGXI_EVPROF_TRACE（CUDA event 流内计时，不排空流水）｜每 token 每 GPU 一行｜
  wall 为 trace 模式口径（每 token 双流各多一次 sync，绝对值偏高 ~10-15%，<b>段内 GPU 时间不受影响</b>）｜
  真 wall 以 bench.sh 为准（见各 run 卡片）｜git <span id="git"></span></div>
</header>
<div class="wrap">
<div class="tabs" id="tabs"></div>
<div id="body"></div>
</div>
<div class="tt" id="tt"></div>
<script>
const D = __DATA__;
const SEG = D.seg_names, COL = D.seg_colors;
document.getElementById('git').textContent = D.git;
const tt = document.getElementById('tt');
function bindTT(c, fn){ c.addEventListener('mousemove', e=>{ const r=c.getBoundingClientRect();
  const x=e.clientX-r.left, y=e.clientY-r.top; const t=fn(x,y);
  if(t){tt.style.display='block';tt.style.left=(e.clientX+12)+'px';tt.style.top=(e.clientY+12)+'px';tt.textContent=t;}
  else tt.style.display='none';}); c.addEventListener('mouseleave',()=>tt.style.display='none'); }
function setup(c, w, h){ c.width=w*devicePixelRatio; c.height=h*devicePixelRatio;
  c.style.width=w+'px'; c.style.height=h+'px'; const g=c.getContext('2d'); g.scale(devicePixelRatio,devicePixelRatio); return g; }
function pctile(v,p){ const s=[...v].sort((a,b)=>a-b); return s[Math.min(s.length-1,Math.floor(p*s.length))]; }

/* ---------- 组件 ---------- */
function cards(el, run){
  const g0=run.gpus[0], wall=g0.wall, busy=g0.busy;
  const gap=wall.map((w,i)=>w-busy[i]);
  const el2=(k,v,u)=>`<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="u">${u||''}</div></div>`;
  el.innerHTML = `<div class="cards">
    ${el2('tokens', run.n_tok, 'decode 步')}
    ${el2('真 wall（bench p10）', run.true_wall, '非 trace 口径')}
    ${el2('trace wall p50', pctile(wall,.5).toFixed(0)+' µs', 'p90 '+pctile(wall,.9).toFixed(0))}
    ${el2('GPU busy p50', pctile(busy,.5).toFixed(0)+' µs', 'p90 '+pctile(busy,.9).toFixed(0))}
    ${el2('GPU 空闲 gap p50', pctile(gap,.5).toFixed(0)+' µs', 'p90 '+pctile(gap,.9).toFixed(0))}
    ${el2('GPU 利用率', (100*pctile(busy,.5)/pctile(wall,.5)).toFixed(0)+' %', 'busy/wall（trace 口径偏低）')}
  </div><div class="desc">${run.note}</div>`;
}
function segTable(el, run){
  let h='<table><tr><th class="l">环节（GPU0 / GPU1, µs/token）</th><th>mean</th><th>p10</th><th>p50</th><th>p90</th><th>max</th><th>σ</th><th>占比</th></tr>';
  const tot=[0,0];
  for(const g of [0,1]){ const gg=run.gpus[g]; if(gg) for(let i=0;i<7;i++) tot[g]+=gg.segs[i].mean; }
  for(let i=0;i<7;i++){
    const cells=[0,1].map(g=>{ const gg=run.gpus[g]; if(!gg) return null; const s=gg.segs[i];
      return `<td>${s.mean.toFixed(0)}</td><td>${s.p10.toFixed(0)}</td><td>${s.p50.toFixed(0)}</td><td>${s.p90.toFixed(0)}</td><td>${s.max.toFixed(0)}</td><td>${s.sd.toFixed(0)}</td><td>${(100*s.mean/tot[g]).toFixed(1)}%</td>`;}).filter(Boolean).join('');
    h+=`<tr><td class="l"><span class="sw" style="background:${COL[i]}"></span>${SEG[i]}</td>${cells}</tr>`;
  }
  el.innerHTML = h+'</table><div class="desc">双卡 split 24+24：每卡只跑 24 层，两列之和 ≈ 该卡每 token 的 GPU 时间；wall ≈ 两卡串行 + head + host 间隙（两卡交叠见"双卡并行"）。</div>';
}
function histChart(el, run, gpu){
  const gg=run.gpus[gpu]; if(!gg) return;
  const W=1180,H=170,pad=36,bw=(W-pad*2)/7;
  const c=document.createElement('canvas'); el.appendChild(c); const g=setup(c,W,H);
  for(let i=0;i<7;i++){ const s=gg.segs[i]; const x0=pad+i*bw+8, ww=bw-16;
    const nb=s.hist.bins.length; const mx=Math.max(...s.hist.bins,1);
    const bwid=ww/nb;
    for(let b=0;b<nb;b++){ const hgt=(H-44)*s.hist.bins[b]/mx;
      g.fillStyle=COL[i]; g.globalAlpha=.25+.75*(b/nb);
      g.fillRect(x0+b*bwid, H-24-hgt, Math.max(1,bwid-0.5), hgt); }
    g.globalAlpha=1; g.fillStyle='#8b949e'; g.font='10px sans-serif';
    g.fillText(SEG[i], x0, H-8);
    g.fillStyle='#c9d1d9';
    g.fillText('μ'+s.mean.toFixed(0)+' p50 '+s.p50.toFixed(0), x0, 14);
  }
  bindTT(c,(x,y)=>{ const i=Math.floor((x-pad)/bw); if(i<0||i>6) return null;
    const s=gg.segs[i]; const b=Math.floor((x-pad-i*bw-8)/((bw-16)/s.hist.bins.length));
    const cnt=s.hist.bins[Math.max(0,Math.min(s.hist.bins.length-1,b))]||0;
    return SEG[i]+'\n['+ (s.hist.lo+(s.hist.hi-s.hist.lo)*Math.max(0,Math.min(1,b/s.hist.bins.length))).toFixed(0)
      +'..'+(s.hist.lo+(s.hist.hi-s.hist.lo)*Math.min(1,(b+1)/s.hist.bins.length)).toFixed(0)+'] µs × '+cnt+' tok'; });
}
function stacked(el, run, gpu){
  const gg=run.gpus[gpu]; if(!gg) return;
  const n=gg.tok.length, W=1180, H=230, padL=44, padB=22, padT=16;
  const c=document.createElement('canvas'); el.appendChild(c); const g=setup(c,W,H);
  const iw=(W-padL-8)/n;
  const wmax=Math.max(...gg.wall);
  for(let t=0;t<n;t++){
    let y=H-padB;
    const x=padL+t*iw;
    for(let i=0;i<7;i++){ const hgt=(H-padB-padT)*gg.segs_tok[t][i]/wmax;
      g.fillStyle=COL[i]; g.fillRect(x,y-hgt,Math.max(1,iw-0.3),hgt); y-=hgt; }
    const gh=(H-padB-padT)*(gg.wall[t]-gg.busy[t])/wmax;
    g.fillStyle='#3a3f45'; g.fillRect(x,y-gh,Math.max(1,iw-0.3),gh);
  }
  g.fillStyle='#8b949e'; g.font='10px sans-serif'; g.fillText(wmax.toFixed(0)+' µs', 4, padT+8);
  g.fillText('0', 4, H-padB);
  let lg=`<span class="sw" style="background:#3a3f45"></span>GPU 空闲(host/sync)`;
  for(let i=0;i<7;i++) lg=`<span class="sw" style="background:${COL[i]}"></span>${SEG[i]}`+lg;
  const dv=document.createElement('div'); dv.className='legend'; dv.innerHTML=lg; el.appendChild(dv);
  bindTT(c,(x,y)=>{ const t=Math.floor((x-padL)/iw); if(t<0||t>=n) return null;
    let s='token '+gg.tok[t]+'\nwall '+gg.wall[t].toFixed(0)+' µs';
    for(let i=0;i<7;i++) s+='\n'+SEG[i]+' '+gg.segs_tok[t][i].toFixed(0);
    s+='\ngap '+(gg.wall[t]-gg.busy[t]).toFixed(0); return s; });
}
function dual(el, run){
  const a=run.gpus[0], b=run.gpus[1]; if(!b) return;
  const W=1180,H=200,padL=44,padB=22,padT=14;
  const c=document.createElement('canvas'); el.appendChild(c); const g=setup(c,W,H);
  const n=a.tok.length, iw=(W-padL-8)/n, wmax=Math.max(...a.wall);
  const plot=(arr,col,fill)=>{ for(let t=0;t<n;t++){ const hgt=(H-padB-padT)*arr[t]/wmax;
    if(fill) g.fillRect(padL+t*iw, H-padB-hgt, Math.max(1,iw-0.3), hgt);
    else { g.fillStyle=col; g.fillRect(padL+t*iw, H-padB-hgt, Math.max(1,iw-0.3), 2); } } };
  g.fillStyle='#123a5e'; plot(b.busy,'#58a6ff',true);
  g.fillStyle='#1f6feb55'; plot(a.busy,'#f85149',true);
  g.fillStyle='#58a6ff'; plot(b.busy);
  g.fillStyle='#f85149'; plot(a.busy);
  g.fillStyle='#8b949e'; g.font='10px sans-serif'; g.fillText(wmax.toFixed(0)+' µs',4,padT+6);
  const dv=document.createElement('div'); dv.className='legend';
  dv.innerHTML=`<span class="sw" style="background:#f85149"></span>GPU0 busy（柱+线）
    <span class="sw" style="background:#58a6ff"></span>GPU1 busy（柱+线）
    <span class="sw" style="background:#8b949e"></span>wall 见堆叠图`;
  el.appendChild(dv);
  bindTT(c,(x,y)=>{ const t=Math.floor((x-padL)/iw); if(t<0||t>=n) return null;
    return 'token '+a.tok[t]+'\nGPU0 busy '+a.busy[t].toFixed(0)+' µs\nGPU1 busy '+b.busy[t].toFixed(0)+' µs'; });
}
function segCompare(el, run){
  const a=run.gpus[0], b=run.gpus[1]; if(!b) return;
  const W=1180,H=240,padL=150,padB=22,padT=14;
  const c=document.createElement('canvas'); el.appendChild(c); const g=setup(c,W,H);
  const tots=[0,0]; for(let i=0;i<7;i++){ tots[0]+=a.segs[i].mean; tots[1]+=b.segs[i].mean; }
  const rowH=(H-padB-padT)/7;
  for(let i=0;i<7;i++){
    const y=padT+i*rowH;
    g.fillStyle='#8b949e'; g.font='11px sans-serif'; g.textAlign='right';
    g.fillText(SEG[i], padL-8, y+rowH/2+4); g.textAlign='left';
    const scale=(W-padL-60)/Math.max(...tots);
    g.fillStyle='#f85149'; g.fillRect(padL, y+3, a.segs[i].mean*scale, rowH/2-4);
    g.fillStyle='#58a6ff'; g.fillRect(padL, y+rowH/2+1, b.segs[i].mean*scale, rowH/2-4);
    g.fillStyle='#c9d1d9'; g.font='10px sans-serif';
    g.fillText('G0 '+a.segs[i].mean.toFixed(0)+' / G1 '+b.segs[i].mean.toFixed(0), padL+a.segs[i].mean*scale+6, y+rowH/2+3);
  }
}
function heatmap(el, run, gpu){
  const gg=run.gpus[gpu]; if(!gg) return;
  const L=gg.layer_mean.length, W=1180, padL=44, padT=34;
  const cw=(W-padL-8)/7, chh=Math.min(16,(520-padT)/L);
  const H=padT+chh*L+18;
  const c=document.createElement('canvas'); el.appendChild(c); const g=setup(c,W,H);
  const mx=Math.max(...gg.layer_mean.flat(), 1e-6);
  for(let l=0;l<L;l++) for(let s=0;s<7;s++){
    const v=gg.layer_mean[l][s]; if(v<=0) continue;
    const t=Math.pow(v/mx,0.45);
    g.fillStyle=COL[s]; g.globalAlpha=0.12+0.88*t;
    g.fillRect(padL+s*cw, padT+l*chh, cw-1, chh-1);
  }
  g.globalAlpha=1; g.fillStyle='#8b949e'; g.font='10px sans-serif';
  for(let s=0;s<7;s++) g.fillText(SEG[s], padL+s*cw+2, padT-6);
  for(let l=0;l<L;l+=2){ g.fillText(l===L-1?'head':String(l), 6, padT+l*chh+chh-3); }
  bindTT(c,(x,y)=>{ const s=Math.floor((x-padL)/cw), l=Math.floor((y-padT)/chh);
    if(s<0||s>6||l<0||l>=L) return null;
    const v=gg.layer_mean[l][s]; if(!v) return null;
    return (l===L-1?'head':'层 '+l)+' · '+SEG[s]+'\n层均 '+v.toFixed(1)+' µs'; });
}
function distChart(el, run, gpu){
  const gg=run.gpus[gpu]; if(!gg) return;
  const W=380,H=150,pad=30;
  const mk=(vals,title,col)=>{ const c=document.createElement('canvas'); el.appendChild(c);
    const g=setup(c,W,H); const s=hist(vals,36);
    const mx=Math.max(...s.bins,1), bw=(W-pad*2)/36;
    for(let i=0;i<36;i++){ const hgt=(H-44)*s.bins[i]/mx;
      g.fillStyle=col; g.globalAlpha=.35+.65*i/36; g.fillRect(pad+i*bw,H-22-hgt,bw-1,hgt); }
    g.globalAlpha=1; g.fillStyle='#8b949e'; g.font='10px sans-serif';
    g.fillText(title, 4, 12);
    g.fillText(s.lo.toFixed(0), pad, H-8); g.fillText(s.hi.toFixed(0), W-pad-30, H-8);
    g.fillStyle='#c9d1d9';
    g.fillText('p50 '+pctile(vals,.5).toFixed(0)+'  p90 '+pctile(vals,.9).toFixed(0), pad, 12);
  };
  mk(gg.wall,'wall (trace 口径)','#8b949e');
  mk(gg.busy,'GPU busy','#3fb950');
  const gap=gg.wall.map((w,i)=>w-gg.busy[i]);
  mk(gap,'GPU 空闲 gap','#d29922');
}
function hist(v,nb){ const lo=Math.min(...v),hi=Math.max(...v),sp=(hi-lo)||1e-9;
  const b=new Array(nb).fill(0); for(const x of v) b[Math.min(nb-1,Math.floor((x-lo)/sp*nb))]++;
  return {lo,hi,bins:b}; }
function scatter(el, run, gpu){
  const gg=run.gpus[gpu]; if(!gg||!gg.scatter.length) return;
  const W=1180,H=260,padL=46,padB=26,padT=14;
  const c=document.createElement('canvas'); el.appendChild(c); const g=setup(c,W,H);
  const xs=gg.scatter.map(p=>p[3]), ys=gg.scatter.map(p=>p[4]); // miss, ms
  const xmax=Math.max(...xs,1), ymax=Math.max(...ys,1);
  const px=x=>padL+(x/xmax)*(W-padL-20);
  const py=y=>H-padB-(y/ymax)*(H-padB-padT);
  for(const p of gg.scatter){ g.fillStyle=p[3]>0?'#f85149':'#3fb950'; g.globalAlpha=.45;
    g.fillRect(px(p[3])-1.5, py(p[4])-1.5, 3, 3); }
  g.globalAlpha=1; g.strokeStyle='#30363d'; g.beginPath();
  g.moveTo(padL,padT); g.lineTo(padL,H-padB); g.lineTo(W-10,H-padB); g.stroke();
  g.fillStyle='#8b949e'; g.font='10px sans-serif';
  g.fillText(ymax.toFixed(0)+' µs', 4, padT+8); g.fillText('miss 数 →', W-70, H-8);
  const d=document.createElement('div'); d.className='legend';
  d.innerHTML='<span class="sw" style="background:#3fb950"></span>miss=0 <span class="sw" style="background:#f85149"></span>miss≥1　每个点 = (层, token) 的专家段';
  el.appendChild(d);
  bindTT(c,(x,y)=>{ let best=null,bd=1e9;
    for(const p of gg.scatter){ const dx=px(p[3])-x, dy=py(p[4])-y, d2=dx*dx+dy*dy;
      if(d2<bd){bd=d2;best=p;} }
    return bd<200?('miss='+best[3]+' 组='+best[1]+' rows='+best[2]+' 专家段 '+best[4]+' µs'):null; });
}

/* ---------- 页面 ---------- */
const tabs=document.getElementById('tabs'), body=document.getElementById('body');
D.runs.forEach((run,ri)=>{
  const b=document.createElement('button'); b.textContent=run.name+(run.true_wall!=='—'?'（真 wall '+run.true_wall+'）':'');
  if(ri===0) b.classList.add('on');
  b.onclick=()=>{ [...tabs.children].forEach(x=>x.classList.remove('on')); b.classList.add('on'); render(run); };
  tabs.appendChild(b);
});
function render(run){
  body.innerHTML='';
  const H=(t,d)=>{ const h=document.createElement('h2'); h.textContent=t; body.appendChild(h);
    const p=document.createElement('p'); p.className='desc'; p.innerHTML=d; body.appendChild(p); };
  const div=cls=>{ const d=document.createElement('div'); if(cls) d.className=cls; body.appendChild(d); return d; };
  H('总览','trace 口径 wall 系统性偏高（每 token 双流各一次同步杀掉 host 预跑），段内 GPU 时间与 gap 结构可信；真 wall 看卡片。');
  cards(div(), run);
  H('每个环节耗时分布（逐 token 统计 · µs/token）','左表数字；下图是同数据的概率分布（直方图，越右越慢）。p10=最快 10% 的 token，p90=最慢 10%。');
  segTable(div(), run); histChart(div('row'), run, 0);
  H('逐 token 分解（GPU0）','每根柱 = 一个 token：七段堆叠 + 顶部灰色 = GPU 空闲（host 决策/同步/launch 间隙）。悬停看数值。');
  stacked(div(), run, 0);
  H('逐 token 分解（GPU1）',''); stacked(div(), run, 1);
  H('双卡并行视角','双卡 split 是"层间串行、卡间几乎无交叠"（host 单线程逐层发 launch）：wall ≈ GPU0 busy + GPU1 busy + head + host 间隙。哪根柱高哪张卡就是关键路径。');
  dual(div(), run); segCompare(div(), run);
  H('wall / busy / gap 分布','wall（trace）与 busy 的差 = GPU 空闲：这是"还能榨"的空间。');
  distChart(div('row'), run, 0);
  H('层 × 段 热力图（GPU0 层均 µs）','纵轴层号（最后行 = head 伪层），横轴七段；颜色越深越慢。一眼定位"哪类层的哪段"是热点。');
  heatmap(div(), run, 0);
  H('专家段现场相关性（GPU0）','散点 = (层, token)：专家段耗时 vs 本层 miss 数。miss=0 还慢的层是"组数/行数"问题，miss≥1 慢是 DMA 暴露。');
  scatter(div(), run, 0);
}
render(D.runs[0]);
</script>
</body></html>"""

if __name__ == "__main__":
    main()
