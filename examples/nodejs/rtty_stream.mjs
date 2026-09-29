// Continuous DeepRTTY reception with overlapping windows.
//
// Each 12-second window is decoded to CTC tokens stamped with an absolute sample
// time and a confidence. Consecutive windows are merged by aligning the tokens in
// their reliable overlap, so each transmitted character is emitted exactly once.
// The merged ITA2 code stream then passes through one stateful decoder, which
// carries the LTRS/FIGS shift across window boundaries.
//
//   ffmpeg -i input.wav -ac 1 -ar 3200 -f s16le - | node rtty_stream.mjs
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import * as ort from "onnxruntime-node";

export const LETTERS = {1:"E",3:"A",5:"S",6:"I",7:"U",9:"D",10:"R",11:"J",12:"N",13:"F",14:"C",15:"K",16:"T",17:"Z",18:"L",19:"W",20:"H",21:"Y",22:"P",23:"Q",24:"O",25:"B",26:"G",28:"M",29:"X",30:"V"};
export const FIGURES = {1:"3",3:"-",5:"\x07",6:"8",7:"7",9:"$",10:"4",11:"'",12:",",13:"!",14:":",15:"(",16:"5",17:'"',18:")",19:"2",20:"#",21:"6",22:"0",23:"1",24:"9",25:"?",26:"&",28:".",29:"/",30:";"};
export const BOTH = {2:"\n",4:" ",8:"\r"};

function reflectPad(audio, pad) {
  const output=new Float32Array(audio.length+2*pad); output.set(audio,pad);
  for (let i=0;i<pad;i+=1) { output[i]=audio[pad-i]; output[pad+audio.length+i]=audio[audio.length-2-i]; }
  return output;
}

// cos/sin tables for the selected bins, computed exactly as the per-sample
// Math.cos/Math.sin calls they replace.
const twiddleCache = new Map();
function twiddles(nfft, first, stop) {
  const key = `${nfft}:${first}:${stop}`;
  if (!twiddleCache.has(key)) {
    const cos = new Float64Array((stop-first)*nfft), sin = new Float64Array((stop-first)*nfft);
    for (let bin=first;bin<stop;bin+=1) for (let n=0;n<nfft;n+=1) {
      const angle=-2*Math.PI*bin*n/nfft;
      cos[(bin-first)*nfft+n]=Math.cos(angle); sin[(bin-first)*nfft+n]=Math.sin(angle);
    }
    twiddleCache.set(key, {cos, sin});
  }
  return twiddleCache.get(key);
}

export function spectrogramData(audio, metadata) {
  const nfft=metadata.fft_length, winLength=metadata.win_length, hop=metadata.hop_length, binHz=metadata.sample_rate/nfft;
  const first=Math.ceil(metadata.spectrogram_min_freq_hz/binHz), stop=Math.floor(metadata.spectrogram_max_freq_hz/binHz)+1, bins=stop-first;
  const window=new Float32Array(nfft), winOffset=Math.floor((nfft-winLength)/2);
  for (let i=0;i<winLength;i+=1) window[winOffset+i]=0.5-0.5*Math.cos(2*Math.PI*i/winLength);
  const {cos, sin}=twiddles(nfft, first, stop), values=new Float64Array(nfft);
  const padded=reflectPad(audio,Math.floor(nfft/2)), frames=1+Math.floor((padded.length-nfft)/hop), magnitude=new Float32Array(frames*bins);
  for (let frame=0;frame<frames;frame+=1) {
    for (let n=0;n<nfft;n+=1) values[n]=padded[frame*hop+n]*window[n];
    for (let b=0;b<bins;b+=1) {
      let re=0, im=0;
      for (let n=0, k=b*nfft;n<nfft;n+=1,k+=1) { re+=values[n]*cos[k]; im+=values[n]*sin[k]; }
      magnitude[frame*bins+b]=Math.hypot(re,im);
    }
  }
  let noiseValues=magnitude;
  if (metadata.normalization==="noise_q25_active_log1p") {
    const frameRank=Math.floor(metadata.noise_level_quantile*(bins-1)), frameLevels=new Float32Array(frames);
    for (let frame=0;frame<frames;frame+=1) {
      const row=magnitude.slice(frame*bins,(frame+1)*bins).sort();
      frameLevels[frame]=row[frameRank];
    }
    const levels=frameLevels.slice().sort(), referenceRank=Math.floor(metadata.silent_frame_reference_quantile*(frames-1));
    const threshold=levels[referenceRank]*metadata.silent_frame_relative_level;
    let activeFrames=0; for (const value of frameLevels) if (value>=threshold) activeFrames+=1;
    noiseValues=new Float32Array(activeFrames*bins);
    for (let frame=0, offset=0;frame<frames;frame+=1) if (frameLevels[frame]>=threshold) {
      noiseValues.set(magnitude.subarray(frame*bins,(frame+1)*bins),offset); offset+=bins;
    }
  }
  const sorted=noiseValues.slice().sort(), rank=Math.floor(metadata.noise_level_quantile*(sorted.length-1));
  let peak=0; for (const value of magnitude) peak=Math.max(peak,value);
  const noise=Math.max(sorted[rank]/metadata.noise_level_rayleigh_factor,peak*metadata.noise_level_min_relative_to_peak,metadata.noise_level_min);
  const data=new Float32Array(magnitude.length); for (let i=0;i<data.length;i+=1) data[i]=Math.log1p(magnitude[i]/noise);
  return {data, frames, bins};
}

// Simple reference collapse used to verify the timestamped stream decoder.
export function greedyCodes(output, blank) {
  const [,frames,classes]=output.dims, codes=[]; let previous=null;
  for (let frame=0;frame<frames;frame+=1) {
    let best=0, value=-Infinity;
    for (let c=0;c<classes;c+=1) if (output.data[frame*classes+c]>value) { value=output.data[frame*classes+c]; best=c; }
    if (best===blank) previous=null;
    else { if (best!==previous) codes.push(best); previous=best; }
  }
  return codes;
}

// Stateful ITA2 decoder: the letters/figures shift persists across decode() calls.
export class Ita2Decoder {
  constructor(usos=true, shift="letters") { this.usos=usos; this.shift=shift; }
  decode(codes) {
    let figures=this.shift==="figures", text="";
    for (const code of codes) {
      if (code===31) figures=false;
      else if (code===27) figures=true;
      else if (code===0) continue;
      else if (BOTH[code]!==undefined) { text+=BOTH[code]; if (code===4 && this.usos) figures=false; }
      else text+=(figures?FIGURES:LETTERS)[code];
    }
    this.shift=figures?"figures":"letters";
    return text;
  }
}

export function decodeIta2(codes, usos=true) {
  return new Ita2Decoder(usos).decode(codes);
}

const CHARACTER_BITS = 1 + 5 + 1.5;
export const DEFAULT_CONFIG = {hopSeconds:8, edgeGuardSeconds:0.5, matchToleranceSeconds:null, batchWindows:1};

// Greedy CTC over one [frames x classes] slice. Returns {frame, code, score} per emitted
// token: run-center frame (possibly .5), code, and peak posterior. Codes equal greedyCodes().
export function ctcGreedyTokens(data, offset, frames, classes, blank) {
  const tokens=[]; let runCode=-1, runStart=0, runPeak=-Infinity;
  const close=(stop)=>{ if (runCode>=0 && runCode!==blank) tokens.push({frame:(runStart+stop-1)/2, code:runCode, score:Math.exp(runPeak)}); };
  for (let frame=0;frame<frames;frame+=1) {
    let best=0, value=-Infinity, base=offset+frame*classes;
    for (let c=0;c<classes;c+=1) if (data[base+c]>value) { value=data[base+c]; best=c; }
    if (best!==runCode) { close(frame); runCode=best; runStart=frame; runPeak=value; }
    else if (value>runPeak) runPeak=value;
  }
  close(frames);
  return tokens;
}

// Time-constrained alignment of two short token sequences. Pairs are allowed only within
// `tolerance` samples; equal codes cost less than substitutions and both cost less than two
// unpaired tokens, so a spurious neighbour cannot steal the partner of a real character.
function align(a, b, tolerance) {
  const n=a.length, m=b.length, width=m+1, gap=1;
  const cost=new Float64Array((n+1)*width), move=new Uint8Array((n+1)*width); // 0 pair, 1 a-only, 2 b-only
  for (let i=1;i<=n;i+=1) { cost[i*width]=i*gap; move[i*width]=1; }
  for (let j=1;j<=m;j+=1) { cost[j]=j*gap; move[j]=2; }
  for (let i=1;i<=n;i+=1) for (let j=1;j<=m;j+=1) {
    let best=cost[(i-1)*width+j]+gap, step=1;
    if (cost[i*width+j-1]+gap<best) { best=cost[i*width+j-1]+gap; step=2; }
    const dt=Math.abs(a[i-1].sample-b[j-1].sample);
    if (dt<=tolerance) {
      const pair=cost[(i-1)*width+j-1]+0.5*dt/tolerance+(a[i-1].code===b[j-1].code?0:0.9);
      if (pair<=best) { best=pair; step=0; }
    }
    cost[i*width+j]=best; move[i*width+j]=step;
  }
  const pairs=[];
  for (let i=n, j=m; i||j;) {
    const step=move[i*width+j];
    if (step===0) { pairs.push([a[i-1],b[j-1]]); i-=1; j-=1; }
    else if (step===1) { pairs.push([a[i-1],null]); i-=1; }
    else { pairs.push([null,b[j-1]]); j-=1; }
  }
  return pairs.reverse();
}

// Merges token streams of overlapping windows pushed in order of increasing start. Only the
// reliable part of each window (interior edges minus `edgeGuard` samples) competes. In the
// reliable overlap of two consecutive windows a paired token comes from the more confident
// window; an unpaired token survives only in the window owning its half of the overlap.
export class OverlapMerger {
  constructor(edgeGuard, tolerance) {
    this.edgeGuard=Math.round(edgeGuard); this.tolerance=tolerance;
    this.pending=[]; this.pendingReliableEnd=-Infinity; this.cutoff=-Infinity; this.windows=0;
  }

  push(windowStart, windowLength, tokens) {
    const reliableStart=windowStart+(this.windows?this.edgeGuard:0);
    tokens=tokens.filter(token=>token.sample>=reliableStart);
    this.windows+=1;
    if (this.windows===1) {
      this.pending=tokens; this.pendingReliableEnd=windowStart+windowLength-this.edgeGuard;
      return [];
    }
    let zoneStart=Math.max(reliableStart,this.cutoff), zoneEnd=this.pendingReliableEnd;
    // No reliable overlap: trust each window up to the middle of the gap.
    if (zoneEnd<zoneStart) zoneStart=zoneEnd=(zoneStart+zoneEnd)/2;
    const middle=(zoneStart+zoneEnd)/2, inZone=token=>token.sample>=zoneStart && token.sample<=zoneEnd;
    const committed=this.pending.filter(token=>token.sample<zoneStart), merged=[];
    for (const [previous,current] of align(this.pending.filter(inZone),tokens.filter(inZone),this.tolerance)) {
      if (previous && current) merged.push(previous.score>=current.score?previous:current);
      else if (previous && previous.sample<middle) merged.push(previous);
      else if (current && current.sample>=middle) merged.push(current);
    }
    // Unpaired tokens between two pairs come back in arbitrary a/b order.
    merged.sort((x,y)=>x.sample-y.sample);
    this.cutoff=zoneEnd;
    this.pending=tokens.filter(token=>token.sample>zoneEnd);
    this.pendingReliableEnd=windowStart+windowLength-this.edgeGuard;
    return committed.concat(merged);
  }

  finish(streamEnd=Infinity) {
    const committed=this.pending.filter(token=>token.sample<streamEnd);
    this.pending=[];
    return committed;
  }
}

export async function createSession(modelPath) {
  return ort.InferenceSession.create(modelPath,{graphOptimizationLevel:"all"});
}

// Push Float32Array chunks of any size at the model sample rate, already centered at 800 Hz;
// receive merged tokens {sample, code, score} as they become final.
export class StreamingDecoder {
  constructor(session, metadata, config={}) {
    this.session=session; this.metadata=metadata; this.config={...DEFAULT_CONFIG,...config};
    const rate=metadata.sample_rate;
    this.outputHop=metadata.output_hop_length;
    this.window=Math.round(metadata.window_seconds*rate);
    // Keep window starts on the output-frame grid so spikes of different windows are comparable.
    this.hop=Math.max(this.outputHop,Math.round(this.config.hopSeconds*rate/this.outputHop)*this.outputHop);
    const edgeGuard=Math.round(this.config.edgeGuardSeconds*rate), character=CHARACTER_BITS/metadata.rtty_baud*rate;
    const tolerance=this.config.matchToleranceSeconds===null?0.45*character:this.config.matchToleranceSeconds*rate;
    if (this.window-this.hop-2*edgeGuard<2*character) throw new Error("Window overlap minus both edge guards must cover at least two characters.");
    this.merger=new OverlapMerger(edgeGuard,tolerance);
    this.buffer=new Float32Array(this.window+this.hop);
    this.fill=0; this.bufferStart=0; this.nextWindow=0; this.lastWindowEnd=0; this.queued=[];
  }

  get totalSamples() { return this.bufferStart+this.fill; }

  async push(audio) {
    let committed=[];
    for (let position=0; position<audio.length;) {
      if (this.fill===this.buffer.length) this.compact();
      const take=Math.min(this.buffer.length-this.fill,audio.length-position);
      this.buffer.set(audio.subarray(position,position+take),this.fill);
      this.fill+=take; position+=take;
      while (this.nextWindow+this.window<=this.totalSamples) {
        const offset=this.nextWindow-this.bufferStart;
        this.queued.push({start:this.nextWindow,audio:this.buffer.slice(offset,offset+this.window)});
        this.lastWindowEnd=this.nextWindow+this.window;
        this.nextWindow+=this.hop;
        committed=committed.concat(await this.run(false));
      }
    }
    return committed;
  }

  // Decode the remaining tail and return every outstanding token. Call once at end of stream.
  async flush() {
    const total=this.totalSamples;
    let committed=await this.run(true);
    if (total && (total>this.lastWindowEnd || this.lastWindowEnd===0)) {
      const start=Math.max(0,total-this.window), audio=new Float32Array(this.window);
      const tail=this.buffer.subarray(start-this.bufferStart,total-this.bufferStart);
      audio.set(tail);
      // Shorter than one window: mirror-pad (like numpy "symmetric") and drop tokens past the end.
      for (let i=tail.length, period=2*tail.length; i<this.window; i+=1) {
        const k=i%period; audio[i]=tail[k<tail.length?k:period-1-k];
      }
      this.queued.push({start,audio});
      committed=committed.concat(await this.run(true));
    }
    return committed.concat(this.merger.finish(total));
  }

  compact() {
    // Drop samples no future window needs, keeping enough for a tail window at flush.
    // A full buffer always holds a cut window, so this frees at least one sample.
    const keepFrom=Math.min(this.nextWindow,this.totalSamples-this.window), drop=keepFrom-this.bufferStart;
    this.buffer.copyWithin(0,drop,this.fill);
    this.fill-=drop; this.bufferStart=keepFrom;
  }

  async run(force) {
    if (!this.queued.length || (!force && this.queued.length<this.config.batchWindows)) return [];
    const batch=this.queued; this.queued=[];
    const specs=batch.map(({audio})=>spectrogramData(audio,this.metadata)), {frames,bins}=specs[0];
    const input=new Float32Array(batch.length*frames*bins);
    specs.forEach((spec,index)=>input.set(spec.data,index*frames*bins));
    const outputs=await this.session.run({[this.metadata.onnx_input_name]:new ort.Tensor("float32",input,[batch.length,1,frames,bins])});
    const output=outputs[this.metadata.onnx_output_name], [,outFrames,classes]=output.dims;
    let committed=[];
    batch.forEach(({start},index)=>{
      const tokens=ctcGreedyTokens(output.data,index*outFrames*classes,outFrames,classes,this.metadata.blank_index)
        .map(({frame,code,score})=>({sample:start+Math.round(frame*this.outputHop),code,score}));
      committed=committed.concat(this.merger.push(start,this.window,tokens));
    });
    return committed;
  }
}

// Decode an arbitrarily long, already-centered recording into merged tokens.
export async function decodeLong(audio, modelPath, metadata, config={}, session=null) {
  const decoder=new StreamingDecoder(session??await createSession(modelPath),metadata,config);
  return (await decoder.push(audio)).concat(await decoder.flush());
}

function parseArgs() {
  const args={model:"../../model.onnx", metadata:"../../model.onnx.json", rawCodes:false, usos:true, config:{}};
  for (let i=2;i<process.argv.length;i+=1) {
    const key=process.argv[i];
    if (key==="--raw-codes") args.rawCodes=true;
    else if (key==="--no-usos") args.usos=false;
    else if (key==="--model" || key==="--metadata") args[key.slice(2)]=process.argv[++i];
    else if (key==="--hop-seconds") args.config.hopSeconds=Number(process.argv[++i]);
    else if (key==="--edge-guard-seconds") args.config.edgeGuardSeconds=Number(process.argv[++i]);
    else if (key==="--batch-windows") args.config.batchWindows=Number(process.argv[++i]);
    else throw new Error(`Unknown argument: ${key}`);
  }
  return args;
}

async function main() {
  const args=parseArgs(), metadata=JSON.parse(fs.readFileSync(path.resolve(args.metadata),"utf8"));
  const decoder=new StreamingDecoder(await createSession(path.resolve(args.model)),metadata,args.config), ita2=new Ita2Decoder(args.usos);
  const emit=(tokens)=>{
    if (!tokens.length) return;
    const codes=tokens.map(token=>token.code);
    process.stdout.write(args.rawCodes?codes.join(" ")+" ":ita2.decode(codes));
  };
  let remainder=Buffer.alloc(0);
  for await (const chunk of process.stdin) {
    const data=Buffer.concat([remainder,chunk]), usable=data.length-data.length%2;
    remainder=data.subarray(usable);
    const audio=new Float32Array(usable/2);
    for (let i=0;i<audio.length;i+=1) audio[i]=data.readInt16LE(2*i)/32768;
    emit(await decoder.push(audio));
  }
  emit(await decoder.flush());
  process.stdout.write("\n");
}

if (process.argv[1] && import.meta.url===pathToFileURL(fs.realpathSync(process.argv[1])).href) {
  main().catch(error=>{console.error(error);process.exitCode=1;});
}
