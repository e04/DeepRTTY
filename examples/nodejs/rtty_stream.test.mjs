// Run from this directory: npm test
import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import {
  BOTH,
  FIGURES,
  Ita2Decoder,
  LETTERS,
  OverlapMerger,
  StreamingDecoder,
  createSession,
  ctcGreedyTokens,
  decodeIta2,
  greedyCodes,
  spectrogramData,
} from "./rtty_stream.mjs";

const RATE=3200, CHAR=7.5/45.45*RATE, LTRS=31, FIGS=27;
const metadata=JSON.parse(fs.readFileSync(new URL("../../model.onnx.json",import.meta.url),"utf8"));
const modelPath=new URL("../../model.onnx",import.meta.url).pathname;

function encode(text) {
  const invert=(table)=>Object.fromEntries(Object.entries(table).map(([code,char])=>[char,Number(code)]));
  const letters=invert(LETTERS), figures=invert(FIGURES), both=invert(BOTH), codes=[]; let shift=null;
  for (const char of text) {
    if (both[char]!==undefined) { codes.push(both[char]); if (char===" ") shift="letters"; }
    else if (letters[char]!==undefined) { if (shift!=="letters") codes.push(LTRS); codes.push(letters[char]); shift="letters"; }
    else { if (shift!=="figures") codes.push(FIGS); codes.push(figures[char]); shift="figures"; }
  }
  return codes;
}

// Phase-continuous 45.45 Bd / 170 Hz FSK centered at 800 Hz, mark low, light deterministic noise.
function synthesize(codes, seconds, lead=0.3) {
  const count=Math.round(seconds*RATE), space=new Float32Array(count), bit=RATE/45.45;
  codes.forEach((code,index)=>{
    const start=lead*RATE+index*CHAR, bits=[0,...[0,1,2,3,4].map(i=>(code>>i)&1)];
    bits.forEach((value,b)=>{ if (!value) for (let i=Math.max(0,Math.ceil(start+b*bit)); i<Math.min(count,Math.ceil(start+(b+1)*bit)); i+=1) space[i]=1; });
  });
  const audio=new Float32Array(count); let phase=0, seed=1;
  const noise=()=>{ seed=(seed*1103515245+12345)%2147483648; return seed/2147483648-0.5; };
  for (let i=0;i<count;i+=1) { phase+=2*Math.PI*(715+170*space[i])/RATE; audio[i]=0.4*Math.cos(phase)+0.05*noise(); }
  return audio;
}

const tokens=(...pairs)=>pairs.map(([seconds,code])=>({sample:Math.round(seconds*RATE),code,score:0.9}));
// Windows [0, 12) and [8, 20) s with a 0.5 s guard: reliable overlap [8.5, 11.5], middle 10.0.
function merge(first, second) {
  const merger=new OverlapMerger(0.5*RATE,0.45*CHAR);
  return [...merger.push(0,12*RATE,first),...merger.push(8*RATE,12*RATE,second),...merger.finish()].map(token=>token.code);
}

test("ITA2 figures shift persists across calls", ()=>{
  const decoder=new Ita2Decoder();
  assert.equal(decoder.decode([FIGS,23,19]),"12");
  assert.equal(decoder.decode([1,10]),"34");
  assert.equal(decoder.decode([4,1])," E");
  assert.equal(decodeIta2([1,10]),"ER");
});

test("vectorized CTC tokens match greedyCodes", ()=>{
  let seed=7; const random=()=>{ seed=(seed*1103515245+12345)%2147483648; return seed/2147483648; };
  for (let trial=0;trial<200;trial+=1) {
    const frames=1+Math.floor(random()*50), data=new Float32Array(frames*33).map(()=>Math.log(random()**4+1e-9));
    for (let f=0;f<frames;f+=1) if (random()<0.5) data[f*33+32]=0;
    const codes=ctcGreedyTokens(data,0,frames,33,32).map(token=>token.code);
    assert.deepEqual(codes,greedyCodes({dims:[1,frames,33],data},32));
  }
});

test("muted prefix does not raise active audio", ()=>{
  const audio=synthesize(encode("RY".repeat(80)),12), muted=audio.slice();
  muted.fill(0,0,9*RATE);
  const reference=spectrogramData(audio,metadata), actual=spectrogramData(muted,metadata), compareFrames=300;
  const start=(actual.frames-compareFrames)*actual.bins;
  let maxDifference=0;
  for (let i=start;i<actual.data.length;i+=1) maxDifference=Math.max(maxDifference,Math.abs(actual.data[i]-reference.data[i]));
  assert.ok(maxDifference<0.11, `max normalized difference ${maxDifference}`);
});

test("character straddling the overlap middle is emitted once", ()=>{
  assert.deepEqual(merge(tokens([9.8,1],[9.99,3],[11.9,7]),tokens([8.2,9],[10.01,3],[10.2,5])),[1,3,5]);
  assert.deepEqual(merge(tokens([9.8,1],[10.01,3]),tokens([9.99,3],[10.2,5])),[1,3,5]);
});

test("unpaired tokens belong to the owning half; confident substitution wins", ()=>{
  assert.deepEqual(merge(tokens([9.0,1],[10.5,5]),tokens([9.2,6],[10.8,3])),[1,3]);
  const first=[{sample:9*RATE,code:1,score:0.55},{sample:10.5*RATE,code:5,score:0.99}];
  const second=[{sample:Math.round(9.01*RATE),code:3,score:0.97},{sample:10.5*RATE,code:6,score:0.51}];
  assert.deepEqual(merge(first,second),[3,5]);
  assert.deepEqual(merge(tokens([9.85,1],[10.0,3]),tokens([9.85,1],[9.93,9],[10.01,3])),[1,3]);
});

test("figures run across window boundaries and chunked push matches whole push", async ()=>{
  const session=await createSession(modelPath);
  const text="CQ DE JA1ABC "+"1234567890/".repeat(22)+" K\r\n", codes=encode(text);
  const audio=synthesize(codes,(codes.length+4)*CHAR/RATE);
  assert.ok(audio.length>3*38400);
  const whole=new StreamingDecoder(session,metadata,{batchWindows:3});
  const merged=[...await whole.push(audio),...await whole.flush()];
  assert.deepEqual(merged.map(token=>token.code),codes);
  assert.equal(new Ita2Decoder().decode(merged.map(token=>token.code)),text);
  const chunked=new StreamingDecoder(session,metadata); let actual=[];
  for (let start=0;start<audio.length;start+=1001) actual=actual.concat(await chunked.push(audio.subarray(start,start+1001)));
  actual=actual.concat(await chunked.flush());
  assert.deepEqual(actual.map(token=>token.code),codes);
});

test("short stream is decoded", async ()=>{
  const codes=encode("CQ CQ DE W1AW K"), decoder=new StreamingDecoder(await createSession(modelPath),metadata);
  const merged=[...await decoder.push(synthesize(codes,5)),...await decoder.flush()];
  assert.deepEqual(merged.map(token=>token.code),codes);
});
