import { createInterface } from 'node:readline';
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash, randomUUID } from 'node:crypto';
import { OLaya } from '../../../packages/olaya-node/src/engine';
import { OntologyDAG, type Concept } from '../../../packages/olaya-node/src/ontology';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const skills = ['olaya-tbox-workflows', 'olaya-evidence-review'];
const caps = {backend:'lexical-node',trained:false,offline:true,workflowExecution:false,workflowReplay:false,description:'Ground text against an explicitly supplied bounded class DAG. Scores are heuristic, not calibrated probabilities. JSON-LD is observation provenance, not a workflow execution trace.'};
const tools = [
 {name:'olaya_capabilities',description:'Return backend identity and implemented capabilities.',inputSchema:{type:'object',properties:{},additionalProperties:false}},
 {name:'olaya_ground',description:'Lexically rank supplied ontology classes; emit JSON-LD observation evidence. Not trained inference or workflow execution.',inputSchema:{type:'object',properties:{state:{type:'string',maxLength:16384},concepts:{type:'array',minItems:1,maxItems:128,items:{type:'object',properties:{id:{type:'integer'},curie:{type:'string',description:'Absolute class IRI'},label:{type:'string'},parents:{type:'array',items:{type:'integer'}},disjointWith:{type:'array',items:{type:'integer'}},synonyms:{type:'array',items:{type:'string'}}},required:['id','curie','label','parents','disjointWith','synonyms'],additionalProperties:false}}},required:['state','concepts'],additionalProperties:false}},
 {name:'olaya_read_reference',description:'Read a packaged skill or reference by its listed resource path.',inputSchema:{type:'object',properties:{path:{type:'string'}},required:['path'],additionalProperties:false}}
];
function files(dir:string):string[]{return readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?files(resolve(dir,e.name)):[resolve(dir,e.name)]);}
function resources(){return files(resolve(root,'skills')).map(p=>({uri:'olaya://bundle/'+relative(root,p).split('\\').join('/'),name:relative(root,p).split('\\').join('/'),mimeType:p.endsWith('.md')?'text/markdown':'application/json'}));}
function reference(path:string){const entry=resources().find(r=>r.name===path);if(!entry)throw Error('Unknown bundled reference');return readFileSync(resolve(root,path),'utf8');}
function validate(concepts:any):Concept[]{
 if(!Array.isArray(concepts)||concepts.length<1||concepts.length>128)throw Error('Supply 1 to 128 classes');
 const ids=new Set(),iris=new Set();
 for(const c of concepts){if(!c||!Number.isSafeInteger(c.id)||typeof c.curie!=='string'||! /^(https?:\/\/[^\s]+|urn:[^\s:]+:[^\s]+)$/.test(c.curie)||typeof c.label!=='string'||!c.label.trim()||c.label.length>1024||ids.has(c.id)||iris.has(c.curie))throw Error('Invalid or duplicate class ID/IRI'); ids.add(c.id);iris.add(c.curie);for(const field of ['parents','disjointWith','synonyms'])if(!Array.isArray(c[field])||c[field].length>128)throw Error('Invalid class arrays');if(!c.synonyms.every((s:any)=>typeof s==='string'&&s.length<=1024))throw Error('Invalid synonyms');}
 const byId=new Map(concepts.map(c=>[c.id,c]));
 for(const c of concepts)for(const id of [...c.parents,...c.disjointWith])if(!Number.isSafeInteger(id)||!ids.has(id)||id===c.id)throw Error('Invalid class reference');
 const visiting=new Set(),visited=new Set(); function visit(id:number){if(visiting.has(id))throw Error('Cyclic class hierarchy');if(visited.has(id))return;visiting.add(id);for(const p of (byId.get(id) as Concept).parents)visit(p);visiting.delete(id);visited.add(id);}for(const c of concepts)visit(c.id);
 return concepts.map(c=>({id:c.id,curie:c.curie,label:c.label,parents:c.parents,disjointWith:c.disjointWith,synonyms:c.synonyms}));
}
async function call(name:string,args:any){
 if(name==='olaya_capabilities')return caps;
 if(name==='olaya_read_reference')return {path:args.path,text:reference(args.path)};
 if(name!=='olaya_ground')throw Error('Unknown tool');
 if(typeof args.state!=='string'||!args.state.trim()||args.state.length>16384)throw Error('Supply nonempty bounded text');
 const concepts=validate(args.concepts); const decision=await new OLaya(new OntologyDAG(concepts)).systemOne(args.state,concepts.length);
 const hash=createHash('sha256').update(JSON.stringify(concepts)).digest('hex');
 return {backend:caps.backend,trained:false,decision,jsonld:{'@context':{o:'urn:olaya:desktop:',prov:'http://www.w3.org/ns/prov#',predictedClass:{'@id':'urn:olaya:desktop:predictedClass','@type':'@id'}},'@id':'urn:uuid:'+randomUUID(),'@type':['o:LexicalObservation','prov:Entity'],predictedClass:decision.selectedConcept.curie,'o:tboxSnapshotSha256':hash,'o:backend':caps.backend,'o:heuristicConfidence':decision.confidence,'o:axiomsVerified':decision.axiomsVerified}};
}
async function handle(q:any){
 if(q.jsonrpc!=='2.0'||typeof q.method!=='string')throw Error('Invalid request');
 const p=q.params??{};
 if(q.id===undefined)return null;
 let result:any;
 switch(q.method){
 case 'initialize':result={protocolVersion:['2024-11-05','2025-03-26','2025-06-18','2025-11-25'].includes(p.protocolVersion)?p.protocolVersion:'2025-11-25',capabilities:{tools:{},resources:{},prompts:{}},serverInfo:{name:'olaya-desktop',version:'0.1.0'},instructions:JSON.stringify(caps)};break;
 case 'ping':result={};break;
 case 'tools/list':result={tools};break;
 case 'tools/call':try{const data=await call(p.name,p.arguments??{});result={content:[{type:'text',text:JSON.stringify(data)}],structuredContent:data};}catch(e:any){result={isError:true,content:[{type:'text',text:e.message}]};}break;
 case 'resources/list':result={resources:resources()};break;
 case 'resources/read':{const r=resources().find(r=>r.uri===p.uri);if(!r)throw Error('Unknown resource');result={contents:[{uri:r.uri,mimeType:r.mimeType,text:reference(r.name)}]};break;}
 case 'prompts/list':result={prompts:skills.map(name=>({name,description:'Load the bundled '+name+' skill'}))};break;
 case 'prompts/get':if(!skills.includes(p.name))throw Error('Unknown prompt');result={messages:[{role:'user',content:{type:'text',text:reference('skills/'+p.name+'/SKILL.md')}}]};break;
 default:return {jsonrpc:'2.0',id:q.id,error:{code:-32601,message:'Method not found'}};
 }
 return {jsonrpc:'2.0',id:q.id,result};
}
const rl=createInterface({input:process.stdin,terminal:false});
let pending=Promise.resolve(); rl.on('line',line=>{pending=pending.then(async()=>{let q:any;try{if(Buffer.byteLength(line)>262144)throw Error('Request too large');q=JSON.parse(line);const r=await handle(q);if(r)process.stdout.write(JSON.stringify(r)+'\n');}catch(e:any){process.stdout.write(JSON.stringify({jsonrpc:'2.0',id:q?.id??null,error:{code:q?-32602:-32700,message:q?e.message:'Invalid JSON'}})+'\n');}});});
