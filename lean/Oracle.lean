import Lean.Data.Json
import OlayaAxioms

/-!
Executable oracle over the verified definitions. Reads one JSON object on stdin:
`{"values": [int...], "parent_edges": [[child, parent]...], "disjoint_edges": [[a, b]...],
"threshold": int}` and prints `{"projected": [int...], "verified": bool}` computed by
`OlayaAxioms.project` / `OlayaAxioms.verifyAxioms`. `tests/test_lean_oracle.py` drives it against
the Python reference on integer inputs. Exit 2 means the lifting fuel ran out.
-/

open Lean OlayaAxioms

def readPairs (j : Json) : Except String (List (Nat × Nat)) := do
  let arr ← j.getArr?
  arr.toList.mapM fun e => do
    let a ← (← e.getArrVal? 0).getNat?
    let b ← (← e.getArrVal? 1).getNat?
    pure (a, b)

structure Input where
  values : List Int
  edges : List Edge
  pairs : List (Nat × Nat)
  threshold : Int

def parseInput (s : String) : Except String Input := do
  let j ← Json.parse s
  let vals ← (← j.getObjVal? "values").getArr?
  let values ← vals.toList.mapM Json.getInt?
  let edges ← readPairs (← j.getObjVal? "parent_edges")
  let pairs ← readPairs (← j.getObjVal? "disjoint_edges")
  let threshold ← (← j.getObjVal? "threshold").getInt?
  pure { values, edges, pairs, threshold }

partial def readAll (h : IO.FS.Stream) (acc : String) : IO String := do
  let line ← h.getLine
  if line.isEmpty then pure acc else readAll h (acc ++ line)

def main : IO UInt32 := do
  let input ← readAll (← IO.getStdin) ""
  match parseInput input with
  | .error e => IO.eprintln s!"invalid input: {e}"; pure 1
  | .ok inp =>
    let k := inp.values.length
    let p : Values := ⟨fun i => inp.values.getD i 0, k⟩
    let fuel := k * k + 2
    match project fuel inp.edges inp.pairs inp.threshold p with
    | none => IO.eprintln "fuel exhausted"; pure 2
    | some q =>
      let projected := (List.range k).map q.f
      let out := Json.mkObj
        [("projected", Json.arr (projected.toArray.map toJson)),
         ("verified", toJson (verifyAxioms inp.edges inp.pairs inp.threshold q))]
      IO.println out.compress
      pure 0
