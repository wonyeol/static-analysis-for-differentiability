(** pyppai: basic abstract interpreter for python probabilistic programs
 **
 ** GNU General Public License
 **
 ** Authors:
 **  Wonyeol Lee, KAIST
 **  Xavier Rival, INRIA Paris
 **  Hongseok Yang, KAIST
 **  Hangyeol Yu, KAIST
 **
 ** Copyright (c) 2019 KAIST and INRIA Paris
 **
 ** ir_parse.ml: entry point for continuity/differentiability analysis *)
open Analysis_sig
open Lib

(** General debug *)
let output = ref true

(** General parsing/processing function *)
let parse_code (i: int option) (input: analysis_input): Ir_sig.prog =
  (* get python_code *)
  let python_code : string =
      match input with
      | AI_pyfile fname -> read_file fname
      | AI_pystring str -> str in
  (* construction of the Py.Object.t using Pyml *)
  let pyobj = Pyobj_util.get_ast python_code in
  (* construction of the Pyast AST *)
  let pyast = Pyast_cast.pyobj_to_modl pyobj in
  (* inline function definition of main *)
  let pyast = Pyast_util.inline_funcdef ["main"] pyast in
  (* construction of the IR AST *)
  let irast = Ir_cast.modl_to_prog pyast in
  if !output then
    begin
      let s =
        match i with
        | None -> ""
        | Some i -> Printf.sprintf "(%d)" i in
      Printf.printf "[IR%s]\n%a\n" s Ir_util.pp_prog irast
    end;
  irast
