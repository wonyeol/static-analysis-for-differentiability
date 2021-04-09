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
 ** entry_diff.ml: entry point for continuity/differentiability analysis *)
open Analysis_sig
open Ai_diff

(** Main function for diff analysis:
 ** - by default, tries to process the guides and models of all the
 **   main Pyro examples
 ** - override mode: analyze a single example *)
let main () =
  let fo:   string option ref = ref None
  and dnum: ad_num ref        = ref AD_sgn
  and verb: bool ref          = ref false
  and goal: diff_prop ref     = ref Diff in
  let dnum_set v = Arg.Unit (fun () -> dnum := v) in
  let goal_set g = Arg.Unit (fun () -> goal := g) in
  Arg.parse
    [ "-ai-box",   dnum_set AD_box,    "Num analysis, Apron, Boxes" ;
      "-ai-oct",   dnum_set AD_oct,    "Num analysis, Apron, Octagons" ;
      "-ai-pol",   dnum_set AD_pol,    "Num analysis, Apron, Polyhedra" ;
      "-ai-sgn",   dnum_set AD_sgn,    "Num analysis, Basic, Signs" ;
      "-verb",     Arg.Set verb,       "Turns on verbose output" ;
      "-lips",     goal_set Lips,      "Analysis for Lipschitzness" ;
      "-diff",     goal_set Diff,      "Analysis for Differentiability" ;
    ] (fun s -> fo := Some s) "Differentiability analysis";
  let l =
    match !fo with
    | Some x -> x
    | None -> failwith "no file given" in
  analyze !dnum !goal l !verb
let _ = ignore (main ( ))
