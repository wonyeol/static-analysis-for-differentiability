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
 ** ai_diff.mli: entry point for continuity/differentiability analysis *)
open Analysis_sig
open Lib

(** Differentiability-related properties. *)
(* type for properties related to differentiability *)
type diff_prop = Diff | Lips | Top
(* order on diff_prop *)
val diff_prop_leq: diff_prop -> diff_prop -> bool

(** Analysis main function wrapper.
 ** Inputs:
 ** - a domain.
 ** - a differentiability-related property `dp` to analyse.
 ** - a file name.
 ** - a flag for verbose print.
 ** Outputs:
 ** - set of parameters w.r.t which density is `dp`.
 ** - set of parameters w.r.t which density may not be `dp`. *)
val analyze: ad_num -> diff_prop -> string -> bool -> SS.t * SS.t
