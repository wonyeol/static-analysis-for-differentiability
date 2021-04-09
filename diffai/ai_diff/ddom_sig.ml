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
 ** ddom_sig.ml: domains signatures for continuity/differentiability analysis *)
open Ir_sig

type constr = C_Pos | C_Neg | C_Num

let pp_constr chan = function
  | C_Pos -> Printf.printf "Pos"
  | C_Neg -> Printf.printf "Neg"
  | C_Num -> Printf.printf "Num" (* means numbers or contains of numbers *)

module type DOM_NUM =
  sig
    val name: string
    (* Abstraction of the numerical state (variables -> values) *)
    type t
    (* Prtty-printing *)
    val pp: out_channel -> t -> unit
    (* Bottom check: when returns true, definitely bottom *)
    val is_bot: t -> bool
    (* Lattice operations *)
    val top: t
    val join: t -> t -> t
    val equal: t -> t -> bool
    (* Post-condition for assignments *)
    val forget: string -> t -> t
    val assign: string -> expr -> t -> t
    val heavoc: string -> constr -> t -> t
    (* Condition tests *)
    val guard: expr -> t -> t
    (* Operation on primitive-function call x=f(el) *)
    val call_prim: string (* x *) -> string (* f *) -> expr list (* el *) -> t -> t
    (* Operation on object call x=(c())(el) where c is an object constructor *)
    val call_obj: string (* x *) -> string (* c *) -> expr list (* el *) -> t -> t
    (* Operations on distributions *)
    val sample: string -> dist -> t -> t
    val check_dist_pars: dist_kind -> expr list -> t -> bool list option
    (* Approximate implication check. false means unknown *)
    val imply: t -> expr -> bool 
  end
