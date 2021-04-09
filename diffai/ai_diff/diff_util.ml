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
 ** diff_util.ml: utilities for the differentiability analysis *)
open Lib


(** Utilities for the printing of results *)
let ppm pp chan m =
  SM.iter (fun s -> Printf.fprintf chan "\t%-30s\t=>\t%a\n" s pp) m

(** Utilities for handling options *)
let bind_opt x f =
  match x with
  | None -> None
  | Some x0 -> f x0

(** Utilities for the operations in the diff domain *)
let map_join_union join m0 m1 =
  SM.fold
    (fun v0 c0 acc ->
      try SM.add v0 (join c0 (SM.find v0 m1)) acc
      with Not_found -> SM.add v0 c0 acc
    ) m0 m1
let map_join_inter join m0 m1 =
  SM.fold
    (fun v0 c0 acc ->
      try match join c0 (SM.find v0 m1) with
      | None -> acc
      | Some c -> SM.add v0 c acc
      with Not_found -> acc
    ) m0 SM.empty
let map_equal pred eq m0 m1 =
  let m0 = SM.filter pred m0 in
  let m1 = SM.filter pred m1 in
  let ck v c0 = try eq c0 (SM.find v m1) with Not_found -> false in
  SM.cardinal m0 = SM.cardinal m1 && SM.for_all ck m0
let lookup_with_default (k: string) (m: 'a SM.t) (default: 'a) : 'a =
  try SM.find k m with Not_found -> default
