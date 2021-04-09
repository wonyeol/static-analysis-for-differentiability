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


(** Utilities for printing maps *)
val ppm: (out_channel -> 'a -> unit) -> out_channel -> 'a SM.t -> unit

(** Utilities for option type *)
val bind_opt: 'a option -> ('a -> 'b option) -> 'b option

(** Utilities for the operations in the diff domain *)
val map_join_union: ('a -> 'a -> 'a)
  -> 'a SM.t -> 'a SM.t -> 'a SM.t
val map_join_inter: ('a -> 'a -> 'a option)
  -> 'a SM.t -> 'a SM.t -> 'a SM.t
val map_equal: (string -> 'a -> bool) -> ('a -> 'a -> bool)
  -> 'a SM.t -> 'a SM.t -> bool
val lookup_with_default: string -> 'a SM.t -> 'a -> 'a
