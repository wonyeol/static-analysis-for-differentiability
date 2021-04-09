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
 ** ddom_num.mli: numerical domains for continuity/differentiability analysis *)
open Ddom_sig

(** Numerical domain based on constants *)
module DN_signs: DOM_NUM

(** Apron domain instances *)
module DN_box: DOM_NUM
module DN_oct: DOM_NUM
module DN_pol: DOM_NUM
