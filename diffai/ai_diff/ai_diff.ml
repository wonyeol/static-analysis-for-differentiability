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
 ** ai_diff.ml: entry point for continuity/differentiability analysis *)
open Analysis_sig
open Ir_sig
open Ir_util
open Lib

open Ddom_sig
open Ddom_num

open Diff_util


(** ASSUMPTIONS on model and guide passed to our analyser:
 *
 * 1. Support of model and guide:
 *    The support of a guide is a subset of a model, i.e., any sampled value from a guide
 *    has a non-zero probability density in a model.
 *    - If this assumption does not hold, ELBO may not be well-defined.
 *    - This assumption also ensures that any sampled value of a latent variable
 *      in a guide is in the support of the corresponding latent variable in a model.
 *      (It is obvious that the sampled value is in the support of its latent variable
 *      in a guide.) So, it is sound to consider differentiability over the support of
 *      each distribution, and not over a superset of the support.
 *
 * 2. Observed values:
 *    For each statement `pyro.sample(_, d, obs=v)`, (i) the observed value v is a constant
 *    and (ii) the probability density of the observing distribution d at v is non-zero.
 *    - (i) is a common assumption. If not satisfied (e.g., v depends on latent variables
 *      or parameters to be trained), then unexpected behavior may arise.
 *    - (ii) guarantees that v is in the support of d. So, it is again sound to consider
 *      differentiability over the support of each distribution.
 *    - NOTE: guaranteed by our analyser, not by assumptions, is that parameters of
 *            each distribution for `sample` or `observe` are in their supposed domain.
 *
 * 3. Uniform distributions:
 *    All parameters of all Uniform distributions are constant, i.e., do not depend on
 *    any of the parameters to be trained, nor on any of the other latent variables.
 *    - If this assumption does not hold, selective reparameterisation can be biased.
 *      For details, refer to `whitebox/examples/pyro/BiasedGradWithUniform.ipynb`.
 *
 * 4. Delta distributions:
 *    (i) A model is not allowed to contain Delta distributions. (ii) A guide can contain
 *    Delta distributions, but assume that every `pyro.sample(_, Delta(v))` is replaced
 *    by `v` before starting our analyser.
 *    - (i) is assumed for brevity. We may consider relaxing it.
 *    - (ii) is sound since taking an expectation over Delta(z;v) is the same as
 *      substituting z with v inside the expectation.
 *    - If the substitution in (ii) is not performed, our analysis result could be
 *      unsound. For instance, consider
 *        guide_1 := C; z ~ Delta(theta), and
 *        guide_2 := C; z ~ Delta(theta); if z > 0 then score(1) else score(2).
 *      Suppose that the density of C is differentiable w.r.t. theta. Then,
 *        (a) density of guide_1 after substituting z with theta is differentiable
 *            w.r.t. theta; and
 *        (b) density of guide_2 after substituting z with theta is not differentiable
 *            w.r.t. theta.
 *      Now suppose that we pass guide_{1,2} to our analyser without performing
 *      the substitution in (ii). To make our analyser produce (a), `diff_dk_sig(Delta)`
 *      should be `_,[true]`, not `_,[false]`. However, this makes our analyser conclude
 *      the negation of (b), since the non-differentiability coming from the branch on z
 *      cannot lead to (b) since the data dependency on theta does not flow into z.
 *      In sum, to guarantee the soundness of our analyser, the substitution in (ii)
 *      should be performed before our analyser is called.*)


(** Differentiability-related properties. *)
(* type for properties related to differentiability:
 *   Diff: differentiable w.r.t. some parameters S.
 *   Lips: locally Lipschitz w.r.t. some parameters S.
 *   Top:  always true. *)
type diff_prop = Diff | Lips | Top

(* order on diff_prop:
 *   We give the following order based on implication:
 *   dp1 <= dp2 <==> for any func f, if f is dp1 w.r.t. S, then f is dp2 w.r.t. S. *)
let diff_prop_leq (dp1: diff_prop) (dp2: diff_prop): bool =
  match dp1, dp2 with
  | _, _ when dp1 = dp2 -> true
  | Diff, Lips | Diff, Top | Lips, Top -> true
  | _ -> false


(** Data for analysis. *)
(* Config vars
 * - dp_goal: property to be analyzed.
 *   - ASSUME: it is not Top and is initialised by `analyze ...`.
 * - debug: global debug. *)
let dp_goal: diff_prop ref = ref Lips
let debug: bool ref = ref false
let printf_debug (f: ('a, out_channel, unit) format): 'a =
  if !debug then Printf.printf f
  else           Printf.ifprintf stdout f

(* SPECIFICATION: Maps each distribution into a tuple:
 *  1. Boolean indicating whether its density is continuously differentiable
 *     w.r.t. a sampled value over its domain
 *     - This boolean is used to determine which sampled variables are reparameterisable.
 *     - For discrete distributions, the boolean is false, since we cannot
 *       differentiate its density w.r.t. a discrete sampled value.
 *
 *  2. List of booleans indicating whether its density is continuously differentiable
 *     w.r.t. each parameter (of distribution) over its domain
 *     - These booleans are used to determine which sampled variables and
 *       parameters (not of distribution, but to be trained) are reparameterisable.
 *     - For discrete parameters (of distribution), the boolean is false
 *       since we cannot differentiate its density w.r.t. a discrete parameter.
 *
 *  3. NOTE: THIS IS CURRENTLY NOT IN USE.
 *     The subsets of the domains of parameters
 *     (i.e., some values should be positive, and the violation
 *     of this domain constraint may lead to non-differentiability
 *     due to altered control flow).
 *     WL: why is this part commented out? where do we check whether
 *         each parameter is in the supposed domain? E.g., sigma of Normal is in R_{>0},
 *         and a and b of Uniform is in {(a,b) in R^2 : a<b}.
 *
 *  REMARKS:
 *  - Precise definition of continuous differentiability is as follows.
 *    Consider a density f(z; a1, ..., an) of a distribution. Let V_cont \subseteq
 *    {z, a1, ..., an} be the set of continuous variables (i.e., variables whose domain
 *    is a union of non-empty open sets in R^|V_cont|), and V_disc =  {z, a1, ..., an} \ V_cont
 *    be the set of non-continuous variables. Let D_cont and D_disc be the domain of
 *    V_cont and V_disc, respectively.
 *
 *    We give some examples on the domain of a density:
 *    - the density of Normal(z; m, s) has the domain {(z, m, s) \in R x R x R_{>0}};
 *    - the density of Exponential(z; r) has the domain {(z, r) \in R_{>=0} x R_{>0}};
 *    - the density of Uniform(z; a, b) has the domain {(z, a, b) \in R x R x R | a <= z < b};
 *    - the probability mass of Poisson(z; l) has the domain {(z, l) \in Z_{>=0} x R_{>0}};
 *    - the probability mass of Delta(z; v) has the domain {(z, v) \in R x R | z = v}.
 *
 *    For any V \subset V_cont, we say d is continuously differentiable w.r.t. V
 *    iff f[V_disc:v_disc] : D_cont (\subseteq R^|V_cont|) -> R is  continuously
 *    differentiable w.r.t. V for all v_disc \in D_disc. Here f[V_disc:v_disc] denotes
 *    the function obtained from f by fixing the value of V_disc to v_disc.
 *
 *  - We consider continuous differentiability instead of mere differentiability
 *    since the former is variable-monotone but the latter is not.
 *
 *  - Special care is required for boolean values of Uniform and Delta, as the support of
 *    Uniform depends on its parameters, and the support of Delta is a singleton set.
 *    To guarantee the unbiasedness of selective reparameterisation, we put some
 *    ASSUMPTIONS on Uniform and Delta (see the top).
 *
 *  WARNING:
 *  - The distribution constructors are usually applied in the broadcasting mode
 *    during sample and observe. The diff_dk_sig function does not consider the use
 *    of broadcasting. The user of the function should take care of it. *)
let diff_dk_sig = function
  | Normal              -> true , [true ; true ] (*, [ SD.Top ; SD.Plus ]*)
  | Exponential         -> true , [true ]        (*, [ SD.Plus ]*)
  | Gamma               -> true , [true ; true ] (*, [ SD.Plus ; SD.Plus ]*)
  | Beta                -> true , [true ; true ] (*, [ SD.Plus ; SD. Plus ]*)
  | Dirichlet _         -> true , [true ]        (*, [ SD.Plus (* (0,inf)^n for n >= 2 *)] *)
  | Poisson             -> false, [true ]        (*, [ SD.Plus (* (0,inf) *)]*)
  | Categorical _       -> false, [true ]        (*, [ SD.Plus (* (0,inf)^n for n >= 2 *)] *)
  | Bernoulli           -> false, [true ]        (*, [ SD.CInter (* [0,1] *)]*)
  | OneHotCategorical _ -> false, [true ]        (*, [ SD.Plus (* (0,inf)^n for n >= 2 *)]*)
  | Subsample (_, _)    -> false, [false; false] (*, [ SD.Bot ; SD.Bot ]*)
  (* Uniform and Delta require special care. *)
  | Uniform _           -> true , [true ; true ] (*, [ SD.Bot ; SD.Bot ] (* 1st param < 2nd param *)*)
  | Delta               -> false, [false]        (*, [ SD.Top ]*)

(* Table storing differentiability information about functions.
 * - None means no information.
 * - Some[b1;b2;...] keeps information about the continuous differentiability of the function.
 *   b_i indicates whether the function is continuously differentiable w.r.t. its i-th argument. *)
let diff_funct_sig v =
  match v with
  (*
   * Functions that return a tensor or float.
   *)
  | "torch.arange"     (* args = ([start,] end [, step]) *)
  | "torch.ones"       (* args = ( *shape ) *)
  | "torch.eye"        (* args = ( n ) *)
  | "torch.rand"       (* args = ( *shape ) *)
  | "torch.randn"      (* args = ( *shape ) *)
  | "torch.zeros"      (* args = ( *shape ) *)
  | "torch.LongTensor" (* args = ( data ). performs rounding operations. *)
    -> Some [ ]

  | "access_with_index"      (* args = (tensor, index) *)
  | "float"                  (* args = (data) *)
  | "torch.cat"              (* args = (tensors[, dim]) *)
  | "torch.exp"              (* args = (tensor) *)
  | "torch.index_select"     (* args = (tensor, dim, index) *)
  | "torch.reshape"          (* args = (tensor, shape) *)
  | "torch.sigmoid"          (* args = (tensor) *)
  | "torch.squeeze"          (* args = (tensor[, dim]) *)
  | "torch.tensor"           (* args = (data) *)
  | "torch.transpose"        (* args = (tensor, dim0, dim1) *)
  | "torch.FloatTensor"      (* args = (data) *)
  | "torch.Tensor.detach"    (* args = (tensor) *)
  | "torch.Tensor.expand"    (* args = (tensor, *shape) *)
  | "torch.Tensor.transpose" (* args = (tensor, dim0, dim1) *)
  | "torch.Tensor.view"      (* args = (tensor, *shape) *)
  | "F.affine_grid"          (* args = (tensor, shape) *)
  | "F.softplus"             (* args = (tensor) *)
    -> Some [ true ]

  | "F.relu"                 (* args = (tensor) *)
    -> (match !dp_goal with
        | Diff -> Some [ false ]
        | Lips -> Some [ true  ]
        | _    -> failwith "error")

  | "torch.matmul"           (* args = (tensor, tensor) *)
    -> Some [ true; true ]

  | "F.grid_sample"          (* args = (input:tensor, grid:tensor [, mode]) *)
     (* WARNING:
      * - `mode` can be either "bilinear" or "nearest"; its default value is "bilinear".
      * - For `mode`="bilinear":
      *   - for differentiability, `Some[true;false]` is sound, but `Some[true;true]` is unsound;
      *   - for Lipschitzness, `Some[true;true]` is sound.
      * - For `mode`="nearest":
      *   - for both differentiability and Lipschitzness,
      *    `Some[true;false]` is sound, but `Some[true;true]` is unsound.
      * - Hence, when `mode`="nearest" and our analyser checks Lipschitzness,
      *   using `Some[true;true]` could produce unsound analysis results. *)
    -> (match !dp_goal with
        | Diff -> Some [ true; false ]
        | Lips -> Some [ true; true  ]
        | _    -> failwith "error")

  | "update_with_field"      (* args = (src, field_name, new_value) *)
  | "update_with_index"      (* args = (src, indices, new_value) *)
    -> Some [ true; false; true ]

  | "torch.Tensor.scatter_add_" (* args = (dim, index, tensor) *)
    -> Some [ false; false; true ]

  (*
   * Functions that return an object, or receives an object as an argument.
   *
   * Note: If a function returns an object, we say that the object is differentiable with respect to
   *       its parameters. The functions and methods invoked on the object should then revise
   *       this default decision if their outcomes are not differentiable with respect to
   *       the parameters used to create their argument objects. This convention is applied to
   *       our handling of Categorical and Categorical.log_prob.
   * Note: There is only one example that requires the extended notion of differentiability
   *       described above: `whitebox/refact/test/pyro_example/lda_guide2.py`.
   *       Since this example is not included in our final benchmarks, we do not need to consider
   *       this extended notion of differentiability, and using `Some[]` for the following cases
   *       would still produce desired analysis results for our final benchmarks.
   *)
  | "Categorical"          (* args = (tensor) *)
  | "Categorical.log_prob" (* args = (distribution, tensor) *)
    -> Some [ true ]

  (*
   * All the functions below are handled in the most imprecise manner;
   * it may be possible to very easily improve on them based on their
   * semantics.
   *)
  (* Functions that may return non-tensor and non-float objects. *)
  (* --- python-related *)
  | "dict"
  | "range"
  (* --- torch-related*)
  | "nn.Linear"
  | "nn.LSTMCell"
  | "nn.Parameter"
  | "nn.ReLU"
  | "nn.RNN"
  | "nn.Sigmoid"  (* -> Some [ true ] *)
  | "nn.Softmax"  (* -> Some [ true ] *)
  | "nn.Softplus" (* -> Some [ true ] *)
  | "nn.Tanh"     (* -> Some [ true ] *)
  | "RYLY[constraints.positive]"
  | "RYLY[constraints.negative]"
  (* --- pyro-related*)
  | "pyro.plate"
  | "pyro.poutine.scale"
  (* --- our ir-related *)
  | "RYLY"
    (* nothing known; will assume non differentiable in all args *)
    -> Some [ ]

  (* Method calls *)
  | "decoder_fst.bias.data.normal_"
  | "decoder_fst.weight.data.normal_"
  | "encoder_y_fst.bias.data.normal_"
  | "encoder_y_fst.weight.data.normal_"
  | "encoder_z_fst.bias.data.normal_"
  | "encoder_z_fst.weight.data.normal_"
  | "layer1.bias.data.normal_"
  | "layer1.weight.data.normal_"
  | "layer2.bias.data.normal_"
  | "layer2.weight.data.normal_"
  | "layer3.bias.data.normal_"
  | "layer3.weight.data.normal_"
  | "z_pres.append"
  | "z_where.append"
    (* nothing known; will assume non differentiable in all args *)
    -> Some [ ]

  (* All the rest *)
  | _
    -> Printf.printf "TODO,function: %S\n" v; None

(* Table storing differentiability information about function-returning function.
 * - None means no information.
 * - Some(b,l) keeps information about the continuous differentiability of
 *   the returned function. b indicates whether the function is continuously differentiable
 *   w.r.t. all implicit parameters kept by the function (if such parameters exist).
 *   l stores continuous differentiability information w.r.t. the function arguments. *)
let diff_functgen_sig f args kwargs =
  match f with
  | "nn.Linear"
  | "nn.Sigmoid"
  | "nn.Softmax"
  | "nn.Softplus"
  | "nn.Tanh"
    -> Some (true, [true])
  | "nn.ReLU"
    -> (match !dp_goal with
        | Diff -> Some (true, [false])
        | Lips -> Some (true, [true ])
        | _    -> failwith "error")
  | "nn.LSTMCell"
    -> Some (true, [true; true])
  | "nn.RNN"
    -> begin
      try
        match (List.find (fun x -> (fst x) = Some "nonlinearity") kwargs) with
        | _, Str "tanh"
          -> Some (true, [true; true])
        | _, Str "relu"
          -> (match !dp_goal with
              | Diff -> Some (false, [false; false])
              | Lips -> Some (true , [true ; true ])
              | _    -> failwith "error")
        | _ -> failwith "diff_functgen_sig, nn.RNN: unreachable."
      with
      | Not_found (* Same as "tanh" since default "nolinearity" is "tanh". *)
        -> Some (true, [true; true])
    end
  | _
    -> None


(* WL: Checked up to here. *)
(** Computation for analysis. *)
module Make = functor (DN: DOM_NUM) ->
  struct

    (* Types for objects, such as constraint objects.
     * These objects should behave as if they were pure (i.e. update-free)
     * objects from the perspective of the analysis. This means that they
     * are indeed pure objects, or the information recorded in a type
     * is invariant with respect to all possible updates. *)
    type obj_t =
      | O_Constr of constr
      | O_Dist of string
      | O_Fun of string
      | O_Nil

    let pp_obj_t chan = function
      | O_Constr(c) ->  Printf.printf "ConstrObj[%a]" pp_constr c
      | O_Dist(d) ->  Printf.printf "DistObj[%s]" d
      | O_Fun(f) ->  Printf.printf "FunObj[%s]" f
      | O_Nil ->  Printf.printf "NoneObj"

    type t =
        { (* "Parameters":
           * Parameters with respect to which we track differentiability *)
          t_pars:    SS.t ;
          (* "Guard Parameters"
           * Parameters that are guarding (* WL: may guard? *) the current path;
           * => we cannot guarantee anything is differentiable wrt those *)
          t_gpars:   SS.t ;
          (* "Variables-Parameter-Non-Diff":
           * Maps each variable to parameters with respect to which it may not
           * be differentiable. If a variable x stores a function f that may
           * depend on parameters, we say that it is differentiable wrt.
           * p if f(v) is differentiable wrt. p for all v.
           * (* WL: Need the last line? In all other parts, we implicitly
           *  * assume that
           *  * "f : X x Y -> R is differentiable wrt x" means
           *  * "f(_,y) is differentiable wrt x for all y",
           *  * which is exactly the same as what the last line says. *) *)
          t_vpndiff: SS.t SM.t ;
          (* "Variables-Parameter-Dependencies":
           * Maps each variable to parameters that it may depend on *)
          t_vpdep:   SS.t SM.t ;
          (* "Density-Parameter-Non-Diff":
           * Set of parameters for which density may not be differentiable *)
          t_dpndiff: SS.t ;
          (* "Variable-Function-Information":
           * Maps each variable to information about a function object
           * stored in the variable.
           * All unmapped variables are implicitly mapped to top, the lack
           * of any information. (* WL: is there top here? *)
           * The stored information is a pair of a boolean and a list.
           * The boolean describes the differentiability with respect to
           * implicit parameters to the function object, if such
           * parameters exist. Thus, if it is true, there are no implicit
           * parameters or the function is differentiable with respect to
           * such parameters. *)
          (* WL: More details. (b,l) ==>
           *   b = 1_[f is differentiable wrt all implicit params].
           *   l_i = 1_[f is differentiable wrt ith arg]. *)
          t_vfinfo:  (bool * bool list) SM.t ;
          (* "Variable-Object-Information":
           * Maps each variable to information about the object stored in the
           * variable. *)
          t_voinfo:  obj_t SM.t ;
          (* "Variable-Numerical predicates":
           * Numerical predicates over variables expressed in parameter
           * domain DN.
           * IMPORTANT: If a variable is bounded to Top, it is numeric. *)
          t_vnum:    DN.t
        }
    let get_d_ndiff (t: t): SS.t = t.t_dpndiff
    let get_d_diff (t: t): SS.t = SS.diff t.t_pars t.t_dpndiff

    let pp chan t =
      let ppm_ss = ppm ss_pp in
      let ppm_fi = ppm (pp_pair pp_bool (pp_list pp_bool)) in
      let ppm_oi = ppm pp_obj_t in
      Printf.printf "pars: %a\npnd:\n%adepd:\n%agpars: %a\ndndiff: %a\n"
        ss_pp  t.t_pars   ppm_ss t.t_vpndiff ppm_ss t.t_vpdep
        ss_pp  t.t_gpars  ss_pp  t.t_dpndiff;
      Printf.printf "vfinfo:\n%avoinfo:\n%a%s:\n%a"
        ppm_fi        t.t_vfinfo
        ppm_oi        t.t_voinfo
        DN.name DN.pp t.t_vnum

    (* Wrappers for some numerical domain operations *)
    let is_bot t = DN.is_bot t.t_vnum
    let guard e t = { t with t_vnum = DN.guard e t.t_vnum }

    (* Lattice operations *)
    let t_union (acc0: t) (acc1: t): t =
      let ss_map_join m0 m1 =
        map_join_union SS.union m0 m1 in
      let info_map_join m0 m1 =
        map_join_inter
          (fun c0 c1 ->
            if c0 = c1 then Some c0
            else None) m0 m1 in
      { t_pars    = SS.union      acc0.t_pars    acc1.t_pars;
        t_gpars   = SS.union      acc0.t_gpars   acc1.t_gpars;
        t_vpndiff = ss_map_join   acc0.t_vpndiff acc1.t_vpndiff;
        t_vpdep   = ss_map_join   acc0.t_vpdep   acc1.t_vpdep;
        t_dpndiff = SS.union      acc0.t_dpndiff acc1.t_dpndiff;
        t_vfinfo  = info_map_join acc0.t_vfinfo  acc1.t_vfinfo;
        t_voinfo  = info_map_join acc0.t_voinfo  acc1.t_voinfo;
        t_vnum    = DN.join       acc0.t_vnum    acc1.t_vnum }

    let t_equal (acc0: t) (acc1: t): bool =
      let ss_map_equal m0 m1 =
        map_equal (fun v ss -> ss <> SS.empty) SS.equal m0 m1 in
      let info_map_equal m0 m1 =
        map_equal (fun v i -> true) (fun p1 p2 -> p1 = p2) m0 m1 in
      SS.equal            acc0.t_pars    acc1.t_pars
        && SS.equal       acc0.t_gpars   acc1.t_gpars
        && ss_map_equal   acc0.t_vpndiff acc1.t_vpndiff
        && ss_map_equal   acc0.t_vpdep   acc1.t_vpdep
        && SS.equal       acc0.t_dpndiff acc1.t_dpndiff
        && info_map_equal acc0.t_vfinfo  acc1.t_vfinfo
        && info_map_equal acc0.t_voinfo  acc1.t_voinfo
        && DN.equal       acc0.t_vnum    acc1.t_vnum

    (** Display of analysis results *)
    let pp_result (abs: t): unit =
      Printf.printf "Params found:\n\t%a\n" ss_pp abs.t_pars;
      Printf.printf "Non-differentiability of variables wrt parameters:\n";
      SM.iter
        (fun v pars ->
          Printf.printf "\t%-30s\t=>\t%a\n" v ss_pp pars
        ) abs.t_vpndiff;
      Printf.printf "Non-differentiability of density wrt parameters:\n\t%a\n"
        ss_pp abs.t_dpndiff;
      Printf.printf "Functions bound to variables:\n";
      SM.iter
        (fun v fi ->
          Printf.printf "\t%-30s\t=>\t%a\n" v (pp_pair pp_bool (pp_list pp_bool)) fi
        ) abs.t_vfinfo;
      Printf.printf "Objects bound to variables:\n";
      SM.iter
        (fun v oi ->
          Printf.printf "\t%-30s\t=>\t%a\n" v pp_obj_t oi
        ) abs.t_voinfo;
      Printf.printf "Numerical information<%s>:\n%a" DN.name DN.pp abs.t_vnum;
      Printf.printf "\n\n================================================\n\n\n"

    (** Analysis of expression *)
    type texp =
        { (* Parameters on which the expression may depend on *)
          te_pdep:   SS.t ;
          (* Parameters with respect to which the expression may depend on
           * and may be non-differentiable:
           * => in general it is always sound to make this field equal to
           *    te_pdep; *)
          te_pndiff: SS.t }

    (* Table storing information about object-returning function.
     * - None means no information.
     * - Some ot keeps information about the returned object *)
    let funct_obj_sig f args kwargs =
      match f with
      | "RYLY[constraints.positive]" ->
          Some(O_Constr(C_Pos))
      | "RYLY[constraints.negative]" ->
          Some(O_Constr(C_Neg))
      | "Categorical" ->
          Some(O_Dist(f))
      | "nn.Sigmoid" | "nn.Softmax" | "nn.Softplus" | "nn.ReLU" | "nn.Tanh" | "nn.Linear" ->
          Some(O_Fun(f))
      | _ ->
          None



    (** Utility functions *)
    let accumulate_guard_pars (accu: SS.t) (del: texp list)
        (ok_el: bool list option): SS.t =
      let default () =
        List.fold_left (fun acc de -> SS.union acc de.te_pdep) accu del in
      match ok_el with
      | None -> default ()
      | Some ok_el ->
          try
            let f acc ok de = if ok then acc else SS.union acc de.te_pdep in
            List.fold_left2 f accu ok_el del
          with Invalid_argument _ -> default ()

    let imply (acc: t) (e_in: expr): bool =
      let rec simplify e =
        match e with
        | Name x ->
            begin
              try
                match (SM.find x acc.t_voinfo) with
                | O_Nil -> Nil
                | _ -> e
              with Not_found -> e
            end
        | UOp (uop, e0) ->
            let e0_sim = simplify e0 in
            UOp (uop, e0_sim)
        | BOp (bop, e0, e1) ->
            let e0_sim = simplify e0 in
            let e1_sim = simplify e1 in
            BOp (bop, e0_sim, e1_sim)
        | Comp (cop, e0, e1) ->
            let e0_sim = simplify e0 in
            let e1_sim = simplify e1 in
            Comp (cop, e0_sim, e1_sim)
        | List es0 ->
            List (List.map simplify es0)
        | _ ->
            e in
      DN.imply acc.t_vnum (simplify e_in)

    (** Basic definitions of the analysis *)
    let diff_expr (acc: t) (e: expr): texp =
      let rec aux = function
        | Nil | True | False | Num _ | Str _ ->
            { te_pdep   = SS.empty ;
              te_pndiff = SS.empty }
        | Name x ->
            let pdep   = try SM.find x acc.t_vpdep   with Not_found -> SS.empty in
            let pndiff = try SM.find x acc.t_vpndiff with Not_found -> SS.empty in
            { te_pdep   = pdep ;
              te_pndiff = pndiff }
        | UOp (Not, e) ->
            let te = aux e in
            { te with
              te_pndiff = te.te_pdep }
        | BOp ((Add | Sub | Mult | Pow), e0, e1) ->
            (* numeric, differentiable cases *)
            let te0 = aux e0 and te1 = aux e1 in
            { te_pdep   = SS.union te0.te_pdep   te1.te_pdep ;
              te_pndiff = SS.union te0.te_pndiff te1.te_pndiff }
        | BOp (Div, e0, e1) ->
            (* numeric, partly differentiable:
             * - Div: discontinuity at 0 *)
            let te0 = aux e0 and te1 = aux e1 in
            let no_div0 = imply acc (Comp (NotEq, e1, Num (Float 0.))) in
            let te_pdep = SS.union te0.te_pdep te1.te_pdep in
            let te_pndiff =
              if no_div0 then SS.union te0.te_pndiff te1.te_pndiff
              else SS.union te0.te_pndiff te1.te_pdep in
            { te_pdep   = te_pdep ;
              te_pndiff = te_pndiff }
        | BOp ((And | Or), e0, e1)
        | Comp (_, e0, e1) ->
            (* comparison and boolean operators *)
            let te0 = aux e0 and te1 = aux e1 in
            let pdep = SS.union te0.te_pdep te1.te_pdep in
            { te_pdep   = pdep ;
              te_pndiff = pdep }
        | UOp ((SampledStr | SampledStrFmt), e0) ->
            let te0 = aux e0 in
            { te0 with
              te_pndiff = te0.te_pdep }
        | List el ->
            let tei = { te_pdep   = SS.empty ;
                        te_pndiff = SS.empty } in
            List.fold_left
              (fun a ep ->
                let tep = aux ep in
                { te_pdep   = SS.union tep.te_pdep   a.te_pdep ;
                  te_pndiff = SS.union tep.te_pndiff a.te_pndiff }
              ) tei el
        | StrFmt (_, el) ->
            let pdep =
              List.fold_left (fun a te -> SS.union a te.te_pdep)
                SS.empty (List.map aux el) in
            { te_pdep   = pdep ;
              te_pndiff = pdep }
        | e ->
            Printf.printf "TODO expression: %a\n" pp_expr e;
            { te_pdep   = acc.t_pars ;
              te_pndiff = acc.t_pars } in
      aux e

    let obj_expr (m : obj_t SM.t) = function
      | Name x ->
          begin
            try Some(SM.find x m) with Not_found -> None
          end
      | _ -> None

    let ndpars_call_args (acc: t) (fsig: bool list) (del: texp list): SS.t =
      let rec aux (fsig: bool list) (del: texp list): SS.t =
        match fsig, del with
        | diffarg :: fsig, d :: del ->
            let pn = aux fsig del in
            let pndiff =
              if diffarg then d.te_pndiff
              else d.te_pdep in
            SS.union pn pndiff
        | _ :: _, [ ] ->
            (* May not be differentiable at all *)
            acc.t_pars
        | [ ], d :: del ->
            let pn = aux [ ] del in
            SS.union pn d.te_pdep
        | [ ], [ ] ->
            SS.empty in
      aux fsig del

    let update_info_map
          (tbl: string -> expr list -> keyword list -> 'a option)
          (m: 'a SM.t) (x: string) (f: string)
          (args: expr list) (kwargs: keyword list): 'a SM.t =
      match tbl f args kwargs with
      | Some info -> SM.add x info m
      | None ->
          begin
            match f, args with
            | "update_with_field", Name y :: _ ->
                begin
                  try SM.add x (SM.find y m) m with Not_found -> SM.remove x m
                end
            | _ -> SM.remove x m
          end
    let update_finfo_map = update_info_map diff_functgen_sig
    let update_oinfo_map = update_info_map funct_obj_sig

    let has_no_obs (acc: t) = function
      | None -> true
      | Some o ->
          match (obj_expr acc.t_voinfo o) with
          | Some O_Nil -> true
          | _ -> false

    let get_obs (acc: t) obs_opt =
      let err_msg = "Should not be reached: sample statement confused as observe" in
      match obs_opt with
      | None -> failwith err_msg
      | Some o ->
          match (obj_expr acc.t_voinfo o) with
          | Some O_Nil -> failwith err_msg
          | Some _ | None -> o

    let do_acmd (acc: t) (ac: acmd): t =
      match ac with
      | AssnCall (_, Name "pyro.param", Str pname :: Name x :: _, kargs)
      | AssnCall (x, Name "pyro.param", Str pname :: _, kargs) ->
          let vnum =
            try
              let (_, e) =
                List.find (fun (k,_) -> k = Some("constraint")) kargs in
              match (obj_expr acc.t_voinfo e) with
              | Some (O_Constr(constr)) ->
                  DN.heavoc x constr acc.t_vnum
              | None | Some (O_Dist _) | Some (O_Fun _) | Some O_Nil ->
                  DN.heavoc x C_Num acc.t_vnum
            with Not_found -> DN.heavoc x C_Num acc.t_vnum in
          { acc with
            t_vpndiff = SM.add x SS.empty acc.t_vpndiff;
            t_vpdep   = SM.add x (SS.singleton pname) acc.t_vpdep;
            t_pars    = SS.add pname acc.t_pars;
            t_vfinfo  = SM.remove x acc.t_vfinfo;
            t_voinfo  = SM.remove x acc.t_voinfo;
            t_vnum    = vnum }
      | AssnCall (_, Name "pyro.param", _, _) ->
          Printf.eprintf "unbound-pyro.param: %a\n" pp_acmd ac;
          failwith "unbound-pyro.param"
      | AssnCall (_, Name "pyro.module", Str pname :: Name x :: _, _ ) ->
          let pndiff =
            let pndiff_old = lookup_with_default x acc.t_vpndiff SS.empty in
            let is_ipdiff,_ = lookup_with_default x acc.t_vfinfo (false,[]) in
            if is_ipdiff then pndiff_old else SS.add pname pndiff_old in
          let pdep =
            let pdep_old = lookup_with_default x acc.t_vpdep SS.empty in
            SS.add pname pdep_old in
          { acc with
            t_vpndiff = SM.add x pndiff acc.t_vpndiff;
            t_vpdep   = SM.add x pdep acc.t_vpdep;
            t_pars    = SS.add pname acc.t_pars }
      | AssnCall (_, Name "pyro.module", Str pname :: _, _ ) ->
          { acc with t_pars = SS.add pname acc.t_pars }
      | AssnCall (_, Name "pyro.module", _, _) ->
          Printf.eprintf "unbound-pyro.module: %a\n" pp_acmd ac;
          failwith "unbound-pyro.module"
      | AssnCall (x, Name v, el, kwargs) ->
          let del = List.map (diff_expr acc) el in
          (* All dependencies in the arguments *)
          let deppar_el =
            List.fold_left (fun a e -> SS.union a e.te_pdep) SS.empty del in
          let deppar_v =
            lookup_with_default v acc.t_vpdep SS.empty in
          let deppar =
            SS.union deppar_v (SS.union deppar_el acc.t_gpars) in
          (* Check if the return value is differentiable *)
          let pndiff =
            try
              let is_ipdiff, fsig = SM.find v acc.t_vfinfo in
              let pndiff_args = ndpars_call_args acc fsig del in
              if is_ipdiff then pndiff_args
              else SS.union deppar_v pndiff_args
            with Not_found ->
              match diff_funct_sig v with
              | None -> deppar
              | Some fsig -> ndpars_call_args acc fsig del in
          (* Non-differentiability with respect to guard parameters *)
          let pndiff = SS.union pndiff acc.t_gpars in
          (* Check if the return value is a function with
           * known information about its differentiability *)
          let finfo_map = update_finfo_map acc.t_vfinfo x v el kwargs in
          let oinfo_map = update_oinfo_map acc.t_voinfo x v el kwargs in
          let num =
            match (obj_expr acc.t_voinfo (Name v)) with
            | None | Some (O_Constr _) | Some (O_Dist _) | Some O_Nil ->
               DN.call_prim x v el acc.t_vnum
            | Some (O_Fun(f)) ->
               DN.call_obj x f el acc.t_vnum in
          { acc with
            t_vpndiff = SM.add x pndiff acc.t_vpndiff;
            t_vpdep   = SM.add x deppar acc.t_vpdep;
            t_vfinfo  = finfo_map;
            t_voinfo  = oinfo_map;
            t_vnum    = num }
      | AssnCall (_, _, _, _) ->
          Printf.printf "TODO,complex assncall: %a\n" pp_acmd ac;
          acc
      | Assert _
      | Assume _ ->
          acc
      | Assn (x, e) ->
          (* Non-differentiability information about the RHS expression *)
          let de = diff_expr acc e in
          (* Non-differentiability with respect to guard parameters *)
          let pdep   = SS.union de.te_pdep   acc.t_gpars in
          let pndiff = SS.union de.te_pndiff acc.t_gpars in
          (* Mark whether x may depend on paramters *)
          let oinfo =
            match e with
            | Nil -> SM.add x O_Nil acc.t_voinfo
            | _ -> SM.remove x acc.t_voinfo in
          { acc with
            t_vpndiff = SM.add x pndiff acc.t_vpndiff;
            t_vpdep   = SM.add x pdep acc.t_vpdep;
            t_vfinfo  = SM.remove x acc.t_vfinfo;
            t_voinfo  = oinfo;
            t_vnum    = DN.assign x e acc.t_vnum }
      | Sample (x (* x *), n (* S *), d (* Distr *), a (* E1, E2 *), o_opt (* Obs *))
        when has_no_obs acc o_opt ->
          let parname =
            match n with
            | Str s -> s
            | StrFmt (s, _) ->
                if false then Printf.printf "Sample formatter: %S\n" s;
                s
            | _ ->
                Printf.eprintf "unbound parameter expression: %a\n" pp_expr n;
                failwith "unbound" in
          (* Non-differentiability information about the current
           * distribution *)
          let dist_diff, fsig_diff = diff_dk_sig (fst d) in
          (* Arguments *)
          let sel = DN.check_dist_pars (fst d) a acc.t_vnum in
          let del = List.map (diff_expr acc) a in
          let dn  = diff_expr acc n in
          (* Guarding parameters *)
          let gpars =
            accumulate_guard_pars (SS.union acc.t_gpars dn.te_pdep) del sel in
          begin
            (match sel with
            | None ->
               printf_debug "sel = None\n"
            | Some l ->
               printf_debug "sel != None, |sel| = %d, # of true in sel = %d\n"
                 (List.length l)
                 (List.length (List.filter (fun b -> b) l)));
            (*Printf.printf "gpars: %a\n" ss_pp gpars;*)
            printf_debug "---------- abstract state ----------\n%a" pp acc;
            printf_debug "------------------------------------\n\n"
          end;
          (* Dependency of a sampled variable *)
          let deppar = SS.add parname gpars in
          (* Differentiability of the density *)
          let dndiff =
            let ndpars0 = ndpars_call_args acc fsig_diff del in
            let ndpars1 =
              if dist_diff then ndpars0 else SS.add parname ndpars0 in
            SS.union gpars ndpars1 in
          { t_pars    = SS.add parname acc.t_pars;
            t_gpars   = gpars;
            t_vpndiff = SM.add x gpars acc.t_vpndiff;
            t_vpdep   = SM.add x deppar acc.t_vpdep;
            t_dpndiff = SS.union dndiff acc.t_dpndiff;
            t_vfinfo  = SM.remove x acc.t_vfinfo;
            t_voinfo  = SM.remove x acc.t_voinfo;
            t_vnum    = DN.sample x d acc.t_vnum }
      | Sample (_ (* ? *), n (* S *), d, a (* E1, E2 *), o_opt (* E0 *)) ->
          (* XR: for an observation, I do not think that x should change;
           *     thus we should not lose precision in it/modify it *)
          (* HY: This part is unsound. The sample statement becomes observe, when it
           *     is not the case that o_opt = Some o and o represents None in Python.
           *     Currently, we assume that the object tracking part is 100% accurate
           *     as far as this check is concerned, so that we can detect the case
           *     simply by checking o_opt is Some o with o != O_Nil. *)
          let o = get_obs acc o_opt in
          let dist_diff, fsig_diff = diff_dk_sig (fst d) in
          (* Arguments *)
          let sel = DN.check_dist_pars (fst d) a acc.t_vnum in
          let del = List.map (diff_expr acc) a in
          let d_o = diff_expr acc o in
          (* Guarding parameters *)
          let gpars = accumulate_guard_pars acc.t_gpars del sel in
          begin
            (match sel with
            | None ->
               printf_debug "sel = None\n"
            | Some l ->
               printf_debug "sel != None, |sel| = %d, # of true in sel = %d\n"
                 (List.length l)
                 (List.length (List.filter (fun b -> b) l)));
            printf_debug "---------- abstract state ----------\n%a" pp acc;
            printf_debug "------------------------------------\n\n"
          end;
          (* Differentiability of the density *)
          let ndensdiff =
            if dist_diff then
              SS.union d_o.te_pndiff (ndpars_call_args acc fsig_diff del)
            else
              SS.union d_o.te_pdep (ndpars_call_args acc fsig_diff del) in
          { acc with
            t_gpars   = gpars;
            t_dpndiff = SS.union ndensdiff acc.t_dpndiff }
    let rec do_stmt acc com =
      match com with
      | Atomic ac ->
          do_acmd acc ac
      | If (e, b0, b1) ->
          let cgpars = acc.t_gpars in
          let d = diff_expr acc e in
          let not_e = (UOp (Not, e)) in
          let acc = { acc with t_gpars = SS.union acc.t_gpars d.te_pdep } in
          let acc0 = guard e acc in
          let acc1 = guard not_e acc in
          if (is_bot acc0 || imply acc not_e) then
            do_block acc1 b1
          else if (is_bot acc1 || imply acc e) then
            do_block acc0 b0
          else
            let acc0 = do_block acc0 b0 in
            let acc1 = do_block acc1 b1 in
            if not (SS.equal acc0.t_pars acc1.t_pars) then
              printf_debug
                "IF: branches disagree on parameters (ok if constant guard)\n";
            if !debug then
              Printf.printf "IF: %s\n%a\n"
                (if d.te_pdep = SS.empty then "precise" else "imprecise")
                pp_stmt com;
            let acc = t_union acc0 acc1 in
            { acc with t_gpars = cgpars }
      | For (Name v, e, b0) ->
          (*let range = Ir_util.range_info e in
          let init, test, inc = Ir_util.cond_of_for_loop v range in
          let acc = do_stmt acc init in*)
          let d = diff_expr acc e in
          if d.te_pdep != SS.empty then failwith "TODO,for"
          else
            let rec iter accv accin =
              flush stdout;
              (*let accout = do_stmt (do_block (guard test accin) b0) inc in*)
              let accout = do_block accin b0 in
              let accj = t_union accv accout in
              printf_debug "iter (%b)\n" (t_equal accv accj);
              if t_equal accv accj then accj
              else iter accj accout in
            (*guard (UOp (Not, test)) (iter acc acc)*)
            iter acc acc
      | For (_, _, b0) ->
          Printf.printf "TODO,for,other index\n";
          do_block acc b0
      | While (_, b0) ->
          Printf.printf "TODO,while\n";
          do_block acc b0
      | With (l, b0) ->
          (* If the with items use any differentiation parameter, we give up *)
          let b =
            List.fold_left
              (fun b -> function
                | e, None -> b && (diff_expr acc e).te_pdep != SS.empty
                | e, Some o ->
                    b && (diff_expr acc e).te_pdep != SS.empty
                      && (diff_expr acc o).te_pdep != SS.empty
              ) true l in
          if b then
            Printf.printf "TODO,with,expression depending on parameters\n";
          do_block acc b0
      | Break | Continue ->
          (* For now, we do not handle these;
           * serious reasoning over complex control flow needed *)
          failwith "TODO,break/continue\n"
    and do_block acc =
      List.fold_left do_stmt acc

    let diff_params ir =
      do_block { t_pars    = SS.empty ;
                 t_vpndiff = SM.empty ;
                 t_vpdep   = SM.empty ;
                 t_gpars   = SS.empty ;
                 t_dpndiff = SS.empty ;
                 t_vfinfo  = SM.empty ;
                 t_voinfo  = SM.empty ;
                 t_vnum    = DN.top } ir

  end


(** Analysis main function wrapper
 ** Inputs:
 ** - a domain
 ** - a file name
 ** Outputs:
 ** - set of parameters for which density is differentiable *)
let analyze (dnum: ad_num) (goal: diff_prop) (f: string) (verbose: bool): SS.t * SS.t =
  debug := verbose;
  (* generate SA *)
  let mod_num =
    match dnum with
    | AD_sgn -> (module DN_signs: DOM_NUM)
    | AD_box -> (module DN_box:   DOM_NUM)
    | AD_oct -> (module DN_oct:   DOM_NUM)
    | AD_pol -> (module DN_pol:   DOM_NUM) in
  let module DN = (val mod_num: DOM_NUM) in
  let module SA = Make( DN ) in
  (* generate ir by parsing file f. *)
  printf_debug "Analyzing for differentiability %S\n" f;
  Ir_parse.output := false ;
  let ir =
    Ir_parse.parse_code None (AI_pyfile f)
    (* replace `Sample(trgt, _, (Delta, []), [arg], None)` by `Assn(trgt, arg)`. *)
    |> Ir_util.simplify_delta_prog in
  printf_debug "Program:\n%a\n" pp_block ir;
  (* analyse property dp_goal of ir. *)
  (match goal with
   | Top -> failwith "analyze: dp_goal must not be Top"
   | _   -> dp_goal := goal);
  let abs = SA.diff_params ir in
  if !debug then SA.pp_result abs;
  (SA.get_d_diff abs, SA.get_d_ndiff abs)
