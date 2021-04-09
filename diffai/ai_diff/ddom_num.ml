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
 ** ddom_num.ml: numerical domains for continuity/differentiability analysis *)
open Ir_sig
open Lib

open Apron
open Apron_sig
open Ddom_sig

open Apron_util
open Diff_util
open Dom_util

(** Numerical domain based on constants *)
module DN_signs = 
  (struct
    let name = "signs"

    (* "Variable-Sign":
     * Maps each variable to the sign of the values stored in the
     * variable. Unmapped variables are implicitly mapped to Top.
     * Also, Top includes the possibility of a value being a
     * non-numerical value. When a variable stores a tensor, this
     * map records information about the values stored in the tensor. 
     * SD.Top means a numeric value. If not numeric, it should be unbound.
     * *)
    type t = SD.t SM.t

    (* Prtty-printing *)
    let pp = ppm SD.pp

    (* Bottom check: when returns true, definitely bottom *)
    let is_bot _ = false

    (* Lattice operations *)
    let top = SM.empty
    let join = map_join_inter (fun c0 c1 -> Some (SD.join c0 c1))
    let equal = map_equal (fun v si -> true) SD.equal

    (* Post-condition for assignments *)
    let forget = SM.remove
    let sign_expr (a: t) (e: expr): SD.t option =
      let rec aux = function
        | Nil | True | False ->
            None
        | Num (Int n) ->
            if n > 0 then Some SD.Plus
            else if n < 0 then Some SD.Minus
            else Some SD.Top
        | Num (Float f) ->
            if f > 0.0 then Some SD.Plus
            else if f < 0.0 then Some SD.Minus
            else Some SD.Top
        | Name x ->
            begin
              try Some (SM.find x a) with Not_found -> None
            end
        | UOp (uop, e0) ->
            bind_opt (aux e0) (fun sg0 ->
              SD.do_uop uop sg0)
        | BOp (bop, e0, e1) ->
            bind_opt (aux e0) (fun sg0 ->
            bind_opt (aux e1) (fun sg1 ->
              SD.do_bop bop sg0 sg1))
        | Comp (cop, e0, e1) ->
            bind_opt (aux e0) (fun sg0 ->  
            bind_opt (aux e1) (fun sg1 ->  
              SD.do_cop cop sg0 sg1))
        | List [] ->
            Some SD.Top
        | List (_ :: _ as el) ->
            let sgl = List.map aux el in
            let f sg_opt_acc sg_opt_cur =
              bind_opt sg_opt_acc (fun sg_acc ->
              bind_opt sg_opt_cur (fun sg_cur ->
                Some (SD.join sg_acc sg_cur))) in
            List.fold_left f (Some SD.Bot) sgl
        | Dict _ | Str _ | StrFmt _ ->
            None in 
      aux e
    let assign (x: string) (e: expr) (a: t): t =
      match sign_expr a e with
      | None -> forget x a 
      | Some se -> SM.add x se a
    let heavoc (x: string) (c: constr) (a: t): t =
      match c with
      | C_Pos -> SM.add x SD.Plus a
      | C_Neg -> SM.add x SD.Minus a
      | C_Num -> SM.add x SD.Top a


    (* Operation on primitive-function call x=f(el) *)
    let call_prim (x: string) (f: string) (el: expr list) (a: t): t =
      match f, el with
      | "torch.ones", _ ->
          SM.add x SD.CInter a
      | "F.softplus", _ 
      | "torch.exp", _ ->
          SM.add x SD.Plus a
      | "torch.sigmoid", _ ->
          SM.add x SD.OInter a
      | "torch.rand", _ ->
          SM.add x SD.Top a
      | "access_with_index", e :: _ 
      | "nn.Parameter", e :: _ 
      | "torch.FloatTensor", e :: _
      | "torch.LongTensor", e :: _
      | "torch.reshape", e :: _ 
      | "torch.squeeze", e :: _ 
      | "torch.tensor", e :: _ 
      | "torch.Tensor.detach", e :: _ ->
          begin
            match sign_expr a e with
            | None -> forget x a
            | Some se -> SM.add x se a
          end
      | "torch.matmul", e0 :: e1 :: _ ->
          begin
            match (sign_expr a e0), (sign_expr a e1) with
            | None, _ | _, None ->
                forget x a
            | Some SD.Bot, _ | _, Some SD.Bot -> 
                SM.add x SD.Bot a
            | Some SD.Plus, Some SD.Plus | Some SD.Minus, Some SD.Minus -> 
                SM.add x SD.Plus a
            | Some SD.Plus, Some SD.Minus | Some SD.Minus, Some SD.Plus ->
                SM.add x SD.Minus a
            | _ -> 
                SM.add x SD.Top a
          end
      | _, _ -> 
          forget x a

    (* Operation on an object call x=(c())(el) for an object constructor c *)
    let call_obj (x: string) (c: string) (el: expr list) (a: t): t =
      match c, el with
      | "nn.Sigmoid", _ | "nn.Softmax", _ ->
          SM.add x SD.OInter a
      | "nn.Softplus", _ ->
          SM.add x SD.Plus a
      | "nn.Linear", _ ->
          SM.add x SD.Top a
      | _, _ -> 
          forget x a

    (* Operations on distributions *)
    let sample (x: string) (d: dist) (a: t): t =
      let dist_sign = SD.do_dist (fst d) in
      SM.add x dist_sign a
    let check_dist_pars (d: dist_kind) (el: expr list) (a: t)
        : bool list option =
      let fsig_dom =
        match d with
        | Normal              -> [ Some SD.Top  ; Some SD.Plus ]
        | Exponential         -> [ Some SD.Plus ]
        | Gamma               -> [ Some SD.Plus ; Some SD.Plus ]
        | Beta                -> [ Some SD.Plus ; Some SD.Plus ]
        | Uniform _           -> [ Some SD.Bot  ; Some SD.Bot (* first param < second param *)]
        | Dirichlet _         -> [ Some SD.Plus   (* (0,inf)^n for n >= 2 *)]
        | Poisson             -> [ Some SD.Plus   (* (0,inf) *)]
        | Categorical _       -> [ Some SD.Plus   (* (0,inf)^n for n >= 2 *)]
        | Bernoulli           -> [ Some SD.CInter (* [0,1] *)]
        | OneHotCategorical _ -> [ Some SD.Plus   (* (0,inf)^n for n >= 2 *)]
        | Delta               -> [ None ]
        | Subsample (_, _)    -> [ Some SD.Bot  ; Some SD.Bot  ] in
      let leq sg_opt0 sg_opt1 =
        match sg_opt0, sg_opt1 with
        | _, None -> true
        | None, _ -> false
        | Some sg0, Some sg1 -> SD.leq sg0 sg1 in
      try Some (List.map2 leq (List.map (sign_expr a) el) fsig_dom)
      with Invalid_argument _ -> None

    let imply (a: t) (e_in: expr) : bool = 
      let negate_cop op = 
        match op with
        | Eq    -> NotEq
        | NotEq -> Eq
        | Lt    -> Gt
        | LtE   -> GtE
        | Gt    -> Lt
        | GtE   -> LtE
        | Is    -> NotIs
        | NotIs -> Is in
      let rec move_not polarity e =
        match e with
        | UOp (Not, e0) -> 
            move_not (not polarity) e0
        | UOp _ -> 
            if polarity then e else UOp (Not, e)
        | BOp (And, e0, e1) ->
            let e0_neg = move_not polarity e0 in
            let e1_neg = move_not polarity e1 in
            let op_neg = if polarity then And else Or in 
            BOp (op_neg, e0_neg, e1_neg)
        | BOp (Or, e0, e1) ->
            let e0_neg = move_not polarity e0 in
            let e1_neg = move_not polarity e1 in
            let op_neg = if polarity then Or else And in 
            BOp (op_neg, e0_neg, e1_neg)
        | BOp _ -> 
            if polarity then e else UOp (Not, e)
        | Comp (cop, e0, e1) ->
            if polarity then e else Comp (negate_cop cop, e0, e1)
        | _ -> 
            if polarity then e else UOp (Not, e) in
      let rec aux e =
        match e with
        | UOp (Not, e0) ->
            begin
              match aux e0 with
              | Some true -> Some false
              | Some false -> Some true
              | None -> None
            end
        | UOp _ ->
            None
        | BOp (And, e0, e1) ->
            begin
              match aux e0, aux e1 with
              | Some true, Some true -> Some true
              | Some false, Some _ | Some _, Some false -> Some false
              | _ -> None
            end
        | BOp (Or, e0, e1) ->
            begin
              match aux e0, aux e1 with
              | Some true, Some _ | Some _, Some true -> Some true
              | Some false, Some false -> Some false
              | _ -> None
            end
        | BOp _ ->
            None
        | Comp (Lt, Num (Float 0.), Num (Float 0.)) 
        | Comp (Gt, Num (Float 0.), Num (Float 0.)) ->
            Some false
        | Comp (LtE, Num (Float 0.), Num (Float 0.)) 
        | Comp (GtE, Num (Float 0.), Num (Float 0.)) ->
            Some true
        | Comp (Lt, Num (Float 0.), e0) 
        | Comp (Gt, e0, Num (Float 0.)) -> 
            begin
               match sign_expr a e0 with
               | None -> None
               | Some SD.Plus -> Some true
               | Some SD.Minus -> Some false
               | _ -> None
            end
        | Comp (Lt, e0, Num (Float 0.)) 
        | Comp (Gt, Num (Float 0.), e0) -> 
            begin
               match sign_expr a e0 with
               | None -> None
               | Some SD.Minus -> Some true
               | Some SD.Plus -> Some false
               | _ -> None
            end
        | Comp (Lt, e0, e1) | Comp (LtE, e0, e1) 
        | Comp (Gt, e1, e0) | Comp (GtE, e1, e0) ->
            begin
              match sign_expr a e0, sign_expr a e1 with
              | None, _ | _, None -> None
              | Some SD.Minus, Some SD.Plus -> Some true
              | Some SD.Plus, Some SD.Minus -> Some false
              | _ -> None
            end
        | Comp (Is, Nil, Nil) ->
            Some true
        | Comp (Is, e0, Nil) | Comp (Is, Nil, e0) ->
            begin
              match sign_expr a e0 with
              | Some SD.Plus | Some SD.Minus | Some SD.Top -> Some false
              | _ -> None
            end
        | Comp (NotIs, Nil, Nil) ->
            Some false
        | Comp (NotIs, e0, Nil) | Comp (NotIs, Nil, e0) ->
            begin
              match sign_expr a e0 with
              | Some SD.Plus | Some SD.Minus | Some SD.Top -> Some true
              | _ -> None
            end
        | Comp (NotEq, Num (Float 0.), Num (Float 0.)) ->
            Some false
        | Comp (NotEq, e0, Num (Float 0.))
        | Comp (NotEq, Num (Float 0.), e0) ->
            begin
              match sign_expr a e0 with
              | Some SD.Minus | Some SD.Plus -> Some true
              | _ -> None
            end
        | _ ->
            None in
      match aux (move_not true e_in) with
      | Some true -> true
      | _ -> false

    (* Post-condition for condition test *)
    let guard (e: expr) (a: t): t = a
  end: DOM_NUM)

(** Apron domain generator *)
module DN_apron_make = functor (M: APRON_MGR) ->
  (struct
    let name = Printf.sprintf "Apron<%s>" M.module_name

    (* Abstraction of the numerical state (variables -> values) *)
    module A = Apron.Abstract1
    let man = M.man

    (* Abstract values:
     * - an enviroment
     * - and a conjunction of constraints in Apron representation (u) *)
    type t = M.t A.t

    (* Prtty-printing *)
    let buf_t (buf: Buffer.t) (t: t): unit =
      Printf.bprintf buf "%a" buf_linconsarray (A.to_lincons_array man t)
    let pp = buf_to_channel buf_t

    (* Bottom check: when returns true, definitely bottom *)
    let is_bot = A.is_bottom man

    (* Lattice operations *)
    let top: t =
      let env_empty = Environment.make [| |] [| |] in
      A.top man env_empty
    let join: t -> t -> t = A.join man
    let equal: t -> t -> bool = A.is_eq man

    (* Post-condition for assignments *)
    let forget (x: string) (t: t): t =
      let var = make_apron_var x in
      A.forget_array man t [| var |] false
    let assign (x: string) (e: expr) (t: t): t =
      (* convert the expression to Apron IR *)
      let lv = make_apron_var x
      and rv = Ir_util.make_apron_expr (A.env t) e in
      (* perform the Apron assignment *)
      A.assign_texpr_array man t [| lv |] [| rv |] None
    let heavoc (x: string) (c: constr) (t: t): t =
      Printf.printf
        "[TODO] unimplemented heavoc called in Apron and handled as forget";
      forget x t

    (* Preparation of a condition for Apron *)
    let make_condition (e: expr) (t: t): Tcons1.t =
      Ir_util.make_apron_cond (A.env t) e

    (* Post-condition for condition test *)
    let guard (e: expr) (t: t): t =
      let eacons = Tcons1.array_make (A.env t) 1 in
      Tcons1.array_set eacons 0 (make_condition e t);
      let t = A.meet_tcons_array man t eacons in
      (* TODO: bottom reduction *)
      t

    (* Satisfiability for condition formula *)
    let sat (e: expr) (t: t): bool =
      let env = A.env t in
      let ce = Ir_util.make_apron_cond env e in
      A.sat_tcons man t ce

    (* Operation on primitive-function call x=f(el) *)
    let call_prim (x: string) (f: string) (el: expr list) (a: t): t =
      match f with
      | _ -> forget x a

    (* Operation on an object call x=(c())(el) for an object constructor c *)
    let call_obj (x: string) (c: string) (el: expr list) (a: t): t =
      match c with
      | _ -> forget x a

    (* Operations on distributions *)
    let sample (x: string) (d: dist) (t: t): t =
      let t = forget x t in
      let lcons =
        match fst d with
        | Normal
        | Uniform None
        | Poisson
        | Categorical _
        | Bernoulli
        | OneHotCategorical _
        | Delta
        | Subsample (_, _) ->
            [ ]
        | Gamma
        | Beta
        | Dirichlet _ ->
            [ Comp (GtE, Name x, Num (Float 0.)) ]
        | Exponential ->
            [ Comp (GtE, Name x, Num (Float 0.)) ;
              Comp (LtE, Name x, Num (Float 1.)) ]
        | Uniform (Some (a, b)) ->
            [ Comp (GtE, Name x, Num (Float a)) ;
              Comp (LtE, Name x, Num (Float b)) ] in
      List.fold_left (fun t c -> guard c t) t lcons
    let check_dist_pars (d: dist_kind) (el: expr list) (t: t)
        : bool list option =
      let ollsat =
        let wrong = Comp (LtE, Num (Float 1.), Num (Float 0.)) in
        let wrong_list = List.map (fun _ -> [ wrong ]) el in
        match d, el with
        | Normal, [ e0 ; e1 ] ->
            Some [ [ ] ; [ Comp (Gt, e1, Num (Float 0.)) ] ]
        | Exponential, [ e0 ] ->
            Some [ [ Comp (Gt, e0, Num (Float 0.)) ] ]
        | Gamma, [ e0 ; e1 ] ->
            Some [ [ Comp (Gt, e0, Num (Float 0.)) ] ;
                   [ Comp (Gt, e1, Num (Float 0.)) ] ]
        | Uniform _, [ e0 ; e1 ] ->
            Some [ [ Comp (Lt, e0, e1) ] ;
                   [ Comp (Lt, e0, e1) ] ]
        | Dirichlet _, [ e0 ] ->
            Some [ [ Comp (Gt, e0, Num (Float 0.)) ] ]
        | Dirichlet _, _ ->
            Some wrong_list
        | Poisson, [ e0 ] ->
            Some [ [ Comp (Gt, e0, Num (Float 0.)) ] ]
        | Categorical _, [ e0 ] ->
            Some [ [ Comp (Gt, e0, Num (Float 0.)) ] ]
        | Categorical _, _ ->
            Some wrong_list
        | Bernoulli, _ -> (* false; to check *)
            Some wrong_list
        | OneHotCategorical _, _ -> (* false; to check *)
            Some wrong_list
        | Delta, [ _ ] ->
            Some [ [ ] ]
        | Subsample _, [ e0 ; e1 ] ->
            Some [ [ Comp (GtE, e0, e1) ; Comp (Gt, e0, Num (Float 0.)) ] ;
                   [ Comp (GtE, e0, e1) ; Comp (Gt, e0, Num (Float 0.)) ] ]
        | Subsample (_, _), _ ->
            Some wrong_list
        | _, _ -> None in
      let f llsat =
        List.map
          (fun lsat ->
            List.fold_left
              (fun acc e ->
                acc && sat e t
              ) true lsat
          ) llsat in
      option_map f ollsat
    let imply (t: t) (e: expr) : bool = sat e t
  end: DOM_NUM)

(** Apron domain instances *)
module DN_box = DN_apron_make( Apron_util.PA_box )
module DN_oct = DN_apron_make( Apron_util.PA_oct )
module DN_pol = DN_apron_make( Apron_util.PA_pol )
