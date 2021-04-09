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
 ** batch_diff.ml: entry point for batch execution of diff analysis *)
open Analysis_sig
open Ai_diff
open Lib

(** Type for regression test data *)
(* type for regression test datum *)
type rt_entry =
    { (* regression test name *)
      rte_name:        string ;
      (* file to analyze *)
      rte_filename:    string ;
      (* !!! TODO: add optional overriding abstract domain info *)
      (* WL: unused.
      (* expected result:
       *   None:   no regression information available;
       *   Some l: we expect differentiability wrt the pars in l *)
      rte_ddiffvars:   string list option ;
       *)
      (* expected result:
       *   None: no regression information available;
       *   Some [(s_1, dp_1);...]: we expect that for each dp : diff_prop,
       *     the density is dp w.r.t. {s_i : dp_i <= dp}. *)
      rte_diffinfo: (string * diff_prop) list option; }

(* type for regression test data *)
type rt_table = rt_entry list

(** Regression test data *)
let tests_diff: rt_table =
  [
    (* Complexity of examples:
     *   air                    (has loops and user-defined funcs)
     *   >> dmm                 (has loops but no user-defined funcs)
     *   >> lda ~ sgdef ~ ssvae (has no loops and no user-defined funcs)
     *   > br ~ csis ~ vae      (shorter LoC) *)

    (* air *)
    { rte_name      = "air model";
      rte_filename  = "../srepar/srepar/examples/air/model.py";
      rte_diffinfo  =
        let params = [
            ("decode_l1", Lips); (* applied to F.relu(...). *)
            ("decode_l2", Diff);
          ] in
        let rvars = [
            ("data",       Top);  (* subsample distribution. *)
            ("z_pres_{}",  Top);  (* discrete distribution. *)
            ("z_where_{}", Lips); (* applied to F.grid_sample(_, ...). *)
            ("z_what_{}",  Lips); (* applied to F.relu(...). *)
            (* Reason for Lips:
             * - "z_where_{}":
             *     cur_z_where = pyro.sample(..."z_where_{}"...)
             *     out = ...cur_z_where...
             *     ...out...
             *     theta = out
             *     grid = ...theta...
             *     ... = F.grid_sample(..., grid)
             * - "z_what_{}":
             *     cur_z_what = pyro.sample(..."z_what_{}"...)
             *     ... = ...F.relu(...cur_z_what...)... *)
          ] in
        Some (params @ rvars); };
    { rte_name      = "air guide";
      rte_filename  = "../srepar/srepar/examples/air/guide.py";
      rte_diffinfo  =
        let params = [
            ("rnn",           Lips); (* applied to F.relu(...). *)
            ("bl_rnn",        Lips); (* applied to F.relu(...). *)
            ("predict_l1",    Lips); (* applied to F.relu(...). *)
            ("predict_l2",    Diff);
            ("encode_l1",     Lips); (* applied to F.relu(...). *)
            ("encode_l2",     Diff);
            ("bl_predict_l1", Lips); (* applied to F.relu(...). *)
            ("bl_predict_l2", Diff);
            ("h_init",        Lips); (* applied to F.relu(...). *)
            ("c_init",        Lips); (* applied to F.relu(...). *)
            ("z_where_init",  Lips); (* applied to F.relu(...). *)
            ("z_what_init",   Lips); (* applied to F.relu(...). *)
            ("bl_h_init",     Lips); (* applied to F.relu(...). *)
            ("bl_c_init",     Lips); (* applied to F.relu(...). *)
            (* Reason for Lips:
             * - "rnn", "h_init", "c_init", "z_where_init", "z_what_init":
             *     state_h = ...h_init...
             *     state_c = ...c_init...
             *     state_z_where = ...z_where_init...
             *     state_z_what = ...z_what_init...
             *     rnn_input = ...state_z_where...state_z_what...
             *     state_h, ... = ...rnn...state_h...state_c...rnn_input...
             *     ... = ...F.relu(...state_h...)...
             * - "bl_rnn", "bl_h_init", "bl_c_init":
             *     state_bl_h = ...bl_h_init...
             *     state_bl_c = ...bl_c_init...
             *     state_bl_h, ... = ...bl_rnn...state_bl_h...state_bl_c...
             *     ... = ...F.relu(...state_bl_h...)... *)
            (* NOTE:
             * - The parameters "bl_..." are all used to compute `baseline_value`
             *   of the sample statement for "z_pres_{}". In the original code,
             *   the value of `baseline_value` is passed to the kwarg `infer`
             *   of `pyro.sample` for "z_pres_{}", but the kwargs is ignored by
             *   our analyser on the ground that it does not affect densities.
             * - Though not affecting densities, `baseline_value` does affect
             *   the value of gradient estimate of ELBO (e.g., in SCORE estimator),
             *   in a differentiable way. In more detail, it affects the value of
             *   gradient estimate, not through densities, but through something
             *   related to control variate. And we need to make our analyser
             *   aware of this fact to ensure its soundness.
             * - Since the way that `baseline_value` is passed to `pyro.sample`
             *   involves `dict` objects in Python, our analyser does not analyse
             *   the original guide directly. Instead, the following lines are
             *   inserted to the original guide to make an equivalent effect
             *   for "bl_..." describe above:
             *     p = torch.exp(bl_value - bl_value)
             *     pyro.sample(_, Normal(p,1), obs=0)
             *   where bl_value is a computed value of `baseline_value`. In this way,
             *   "bl_..." now affects the densities directly, but the above observe
             *   statement makes a constant score, thereby not changing the behavior
             *   of the original guide. *)
          ] in
        let rvars = [
            ("data",       Top);  (* subsample distribution. *)
            ("z_pres_{}",  Top);  (* discrete distribution. *)
            ("z_where_{}", Top);  (* applied to F.relu(...) and _/(...). *)
            ("z_what_{}",  Lips); (* applied to F.relu(...). *)
            (* Reason for Lips:
             * - "z_where_{}", "z_what_{}":
             *     cur_z_where = pyro.sample(..."z_where_{}"...)
             *     cur_z_what = pyro.sample(..."z_what_{}"...)
             *     state_z_where = cur_z_where
             *     state_z_what = cur_z_what
             *     [next iteration of a loop]
             *     rnn_input = ...state_z_where...state_z_what...
             *     state_h, ... = ...rnn_input...
             *     ... = ...F.relu(...state_h...)... *)
            (* NOTE:
             * - diff_prop of "z_where_{}" is Top, not Lips,
             *   since it is used as a denomiator of a division:
             *     cur_z_where = pyro.sample("z_where_{}"..., Normal...)
             *     out = ...cur_z_where...
             *     out = out / (...cur_z_where...)
             * - [Q] does this produce any NaN? If so, pointing this out would be helpful. *)
          ] in
        Some (params @ rvars); };

    (* br *)
    { rte_name      = "br model";
      rte_filename  = "../srepar/srepar/examples/br/model.py";
      rte_diffinfo  =
        let params = [] in
        let rvars = [
            ("a",     Diff);
            ("bA",    Diff);
            ("bR",    Diff);
            ("bAR",   Diff);
            ("sigma", Diff);
          ] in
        Some (params @ rvars); };
    { rte_name      = "br guide";
      rte_filename  = "../srepar/srepar/examples/br/guide.py";
      rte_diffinfo  =
        let params = [
            ("a_loc",         Diff);
            ("a_scale",       Diff);
            ("sigma_loc",     Diff);
            ("weights_loc",   Diff);
            ("weights_scale", Diff);
          ] in
        let rvars = [
            ("a",     Diff);
            ("bA",    Diff);
            ("bR",    Diff);
            ("bAR",   Diff);
            ("sigma", Diff);
          ] in
        Some (params @ rvars); };

    (* csis *)
    { rte_name      = "csis model";
      rte_filename  = "../srepar/srepar/examples/csis/model.py";
      rte_diffinfo  =
        let params = [] in
        let rvars = [
            ("z", Diff);
          ] in
        Some (params @ rvars); };
    { rte_name      = "csis guide";
      rte_filename  = "../srepar/srepar/examples/csis/guide.py";
      rte_diffinfo  =
        let params = [
            ("first",  Lips); (* applied to nn.ReLU()(...). *)
            ("second", Lips); (* applied to nn.ReLU()(...). *)
            ("third",  Lips); (* applied to nn.ReLU()(...). *)
            ("fourth", Lips); (* applied to nn.ReLU()(...). *)
            ("fifth",  Diff);
          ] in
        let rvars = [
            ("z", Diff);
          ] in
        Some (params @ rvars); };

    (* dmm *)
    { rte_name      = "dmm model";
      rte_filename  = "../srepar/srepar/examples/dmm/model.py";
      rte_diffinfo  =
        let params = [
            ("e_lin_z_to_hidden",               Lips); (* applied to nn.ReLU()(...). *)
            ("e_lin_hidden_to_hidden",          Lips); (* applied to nn.ReLU()(...). *)
            ("e_lin_hidden_to_input",           Diff);
            ("t_lin_gate_z_to_hidden",          Lips); (* applied to nn.ReLU()(...). *)
            ("t_lin_gate_hidden_to_z",          Diff);
            ("t_lin_proposed_mean_z_to_hidden", Lips); (* applied to nn.ReLU()(...). *)
            ("t_lin_proposed_mean_hidden_to_z", Lips); (* applied to nn.ReLU()(...). *)
            ("t_lin_sig",                       Diff);
            ("t_lin_z_to_loc",                  Diff);
            (* Reason for Lips:
             * - "t_lin_proposed_mean_hidden_to_z":
             *     proposed_mean = ...(t_lin_proposed_mean_hidden_to_z)...
             *     ... = ...t_relu(proposed_mean)... *)
          ] in
        let rvars = [
            ("z_{}", Lips); (* applied to nn.ReLU()(...). *)
            (* Reason for Lips:
             * - "z_{}":
             *     z_t = pyro.sample(..."z_{}"...)
             *     ... = e_relu(...z_t...) *)
          ] in
        Some (params @ rvars); };
    { rte_name      = "dmm guide";
      rte_filename  = "../srepar/srepar/examples/dmm/guide.py";
      rte_diffinfo  =
        let params = [
            ("c_lin_z_to_hidden",     Diff);
            ("c_lin_hidden_to_loc",   Diff);
            ("c_lin_hidden_to_scale", Diff);
            ("rnn",                   Lips); (* has `nonlinearity`="relu". *)
          ] in
        let rvars  = [
            ("z_{}", Diff);
          ] in
        Some (params @ rvars); };

    (* lda *)
    { rte_name      = "lda model";
      rte_filename  = "../srepar/srepar/examples/lda/model.py";
      rte_diffinfo  =
        let params = [] in
        let rvars  = [
            ("topic_weights", Diff);
            ("topic_words",   Diff);
            ("doc_topics",    Diff);
            ("word_topics",   Top); (* discrete distribution. *)
          ] in
        Some (params @ rvars); };
    { rte_name      = "lda guide";
      rte_filename  = "../srepar/srepar/examples/lda/guide.py";
      rte_diffinfo  =
        let params = [
            ("layer1",                  Diff);
            ("layer2",                  Diff);
            ("layer3",                  Diff);
            ("topic_weights_posterior", Diff);
            ("topic_words_posterior",   Diff);
          ] in
        let rvars  = [
            ("topic_weights", Diff);
            ("topic_words",   Diff);
            ("documents",     Top); (* subsample distribution. *)
            (* ("doc_topics",    Top); *)
            (* NOTE:
             * - We do not a latent variable sampled from Dirac as a parmaeter
             *   and regard it just a named assigned variable (see ASSUMPTION 4 in
             *   `whitebox/refact/ai_diff/ai_diff.ml`).
             * - So commented out "doc_topics" in the above oracle for guide
             *   because it is such a Dirac-sampled variable. *)
          ] in
        Some (params @ rvars); };

    (* sgdef *)
    { rte_name      = "sgdef model";
      rte_filename  = "../srepar/srepar/examples/sgdef/model.py";
      rte_diffinfo  =
        let params = [] in
        let rvars  = [
            ("w_top",    Diff);
            ("w_mid",    Diff);
            ("w_bottom", Diff);
            ("z_top",    Diff);
            ("z_mid",    Diff);
            ("z_bottom", Diff);
          ] in
        Some (params @ rvars); };
    { rte_name      = "sgdef guide";
      rte_filename  = "../srepar/srepar/examples/sgdef/guide.py";
      rte_diffinfo  =
        let params = [
            ("log_alpha_w_q_top",    Diff);
            ("log_mean_w_q_top",     Diff);
            ("log_alpha_w_q_mid",    Diff);
            ("log_mean_w_q_mid",     Diff);
            ("log_alpha_w_q_bottom", Diff);
            ("log_mean_w_q_bottom",  Diff);
            ("log_alpha_z_q_top",    Diff);
            ("log_mean_z_q_top",     Diff);
            ("log_alpha_z_q_mid",    Diff);
            ("log_mean_z_q_mid",     Diff);
            ("log_alpha_z_q_bottom", Diff);
            ("log_mean_z_q_bottom",  Diff);
          ] in
        let rvars  = [
            ("w_top",    Diff);
            ("w_mid",    Diff);
            ("w_bottom", Diff);
            ("z_top",    Diff);
            ("z_mid",    Diff);
            ("z_bottom", Diff);
          ] in
        Some (params @ rvars); };

    (* ssvae *)
    { rte_name      = "ssvae model";
      rte_filename  = "../srepar/srepar/examples/ssvae/model.py";
      rte_diffinfo  =
        let params = [
            ("decoder_fst", Diff);
            ("decoder_snd", Diff);
          ] in
        let rvars  = [
            ("z", Diff);
            ("y", Top); (* discrete distribution. *)
            (* NOTE:
             * - "y" is sampled if given `ys` is `None`, and observed otherwise.
             *   Since there is a trace where "y" is sampled, our analyser adds "y"
             *   to parameters. *)
          ] in
        Some (params @ rvars); };
    { rte_name      = "ssvae guide";
      rte_filename  = "../srepar/srepar/examples/ssvae/guide.py";
      rte_diffinfo  =
        let params = [
            ("encoder_y_fst",  Diff);
            ("encoder_y_snd",  Diff);
            ("encoder_z_fst",  Diff);
            ("encoder_z_out1", Diff);
            ("encoder_z_out2", Diff);
          ] in
        let rvars  = [
            ("y", Top);  (* discrete distribution. *)
            ("z", Diff);
            (* NOTE:
             * - "y" is sampled if given `ys` is `None`, and do nothing otherwise.
             *   Since there is a trace where "y" is sampled, our analyser adds "y"
             *   to parameters. *)
          ] in
        Some (params @ rvars); };

    (* vae *)
    { rte_name      = "vae model";
      rte_filename  = "../srepar/srepar/examples/vae/model.py";
      rte_diffinfo  =
        let params = [
            ("decoder_fc1",  Diff);
            ("decoder_fc21", Diff);
          ] in
        let rvars  = [
            ("latent", Diff);
          ] in
        Some (params @ rvars); };
    { rte_name      = "vae guide";
      rte_filename  = "../srepar/srepar/examples/vae/guide.py";
      rte_diffinfo  =
        let params = [
            ("encoder_fc1",  Diff);
            ("encoder_fc21", Diff);
            ("encoder_fc22", Diff);
          ] in
        let rvars  = [
            ("latent", Diff);
          ] in
        Some (params @ rvars); };
  ]


(** Reg-test one example *)
let do_regtest dnum (dp_goal: diff_prop)
      (c_crash, c_ok, c_ko, c_unk, c_ko_rts) (rt: rt_entry)
    : int * int * int * int * (rt_entry list) =
  let ddiff_expected =
    let map_fun ((s,dp): string * diff_prop): string list =
      if diff_prop_leq dp dp_goal then [s] else [] in
    let option_fun (sdp_list: (string * diff_prop) list): string list =
      List.flatten (List.map map_fun sdp_list) in
    option_map option_fun rt.rte_diffinfo in
  let ddiff_analyzed =
    (* Some (analyze dnum dp_goal rt.rte_filename) in *)
    try Some (analyze dnum dp_goal rt.rte_filename false) (* debug=false *)
    with e -> None in
  match ddiff_expected, ddiff_analyzed with
  | _, None ->
      (* the analysis crashed *)
      c_crash + 1, c_ok, c_ko, c_unk, c_ko_rts
  | Some l, Some (s_res, _) ->
      (* some expected result is known *)
      let s_norm = List.fold_left (fun a i -> SS.add i a) SS.empty l in
      let _ = Printf.printf "RESULT FOR %s:\noracle:\t%a\nresult:\t%a\n\n\n"
                rt.rte_filename ss_pp s_norm ss_pp s_res in
      if SS.equal s_norm s_res then
        (* the analysis result coincides with the expected result *)
        c_crash, c_ok + 1, c_ko, c_unk, c_ko_rts
      else
        (* the analysis result does not coincide with the expected result *)
        c_crash, c_ok, c_ko + 1, c_unk, (rt :: c_ko_rts)
  | None, Some _ ->
      (* the analysis completes, but expected result was not provided *)
      c_crash, c_ok, c_ko, c_unk + 1, c_ko_rts

(** Reg-test all examples *)
let do_regtests dnum (dp_goal: diff_prop) : unit =
  let c_tot = List.length tests_diff in
  let c_crash, c_ok, c_ko, c_unk, c_ko_rts_rev =
    List.fold_left (do_regtest dnum dp_goal) (0, 0, 0, 0, []) tests_diff in
  let c_ko_rts = List.rev c_ko_rts_rev in
  Printf.printf "================================================\n";
  Printf.printf "REGRESSION TEST RESULTS:\n";
  Printf.printf "================================================\n";
  Printf.printf "Total:   %d\n" c_tot;
  Printf.printf "OK:      %d\n" c_ok;
  Printf.printf "KO:      %d\n" c_ko;
  Printf.printf "Crash:   %d\n" c_crash;
  Printf.printf "Unknown: %d\n" c_unk;
  Printf.printf "Failed cases:\n";
  List.iter (fun rt -> Printf.printf "\t%s\n" rt.rte_name) c_ko_rts;
  Printf.printf "================================================\n"


(** Apply selective reparam for one example *)
let apply_srepar (fin_name: string) (_dndiff: SS.t): unit =
  let aux (fout_name_suffix: string) (dndiff_str: string): unit =
    (* python codes to inject/match *)
    let pp_preamble chan () = (* for import srepar *)
      Printf.fprintf chan "'''\n";
      Printf.fprintf chan "Auto-generated by `whitebox/refact/batch_diff/batch_diff.ml`.\n";
      Printf.fprintf chan "'''\n";
      Printf.fprintf chan "from srepar.lib.srepar import set_no_reparam_names\n\n" in
    let pp_srepar chan (s: string) = (* for selective reparam *)
      Printf.fprintf chan "    # set up reparameterisation\n";
      Printf.fprintf chan "    set_no_reparam_names(%s)\n\n" s in
    let re_main = (* to be matched as the start of main func *)
      Str.regexp {|def[ ]+main[ ]*(.*)[ ]*:[ ]*$|} in

    (* inchan, outchan *)
    let fout_name =
      let dirname        = Filename.dirname  fin_name in
      let fin_name_extrm = Filename.basename fin_name |> Filename.remove_extension in
      let fin_name_ext   = Filename.basename fin_name |> Filename.extension in
      Filename.concat dirname
        (Printf.sprintf "%s_%s%s" fin_name_extrm fout_name_suffix fin_name_ext) in
    let fin  = Unix.openfile fin_name  [ Unix.O_RDONLY ] 0o644 in
    let fout = Unix.openfile fout_name [ Unix.O_WRONLY; Unix.O_CREAT ] 0o644 in
    let inchan  = Unix.in_channel_of_descr  fin  in
    let outchan = Unix.out_channel_of_descr fout in

    (* inject python codes *)
    Printf.fprintf outchan "%a" pp_preamble ();
    let pp_srepar_done = ref false in
    try
      while true do
        let line = input_line inchan in
        Printf.fprintf outchan "%s\n" line;
        if (not !pp_srepar_done) && (Str.string_match re_main line 0) then
          ( pp_srepar_done := true;
            Printf.fprintf outchan "%a" pp_srepar dndiff_str )
      done
    with End_of_file -> assert(!pp_srepar_done) in

  (* dndiff_str *)
  let dndiff =
    (* special handle for air/{model,guide}.py *)
    if (Filename.dirname fin_name) = "../srepar/srepar/examples/air"
    then (Printf.printf "  NOTE: 'z_where_{}' is manually removed from non-reparam'l params";
          Printf.printf " when python codes are generated!\n";
          SS.diff _dndiff (SS.singleton "z_where_{}"))
    else _dndiff in
  let dndiff_str =
    Printf.sprintf "[%s]"
      ((buf_to_string (buf_list ", " buf_string)) (SS.elements dndiff)) in

  (* inject python codes *)
  aux "ours"  dndiff_str;
  aux "score" "True";
  aux "repar" "[]"

(** Apply selective reparam for all examples *)
let apply_srepars dnum (dp_goal: diff_prop) : unit =
  let rec aux (tests: rt_table): unit =
    match tests with
    | rt1 :: rt2 :: tests_tl ->
       begin
         let (_, dndiff1) = analyze dnum dp_goal rt1.rte_filename false in
         let (_, dndiff2) = analyze dnum dp_goal rt2.rte_filename false in
         let dndiff = SS.union dndiff1 dndiff2 in
         Printf.printf "%-35s\t=>\t%a\n"
           (Filename.dirname rt1.rte_filename) ss_pp dndiff;
         apply_srepar rt1.rte_filename dndiff;
         apply_srepar rt2.rte_filename dndiff;
         aux tests_tl
       end
    | [] -> ()
    | _ -> failwith "apply_repar: tests_diff has an odd number of entries" in
  Printf.printf "\n\n";
  Printf.printf "Generating selectively reparameterised models/guides...\n\n";
  Printf.printf "==================================\n";
  Printf.printf " NON-REPARAMETERISABLE PARAMETERS \n";
  Printf.printf "==================================\n";
  aux tests_diff


(** Function to iterate the analysis *)
let main () =
  let dnum: ad_num ref = ref AD_sgn in
  let dnum_set v = Arg.Unit (fun () -> dnum := v) in
  Arg.parse [
      "-ai-box",   dnum_set AD_box,    "Num analysis, Apron, Boxes" ;
      "-ai-oct",   dnum_set AD_oct,    "Num analysis, Apron, Octagons" ;
      "-ai-pol",   dnum_set AD_pol,    "Num analysis, Apron, Polyhedra" ;
      "-ai-sgn",   dnum_set AD_sgn,    "Num analysis, Basic, Signs" ;
    ]
    (fun s -> failwith (Printf.sprintf "unbound %S" s))
    "Differentiability analysis, batch mode";
  (* WL: TODO. receive dp_goal as command-line argument. *)
  (* analyse dp_goal = Lips. *)
  do_regtests !dnum (Lips);
  (* apply selective reparam. *)
  apply_srepars !dnum (Lips)

(** Start *)
let _ = ignore (main ())
